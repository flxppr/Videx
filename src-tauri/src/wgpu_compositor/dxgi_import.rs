//! Zero-copy D3D11VA → wgpu texture import for Windows discrete GPUs.
//!
//! # Pipeline
//!
//! ```text
//! FFmpeg D3D11VA decoder
//!   └─ AVFrame.data[0] = ID3D11Texture2D*    (GPU VRAM — no copy yet)
//!   └─ AVFrame.data[1] = array_index
//!        │
//!        ▼  IDXGIResource1::CreateSharedHandle (NT handle, ~0 μs)
//!        │
//!        ▼  ID3D12Device::OpenSharedHandle     (~0 μs, same adapter bus)
//!        │
//!        ▼  wgpu::Device::create_texture_from_hal
//!             └─ two TextureViews: Plane0 (Y), Plane1 (UV)
//! ```
//!
//! The entire chain is zero-copy on modern NVIDIA/AMD discrete GPUs because
//! the D3D11 texture lives in VRAM that is accessible from D3D12 without a
//! PCIe round-trip, provided both devices share the same physical adapter.
//!
//! # Fallback
//!
//! Every step returns `Option` / `Result`. The caller must fall back to the
//! existing CPU path (`av_hwframe_transfer_data` + `queue.write_texture`) on
//! any failure so correctness is never compromised.

#![cfg(target_os = "windows")]

use windows::core::{Interface, PCWSTR};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_FEATURE_DATA_D3D11_OPTIONS, D3D11_FEATURE_DATA_D3D11_OPTIONS4,
    D3D11_FEATURE_DATA_D3D11_OPTIONS5, D3D11_FEATURE_D3D11_OPTIONS,
    D3D11_FEATURE_D3D11_OPTIONS4, D3D11_FEATURE_D3D11_OPTIONS5, D3D11_TEXTURE2D_DESC,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_DATA_D3D12_OPTIONS4, D3D12_FEATURE_D3D12_OPTIONS4, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12;
use windows::Win32::Graphics::Dxgi::IDXGIResource1;

/// Raw handles needed to import a D3D11VA frame into wgpu without a PCIe copy.
pub struct D3d11SharedFrame {
    /// DXGI NT shared handle (closed automatically on Drop).
    pub nt_handle: HANDLE,
    /// Texture array slice that contains this frame (D3D11VA array decode).
    pub array_index: u32,
    /// Decoded luma width in pixels.
    pub width: u32,
    /// Decoded luma height in pixels.
    pub height: u32,
}

// Windows NT kernel handles are process-wide and thread-safe to transfer across threads.
unsafe impl Send for D3d11SharedFrame {}
unsafe impl Sync for D3d11SharedFrame {}

impl D3d11SharedFrame {
    /// Duplicate the underlying NT kernel handle using Win32 `DuplicateHandle`.
    /// This allows a shared frame stored in the lookahead queue to produce independent
    /// handles for multiple import or presentation passes without lifetime conflicts.
    pub fn duplicate(&self) -> Option<Self> {
        if self.nt_handle.is_invalid() {
            return None;
        }
        unsafe {
            use windows::Win32::Foundation::DUPLICATE_SAME_ACCESS;
            use windows::Win32::System::Threading::GetCurrentProcess;
            let mut target_handle = HANDLE::default();
            let process = GetCurrentProcess();
            let ret = windows::Win32::Foundation::DuplicateHandle(
                process,
                self.nt_handle,
                process,
                &mut target_handle,
                0,
                false,
                DUPLICATE_SAME_ACCESS,
            );
            if ret.is_ok() && !target_handle.is_invalid() {
                Some(Self {
                    nt_handle: target_handle,
                    array_index: self.array_index,
                    width: self.width,
                    height: self.height,
                })
            } else {
                None
            }
        }
    }
}

impl Drop for D3d11SharedFrame {
    fn drop(&mut self) {
        if !self.nt_handle.is_invalid() {
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(self.nt_handle);
            }
            self.nt_handle = HANDLE::default();
        }
    }
}

fn log_d3d11_interop_contract(texture: &ID3D11Texture2D, desc: &D3D11_TEXTURE2D_DESC) {
    const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
    const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
    const D3D11_BIND_DECODER: u32 = 0x200;
    const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
    const D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX: u32 = 0x100;
    const D3D11_RESOURCE_MISC_GDI_COMPATIBLE: u32 = 0x200;
    const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;

    let shared = desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED != 0;
    let shared_nt_handle = desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE != 0;
    let keyed_mutex = desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX != 0;
    let gdi_compatible = desc.MiscFlags & D3D11_RESOURCE_MISC_GDI_COMPATIBLE != 0;
    let baseline_shape = desc.ArraySize == 1
        && desc.MipLevels == 1
        && desc.Usage.0 == 0
        && desc.BindFlags & (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET)
            == (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET)
        && desc.CPUAccessFlags == 0
        && desc.SampleDesc.Count == 1;

    log::info!(
        "[dxgi_contract] D3D11 resource descriptor: width={} height={} format={:?} mip_levels={} array_size={} usage={:?} usage_value={} bind_flags=0x{:08x} cpu_access_flags=0x{:08x} misc_flags=0x{:08x} sample_count={} sample_quality={} array_index_is_slice=true",
        desc.Width,
        desc.Height,
        desc.Format,
        desc.MipLevels,
        desc.ArraySize,
        desc.Usage,
        desc.Usage.0,
        desc.BindFlags,
        desc.CPUAccessFlags,
        desc.MiscFlags,
        desc.SampleDesc.Count,
        desc.SampleDesc.Quality,
    );
    log::info!(
        "[dxgi_contract] D3D11 resource flags: nv12={} shared={} shared_nthandle={} shared_keyed_mutex={} gdi_compatible={} bind_decoder={} bind_shader_resource={} bind_render_target={} baseline_d3d11_1_shape={} nv12_array_requires_extended_or_driver_specific_support={} nt_handle_create_semantics=expected",
        desc.Format == DXGI_FORMAT_NV12,
        shared,
        shared_nt_handle,
        keyed_mutex,
        gdi_compatible,
        desc.BindFlags & D3D11_BIND_DECODER != 0,
        desc.BindFlags & D3D11_BIND_SHADER_RESOURCE != 0,
        desc.BindFlags & D3D11_BIND_RENDER_TARGET != 0,
        baseline_shape,
        desc.Format == DXGI_FORMAT_NV12 && desc.ArraySize > 1,
    );

    let device = match unsafe { texture.GetDevice() } {
        Ok(device) => device,
        Err(error) => {
            log::warn!(
                "[dxgi_contract] D3D11 GetDevice failed; feature/tier queries unavailable: hresult=0x{:08x} error={:?}",
                error.code().0 as u32,
                error
            );
            return;
        }
    };

    let mut options = D3D11_FEATURE_DATA_D3D11_OPTIONS::default();
    match unsafe {
        device.CheckFeatureSupport(
            D3D11_FEATURE_D3D11_OPTIONS,
            &mut options as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&options) as u32,
        )
    } {
        Ok(()) => log::info!(
            "[dxgi_contract] D3D11 options: ExtendedResourceSharing={}",
            options.ExtendedResourceSharing.0 != 0,
        ),
        Err(error) => log::warn!(
            "[dxgi_contract] D3D11 options query failed: hresult=0x{:08x} error={:?}",
            error.code().0 as u32,
            error
        ),
    }

    let mut options4 = D3D11_FEATURE_DATA_D3D11_OPTIONS4::default();
    match unsafe {
        device.CheckFeatureSupport(
            D3D11_FEATURE_D3D11_OPTIONS4,
            &mut options4 as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&options4) as u32,
        )
    } {
        Ok(()) => log::info!(
            "[dxgi_contract] D3D11 options4: ExtendedNV12SharedTextureSupported={} nv12_extended_support={}",
            options4.ExtendedNV12SharedTextureSupported.0 != 0,
            options4.ExtendedNV12SharedTextureSupported.0 != 0,
        ),
        Err(error) => log::warn!(
            "[dxgi_contract] D3D11 options4 query failed: hresult=0x{:08x} error={:?}",
            error.code().0 as u32,
            error
        ),
    }

    let mut options5 = D3D11_FEATURE_DATA_D3D11_OPTIONS5::default();
    match unsafe {
        device.CheckFeatureSupport(
            D3D11_FEATURE_D3D11_OPTIONS5,
            &mut options5 as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&options5) as u32,
        )
    } {
        Ok(()) => log::info!(
            "[dxgi_contract] D3D11 shared resource tier: tier={} raw={:?}",
            options5.SharedResourceTier.0,
            options5.SharedResourceTier,
        ),
        Err(error) => log::warn!(
            "[dxgi_contract] D3D11 shared-resource-tier query failed: hresult=0x{:08x} error={:?}",
            error.code().0 as u32,
            error
        ),
    }
}

/// Return whether a D3D11 resource is eligible for the DXGI zero-copy import
/// path before an NT handle is created.  AMD's DX12 driver removes the device
/// when it opens the D3D11VA NV12 decoder-array shape, even though the feature
/// and resource-tier probes report support.  Keep this guard limited to that
/// exact decoder/shared-array shape so ordinary single-surface DXGI imports
/// remain available.
fn dxgi_zero_copy_preflight_eligible(desc: &D3D11_TEXTURE2D_DESC) -> bool {
    const D3D11_BIND_DECODER: u32 = 0x200;
    const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;

    !(desc.Format == DXGI_FORMAT_NV12
        && desc.ArraySize > 1
        && desc.BindFlags & D3D11_BIND_DECODER != 0
        && desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE != 0)
}

/// Extract a DXGI shared NT handle from an FFmpeg D3D11VA hardware `AVFrame`.
///
/// FFmpeg D3D11VA stores frames as:
///   `frame->data[0]` = `ID3D11Texture2D*` (the texture)
///   `frame->data[1]` = array index cast to pointer (for array-texture decoders)
///
/// # Safety
///
/// `frame_ptr` must be a valid, non-null `*const AVFrame` whose `hw_frames_ctx`
/// references a live D3D11VA hw-frames context.  Call this only while the
/// `AVFrame` is still in scope (i.e., before `av_frame_unref`).
pub unsafe fn extract_shared_handle(
    frame_ptr: *const ffmpeg_sys_next::AVFrame,
) -> Option<D3d11SharedFrame> {
    if frame_ptr.is_null() {
        return None;
    }

    // data[0] = ID3D11Texture2D*  (raw COM pointer, not ref-counted here)
    // data[1] = array slice index (cast to *mut u8 by FFmpeg convention)
    let texture_raw = (*frame_ptr).data[0] as *mut std::ffi::c_void;
    let array_index = (*frame_ptr).data[1] as usize as u32;

    if texture_raw.is_null() {
        return None;
    }

    // Borrow the COM pointer — do NOT call AddRef/Release; FFmpeg owns this.
    // We use windows-rs `from_raw_borrowed` which creates a non-owning borrow.
    // Bind the cast to a named local first: MSVC's stricter NLL rules reject
    // the inline temporary `&(texture_raw as *mut _)` with E0716.
    let texture_ptr = texture_raw as *mut _;
    let texture: &ID3D11Texture2D = windows::core::from_raw_borrowed(&texture_ptr)?;

    // Get DXGI resource interface so we can create an NT shared handle.
    let resource: IDXGIResource1 = texture.cast().ok()?;

    let mut desc = D3D11_TEXTURE2D_DESC::default();
    texture.GetDesc(&mut desc);
    log_d3d11_interop_contract(texture, &desc);
    if !dxgi_zero_copy_preflight_eligible(&desc) {
        log::warn!(
            "[dxgi_import] DX12 DXGI zero-copy disabled for NV12 array resource before CreateSharedHandle; using CPU-NV12 upload path to avoid device removal (array_size={} bind_flags=0x{:08x} misc_flags=0x{:08x})",
            desc.ArraySize,
            desc.BindFlags,
            desc.MiscFlags,
        );
        return None;
    }
    log::info!(
        "[dxgi_import] operation begin: CreateSharedHandle width={} height={} array_layers={} array_index={} mip_levels={} format={:?} handle_kind=NT access=DXGI_SHARED_RESOURCE_READ",
        desc.Width,
        desc.Height,
        desc.ArraySize,
        array_index,
        desc.MipLevels,
        desc.Format,
    );

    // DXGI_SHARED_RESOURCE_READ = 0x80000000
    let nt_handle: HANDLE = match resource.CreateSharedHandle(
        None,           // default security
        0x8000_0000u32, // DXGI_SHARED_RESOURCE_READ
        PCWSTR::null(), // no name
    ) {
        Ok(handle) => handle,
        Err(error) => {
            log::error!(
                "[dxgi_import] CreateSharedHandle failed: hresult=0x{:08x} error={:?}",
                error.code().0 as u32,
                error
            );
            return None;
        }
    };

    if nt_handle.is_invalid() {
        log::error!("[dxgi_import] CreateSharedHandle returned an invalid handle");
        return None;
    }

    log::info!("[dxgi_import] CreateSharedHandle succeeded handle={:?}", nt_handle);

    Some(D3d11SharedFrame {
        nt_handle,
        array_index,
        width: desc.Width,
        height: desc.Height,
    })
}

/// Imported NV12 texture with its biplanar wgpu views, ready for the YUV shader.
pub struct ImportedNv12Texture {
    /// The wgpu texture wrapping the D3D11VA VRAM surface (no PCIe copy).
    /// Kept alive as long as the views are in use.
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    /// `TextureAspect::Plane0` — luma (Y), format-compatible with `R8Unorm`.
    pub y_view: wgpu::TextureView,
    /// `TextureAspect::Plane1` — chroma (UV), format-compatible with `Rg8Unorm`.
    pub uv_view: wgpu::TextureView,
}

fn log_dx12_device_status(device: &wgpu::Device, phase: &str) {
    use wgpu::hal::api::Dx12;

    unsafe {
        device.as_hal::<Dx12, _, ()>(|hal_device| match hal_device {
            Some(hal_device) => match hal_device.raw_device().GetDeviceRemovedReason() {
                Ok(()) => log::info!(
                    "[dxgi_import] DX12 device probe: phase='{}' status=healthy",
                    phase
                ),
                Err(error) => log::error!(
                    "[dxgi_import] DX12 device probe: phase='{}' status=removed hresult=0x{:08x} error={:?}",
                    phase,
                    error.code().0 as u32,
                    error
                ),
            },
            None => log::warn!(
                "[dxgi_import] DX12 device probe: phase='{}' HAL device unavailable",
                phase
            ),
        });
    }
}

fn log_dx12_shared_resource_capability(device: &ID3D12Device, phase: &str) {
    let adapter_luid = unsafe { device.GetAdapterLuid() };
    let mut options4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
    match unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS4,
            &mut options4 as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&options4) as u32,
        )
    } {
        Ok(()) => log::info!(
            "[dxgi_contract] DX12 shared-resource capability: phase='{}' adapter_luid={:08x}:{:08x} compatibility_tier={} raw={:?} nv12_tier_support={}",
            phase,
            adapter_luid.HighPart as u32,
            adapter_luid.LowPart,
            options4.SharedResourceCompatibilityTier.0,
            options4.SharedResourceCompatibilityTier,
            options4.SharedResourceCompatibilityTier.0 >= 2,
        ),
        Err(error) => log::warn!(
            "[dxgi_contract] DX12 shared-resource-compatibility-tier query failed: phase='{}' hresult=0x{:08x} error={:?}",
            phase,
            error.code().0 as u32,
            error
        ),
    }
}

/// Open a DXGI NT shared handle via the wgpu DX12 HAL and produce biplanar views.
///
/// # Arguments
///
/// * `device` — the wgpu device (must use the DX12 backend on Windows).
/// * `shared` — handle produced by `extract_shared_handle`; this function takes
///              ownership and will close it regardless of success/failure.
///
use crate::wgpu_compositor::render_path::DxgiFailureReason;

/// Open a DXGI NT shared handle via the wgpu DX12 HAL and produce biplanar views.
///
/// # Arguments
///
/// * `device` — the wgpu device (must use the DX12 backend on Windows).
/// * `shared` — handle produced by `extract_shared_handle`; this function takes
///              ownership and will close it regardless of success/failure.
///
/// # Returns
///
/// `Ok(ImportedNv12Texture)` on success, or `Err(DxgiFailureReason)` with the
/// precise failure cause. The caller **must** fall back to the CPU upload path on error.
pub fn import_into_wgpu(
    device: &wgpu::Device,
    shared: D3d11SharedFrame,
) -> Result<ImportedNv12Texture, DxgiFailureReason> {
    use wgpu::hal::api::Dx12;

    if !device
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        log::warn!("DXGI zero-copy import skipped: device does not support TEXTURE_FORMAT_NV12");
        return Err(DxgiFailureReason::UnsupportedFormat);
    }

    let nt_handle = shared.nt_handle;
    let array_index = shared.array_index;
    let width = shared.width;
    let height = shared.height;

    log::info!(
        "[dxgi_import] begin: width={} height={} array_index={} handle={:?}",
        width,
        height,
        array_index,
        nt_handle,
    );
    log_dx12_device_status(device, "before_open_shared_handle");

    // We close nt_handle in all branches (success and failure).
    let result = unsafe {
        device.as_hal::<Dx12, _, Result<ImportedNv12Texture, DxgiFailureReason>>(|hal_device| {
            let hal_device = hal_device.ok_or(DxgiFailureReason::ImportFailed)?;

            // Get the raw ID3D12Device so we can open the DXGI shared handle.
            let d3d12_device: &ID3D12Device = hal_device.raw_device();
            log_dx12_shared_resource_capability(d3d12_device, "before_open_shared_handle");
            log::info!(
                "[dxgi_contract] handle open semantics: source=IDXGIResource1::CreateSharedHandle handle_kind=NT destination=ID3D12Device::OpenSharedHandle access=implicit_resource_open",
            );

            // Open the D3D11 texture's DXGI handle as a D3D12 resource.
            let mut d3d12_resource: Option<ID3D12Resource> = None;
            if let Err(error) = d3d12_device.OpenSharedHandle(nt_handle, &mut d3d12_resource) {
                log::error!(
                    "[dxgi_import] OpenSharedHandle failed: hresult=0x{:08x} error={:?}",
                    error.code().0 as u32,
                    error
                );
                // On the affected AMD/DX12 path, OpenSharedHandle can remove
                // the device while returning a driver-internal error.  The
                // wgpu device-lost callback is asynchronous and may still
                // report `Unknown`, so query the native device immediately at
                // the failure boundary and classify it before any wgpu
                // texture/resource operation is attempted.
                match d3d12_device.GetDeviceRemovedReason() {
                    Ok(()) => {
                        log::error!(
                            "[dxgi_import] OpenSharedHandle failed without immediate DX12 device removal; retaining ordinary import failure"
                        );
                    }
                    Err(removal_error) => {
                        log::error!(
                            "[dxgi_import] OpenSharedHandle caused DX12 device removal: open_hresult=0x{:08x} removed_hresult=0x{:08x} removal_error={:?}; DXGI zero-copy must be disabled and CPU-NV12 requires a healthy GPU context",
                            error.code().0 as u32,
                            removal_error.code().0 as u32,
                            removal_error,
                        );
                        return Err(DxgiFailureReason::DeviceLost);
                    }
                }
                return Err(DxgiFailureReason::ImportFailed);
            }
            if let Err(removal_error) = d3d12_device.GetDeviceRemovedReason() {
                log::error!(
                    "[dxgi_import] DX12 device was removed immediately after OpenSharedHandle succeeded: removed_hresult=0x{:08x} removal_error={:?}; stopping before resource inspection or wgpu wrapping",
                    removal_error.code().0 as u32,
                    removal_error,
                );
                return Err(DxgiFailureReason::DeviceLost);
            }
            log_dx12_device_status(device, "after_open_shared_handle");
            let d3d12_resource: ID3D12Resource =
                d3d12_resource.ok_or(DxgiFailureReason::InvalidTexture)?;

            // Verify the format is NV12 as expected.
            let resource_desc: D3D12_RESOURCE_DESC = d3d12_resource.GetDesc();
            log::info!(
                "[dxgi_import] opened resource: dimension={:?} format={:?} width={} height={} array_layers={} mip_levels={} sample_count={} sample_quality={}",
                resource_desc.Dimension,
                resource_desc.Format,
                resource_desc.Width,
                resource_desc.Height,
                resource_desc.DepthOrArraySize,
                resource_desc.MipLevels,
                resource_desc.SampleDesc.Count,
                resource_desc.SampleDesc.Quality,
            );
            log_dx12_device_status(device, "after_resource_desc");
            if resource_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
                || resource_desc.Format != DXGI_FORMAT_NV12
            {
                return Err(DxgiFailureReason::UnsupportedFormat);
            }

            // Invariant 1: Catch array_index out of bounds for the texture array.
            // D3D11VA decoders allocate multi-slice texture arrays (e.g. 16-32 slices).
            let array_size = resource_desc.DepthOrArraySize as u32;
            if array_index >= array_size {
                log::warn!(
                    "[dxgi_import] array_index {array_index} >= DepthOrArraySize {array_size}"
                );
                return Err(DxgiFailureReason::WrongArraySlice);
            }

            // Wrap the D3D12 resource as a wgpu HAL texture.
            // `texture_from_raw` is the wgpu 24.x DX12 HAL entry point.
            log::info!("[dxgi_import] operation begin: texture_from_raw");
            let hal_texture = <Dx12 as wgpu::hal::Api>::Device::texture_from_raw(
                d3d12_resource,
                wgpu::TextureFormat::NV12,
                wgpu::TextureDimension::D2,
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: array_size,
                },
                1,
                1,
            );
            log_dx12_device_status(device, "after_texture_from_raw");

            // Promote to wgpu::Texture.
            log::info!("[dxgi_import] operation begin: create_texture_from_hal");
            let texture = device.create_texture_from_hal::<Dx12>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("D3D11VA NV12 ZeroCopy"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: array_size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::NV12,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            );
            log_dx12_device_status(device, "after_create_texture_from_hal");

            // Plane 0 = Y (luma), sampled as R8Unorm.
            // Select the specific array slice decoded by FFmpeg.
            log::info!(
                "[dxgi_import] operation begin: create_plane_views array_index={}",
                array_index
            );
            let y_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("NV12 Y plane"),
                format: Some(wgpu::TextureFormat::R8Unorm),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::Plane0,
                base_array_layer: array_index,
                array_layer_count: Some(1),
                ..Default::default()
            });
            log_dx12_device_status(device, "after_y_plane_view");

            // Plane 1 = UV (chroma, interleaved), sampled as Rg8Unorm.
            // Select the specific array slice decoded by FFmpeg.
            let uv_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("NV12 UV plane"),
                format: Some(wgpu::TextureFormat::Rg8Unorm),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::Plane1,
                base_array_layer: array_index,
                array_layer_count: Some(1),
                ..Default::default()
            });
            log_dx12_device_status(device, "after_uv_plane_view");

            Ok(ImportedNv12Texture {
                texture,
                y_view,
                uv_view,
            })
        })
    };

    // `shared` is dropped here, calling D3d11SharedFrame::drop which closes nt_handle safely.
    // Do not probe the wgpu device after the native device has already been
    // classified as removed.  This keeps the failure path free of additional
    // use-after-device-loss activity.
    if !matches!(result, Err(DxgiFailureReason::DeviceLost)) {
        log_dx12_device_status(device, "after_import");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_nv12_shared_array_is_rejected_before_handle_creation() {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        desc.Format = DXGI_FORMAT_NV12;
        desc.ArraySize = 17;
        desc.BindFlags = 0x200;
        desc.MiscFlags = 0x800;

        assert!(!dxgi_zero_copy_preflight_eligible(&desc));
    }

    #[test]
    fn single_surface_nv12_remains_eligible() {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        desc.Format = DXGI_FORMAT_NV12;
        desc.ArraySize = 1;
        desc.BindFlags = 0x200;
        desc.MiscFlags = 0x800;

        assert!(dxgi_zero_copy_preflight_eligible(&desc));
    }

    #[test]
    fn unrelated_shared_array_shape_remains_eligible() {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        desc.Format = DXGI_FORMAT_NV12;
        desc.ArraySize = 17;
        desc.BindFlags = 0x8;
        desc.MiscFlags = 0x800;

        assert!(dxgi_zero_copy_preflight_eligible(&desc));
    }
}
