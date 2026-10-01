use super::preview_capabilities::PreviewCapabilities;
use crate::native_core::NativeGpuRuntimeStatus;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::{Adapter, Device, DeviceType, Instance, Queue};

/// Detailed metadata about the active GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedGpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver: String,
    pub driver_info: String,
    pub is_discrete: bool,
}

/// Runtime diagnostics shared by the GPU context and the native-surface path.
/// A device-loss callback is asynchronous, so the surface path must be able to
/// check this state before calling into wgpu again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLossDiagnostic {
    pub reason: String,
    pub message: String,
    pub phase: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UncapturedGpuError {
    pub message: String,
    pub phase: String,
}

#[derive(Debug, Default)]
pub struct DeviceLossState {
    lost: AtomicBool,
    phase: Mutex<String>,
    diagnostic: Mutex<Option<DeviceLossDiagnostic>>,
    first_error: Mutex<Option<UncapturedGpuError>>,
}

impl DeviceLossState {
    pub fn set_phase(&self, phase: impl Into<String>) {
        if let Ok(mut current) = self.phase.lock() {
            *current = phase.into();
        }
    }

    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    pub fn diagnostic(&self) -> Option<DeviceLossDiagnostic> {
        self.diagnostic.lock().ok().and_then(|value| value.clone())
    }

    pub fn first_error(&self) -> Option<UncapturedGpuError> {
        self.first_error.lock().ok().and_then(|value| value.clone())
    }

    fn record_loss(&self, reason: String, message: String) {
        self.lost.store(true, Ordering::Release);
        let phase = self
            .phase
            .lock()
            .map(|value| value.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        if let Ok(mut diagnostic) = self.diagnostic.lock() {
            *diagnostic = Some(DeviceLossDiagnostic {
                reason,
                message,
                phase,
            });
        }
    }

    fn record_error(&self, message: String) {
        let phase = self
            .phase
            .lock()
            .map(|value| value.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        if let Ok(mut first_error) = self.first_error.lock() {
            if first_error.is_none() {
                *first_error = Some(UncapturedGpuError { message, phase });
            }
        }
    }
}

fn device_checkpoint(
    device: &Device,
    state: &DeviceLossState,
    phase: &str,
) -> Result<(), String> {
    state.set_phase(phase);
    log::info!(
        "[gpu] checkpoint begin: phase='{}'",
        phase
    );
    let poll_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.poll(wgpu::Maintain::Wait)
    }));
    if let Err(payload) = poll_result {
        let message = if let Some(value) = payload.downcast_ref::<&str>() {
            (*value).to_string()
        } else if let Some(value) = payload.downcast_ref::<String>() {
            value.clone()
        } else {
            "wgpu device poll panicked".to_string()
        };
        log::error!(
            "[gpu] checkpoint panic: phase='{}' message='{}'",
            phase,
            message
        );
        return Err(format!("GPU checkpoint '{phase}' panicked: {message}"));
    }
    if state.is_lost() {
        if let Some(diagnostic) = state.diagnostic() {
            log::error!(
                "[gpu] first checkpoint detecting device loss: phase='{}' callback_phase='{}' reason={} message={}",
                phase,
                diagnostic.phase,
                diagnostic.reason,
                diagnostic.message
            );
            return Err(format!(
                "GPU device lost at checkpoint '{phase}' (callback phase={}, reason={}, message={})",
                diagnostic.phase, diagnostic.reason, diagnostic.message
            ));
        }
        return Err(format!("GPU device lost at checkpoint '{phase}'"));
    }
    if let Some(error) = state.first_error() {
            log::warn!(
                "[gpu] first uncaptured error observed at phase='{}': {}",
                error.phase,
                error.message
            );
    }
    log::info!(
        "[gpu] checkpoint passed: phase='{}'",
        phase
    );
    Ok(())
}

fn device_feature_attempts(available_features: wgpu::Features) -> Vec<wgpu::Features> {
    let optional_features =
        wgpu::Features::TEXTURE_FORMAT_16BIT_NORM | wgpu::Features::TEXTURE_FORMAT_NV12;
    let full_features = available_features & optional_features;
    let mut attempts = vec![full_features];
    let without_hdr = full_features & !wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let without_nv12 = full_features & !wgpu::Features::TEXTURE_FORMAT_NV12;
    for candidate in [without_hdr, without_nv12, wgpu::Features::empty()] {
        if !attempts.iter().any(|attempt| *attempt == candidate) {
            attempts.push(candidate);
        }
    }
    attempts
}

#[cfg(target_os = "windows")]
static SELECTED_DXGI_ADAPTER_INDEX: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(-1);

/// Query the DXGI adapter index that matches the active discrete wgpu GPU.
/// Used by FFmpeg D3D11VA hardware initialization to bind to the same physical adapter.
#[cfg(target_os = "windows")]
pub fn get_selected_dxgi_adapter_index() -> Option<u32> {
    let idx = SELECTED_DXGI_ADAPTER_INDEX.load(std::sync::atomic::Ordering::Relaxed);
    if idx >= 0 {
        Some(idx as u32)
    } else {
        None
    }
}

#[cfg(target_os = "windows")]
static DXGI_RUNTIME_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn mark_dxgi_runtime_disabled() {
    #[cfg(target_os = "windows")]
    DXGI_RUNTIME_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub fn is_dxgi_runtime_enabled() -> bool {
    #[cfg(target_os = "windows")]
    {
        if std::env::var("CLYPRA_DISABLE_DXGI").as_deref() == Ok("1")
            || std::env::var("CLYPRA_DISABLE_DXGI_ZERO_COPY").as_deref() == Ok("1")
        {
            return false;
        }
        !DXGI_RUNTIME_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub info: SelectedGpuInfo,
    pub capabilities: PreviewCapabilities,
    pub device: Device,
    pub queue: Queue,
    pub nv12_supported: bool,
    pub dxgi_adapter_index: Option<u32>,
    pub device_loss: Arc<DeviceLossState>,
}

impl GpuContext {
    /// Enumerates and scores all available graphics adapters to select the
    /// optimal device, then initializes its Device and Queue.
    ///
    /// This deliberately does not accept a `Surface`. Adapter/device discovery
    /// may run on a worker, while a native surface must be created on the UI
    /// thread (notably, CAMetalLayer creation on macOS). Surface compatibility
    /// is negotiated later by `native_surface::configure_surface`, which is
    /// always dispatched with `run_on_main_thread`.
    pub async fn select_best_gpu(instance: &Instance) -> Result<Self, String> {
        // enumerate_adapters is not available on wasm32 (no enumeration API in
        // the browser sandbox). The WASM crate uses init_gpu() directly and
        // never calls select_best_gpu on that target, but the function must
        // still compile. Guard the native-only path.
        #[cfg(not(target_arch = "wasm32"))]
        let best_adapter = {
            let adapters = instance.enumerate_adapters(wgpu::Backends::all());
            if !adapters.is_empty() {
                // Score adapters: Prioritize Discrete GPUs (1000), then Integrated (200), penalize CPU/Virtual
                let mut scored_adapters: Vec<(u32, Adapter)> = adapters
                    .into_iter()
                    .map(|adapter| {
                        let info = adapter.get_info();
                        #[allow(unused_mut)]
                        let mut score = match info.device_type {
                            DeviceType::DiscreteGpu => 1000,
                            DeviceType::IntegratedGpu => 200,
                            DeviceType::VirtualGpu => 50,
                            DeviceType::Cpu => 10,
                            DeviceType::Other => 0,
                        };
                        #[cfg(target_os = "windows")]
                        if info.backend == wgpu::Backend::Dx12 {
                            // Boost DX12 backend on Windows so D3D11VA DXGI shared texture import succeeds
                            score += 500;
                        }
                        (score, adapter)
                    })
                    .collect();

                scored_adapters.sort_by_key(|b| std::cmp::Reverse(b.0));
                scored_adapters.remove(0).1
            } else {
                // Fallback to request_adapter if enumerate_adapters returns empty on some platforms
                if let Some(adapter) = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    adapter
                } else if let Some(adapter) = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    adapter
                } else if let Some(adapter) = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::None,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                    })
                    .await
                {
                    adapter
                } else {
                    instance
                        .enumerate_adapters(wgpu::Backends::all())
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            "No compatible graphics adapters or software rasterizers found."
                                .to_string()
                        })?
                }
            }
        };

        // On wasm32 we fall back to a simple request_adapter — select_best_gpu
        // is not the primary path (init_gpu() in clypra-render-wasm is), but
        // we need this to compile.
        #[cfg(target_arch = "wasm32")]
        let best_adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No WebGPU adapter found".to_string())?;

        let info = best_adapter.get_info();
        let is_discrete = info.device_type == DeviceType::DiscreteGpu;

        let gpu_info = SelectedGpuInfo {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            vendor_id: info.vendor,
            device_id: info.device,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            is_discrete,
        };

        log::info!(
            "🎮 Bound Clypra Media Engine to: {} (vendor=0x{:04x}, device=0x{:04x}, driver='{}', driver_info='{}', type={:?}, backend={:?})",
            gpu_info.name,
            gpu_info.vendor_id,
            gpu_info.device_id,
            gpu_info.driver,
            gpu_info.driver_info,
            gpu_info.device_type,
            gpu_info.backend
        );

        let available_features = best_adapter.features();
        let adapter_limits = best_adapter.limits();
        log::info!(
            "[gpu] adapter capabilities: name='{}' vendor=0x{:04x} device=0x{:04x} type={:?} backend={:?} driver='{}' driver_info='{}' supported_features={:?} supported_limits={:?}",
            info.name,
            info.vendor,
            info.device,
            info.device_type,
            info.backend,
            info.driver,
            info.driver_info,
            available_features,
            adapter_limits,
        );
        let feature_attempts = device_feature_attempts(available_features);
        let full_features = feature_attempts[0];

        // Request only the canonical compositor limits. Asking for every limit
        // exposed by a DX12 adapter can select backend-specific optional
        // capabilities that are not stable for a presentation device.
        let required_limits =
            wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
        if !required_limits.check_limits(&adapter_limits) {
            return Err(format!(
                "Adapter '{}' ({}) does not meet Videx preview limits",
                info.name, info.backend
            ));
        }
        log::info!(
            "[gpu] device negotiation: backend={:?}, features={:?}, requested_limits={{max_texture_dimension_2d:{}, max_bind_groups:{}, max_storage_buffers_per_shader_stage:{}}}",
            info.backend,
            full_features,
            required_limits.max_texture_dimension_2d,
            required_limits.max_bind_groups,
            required_limits.max_storage_buffers_per_shader_stage,
        );

        let mut device_result = None;
        let mut device_errors = Vec::new();
        for required_features in feature_attempts {
            log::info!(
                "[gpu] requesting device: adapter='{}' backend={} features={:?}",
                info.name,
                info.backend,
                required_features,
            );
            match best_adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Native Wgpu Device"),
                        required_features,
                        required_limits: required_limits.clone(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
            {
                Ok(result) => {
                    device_result = Some((result, required_features));
                    break;
                }
                Err(error) => {
                    log::warn!(
                        "[gpu] device request rejected: adapter='{}' backend={} features={:?} error={error}",
                        info.name,
                        info.backend,
                        required_features,
                    );
                    device_errors.push(format!("features={required_features:?}: {error}"));
                }
            }
        }

        let ((device, queue), negotiated_features) = device_result.ok_or_else(|| {
            format!(
                "Failed to request wgpu device for adapter '{}' ({}): {}",
                info.name,
                info.backend,
                device_errors.join("; ")
            )
        })?;
        log::info!(
            "[gpu] device ready: adapter='{}' backend={} negotiated_features={:?} negotiated_limits={:?}",
            info.name,
            info.backend,
            negotiated_features,
            device.limits(),
        );

        let device_loss = Arc::new(DeviceLossState::default());
        device_loss.set_phase("device_created");
        let callback_state = Arc::clone(&device_loss);
        let callback_info = gpu_info.clone();
        device.set_device_lost_callback(move |reason, message| {
            callback_state.record_loss(format!("{reason:?}"), message.clone());
            let callback_phase = callback_state
                .diagnostic()
                .map(|diagnostic| diagnostic.phase)
                .unwrap_or_else(|| "unknown".to_string());
            log::error!(
                "[gpu] DEVICE LOST: adapter='{}' vendor=0x{:04x} device=0x{:04x} type={} backend={} callback_phase='{}' callback_reason={reason:?} message={message}",
                callback_info.name,
                callback_info.vendor_id,
                callback_info.device_id,
                callback_info.device_type,
                callback_info.backend,
                callback_phase,
            );
        });

        let error_state = Arc::clone(&device_loss);
        let error_info = gpu_info.clone();
        device.on_uncaptured_error(Box::new(move |error| {
            error_state.record_error(format!("{error:?}"));
            log::error!(
                "[gpu] uncaptured wgpu error at phase='{}': adapter='{}' vendor=0x{:04x} device=0x{:04x} backend={} error={error:?}",
                error_state.phase.lock().map(|phase| phase.clone()).unwrap_or_else(|_| "unknown".to_string()),
                error_info.name,
                error_info.vendor_id,
                error_info.device_id,
                error_info.backend,
            );
            if let wgpu::Error::OutOfMemory { .. } = error {
                error_state.record_loss("OutOfMemory".to_string(), format!("{error:?}"));
            }
        }));

        device_checkpoint(&device, &device_loss, "after_device_request")?;

        let nv12_supported = device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_NV12);
        let capabilities = PreviewCapabilities::probe(&best_adapter, &device);

        log::info!(
            "🚀 Negotiated Preview Capabilities: NV12={}, DXGI ZeroCopy={}, HW Decode={}, Native Surface={}, HDR={}",
            capabilities.wgpu_nv12,
            capabilities.dxgi_import,
            capabilities.hw_decode,
            capabilities.native_surface,
            capabilities.hdr,
        );

        #[allow(unused_mut)]
        let mut dxgi_adapter_index = None;
        #[cfg(target_os = "windows")]
        if info.backend == wgpu::Backend::Dx12 {
            use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
            if let Ok(factory) = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() } {
                let mut i = 0u32;
                while let Ok(dxgi_adapter) = unsafe { factory.EnumAdapters1(i) } {
                    if let Ok(desc) = unsafe { dxgi_adapter.GetDesc1() } {
                        if desc.VendorId == info.vendor && desc.DeviceId == info.device {
                            SELECTED_DXGI_ADAPTER_INDEX
                                .store(i as i32, std::sync::atomic::Ordering::Relaxed);
                            dxgi_adapter_index = Some(i);
                            log::info!(
                                "🔗 [adapter_selector] Matched wgpu DX12 adapter '{}' to DXGI adapter index {}",
                                info.name,
                                i
                            );
                            break;
                        }
                    }
                    i += 1;
                }
            }
        }

        Ok(Self {
            instance: instance.clone(),
            adapter: best_adapter,
            info: gpu_info,
            capabilities,
            device,
            queue,
            nv12_supported,
            dxgi_adapter_index,
            device_loss,
        })
    }

    pub fn mark_phase(&self, phase: impl Into<String>) {
        self.device_loss.set_phase(phase);
    }

    pub fn is_device_lost(&self) -> bool {
        self.device_loss.is_lost()
    }

    /// Record a synchronous native device-removal observation.  DXGI can
    /// remove the D3D12 device before wgpu's asynchronous callback runs, so
    /// callers at a native API failure boundary must be able to make the
    /// session state sticky immediately.
    pub fn mark_device_lost(
        &self,
        phase: impl Into<String>,
        reason: impl Into<String>,
        message: impl Into<String>,
    ) {
        let should_enrich_unknown = self
            .device_loss
            .diagnostic()
            .map(|diagnostic| diagnostic.reason == "Unknown")
            .unwrap_or(true);
        if !self.is_device_lost() || should_enrich_unknown {
            let phase = phase.into();
            self.device_loss.set_phase(phase);
            self.device_loss.record_loss(reason.into(), message.into());
        }
    }

    pub fn device_loss_diagnostic(&self) -> Option<DeviceLossDiagnostic> {
        self.device_loss.diagnostic()
    }

    pub fn first_uncaptured_error(&self) -> Option<UncapturedGpuError> {
        self.device_loss.first_error()
    }

    pub fn checkpoint(&self, phase: &str) -> Result<(), String> {
        device_checkpoint(&self.device, &self.device_loss, phase)
    }

    /// Reads the backend's own device-removal status without changing the
    /// rendering path. wgpu intentionally maps DXGI device-removal HRESULTs to
    /// `DeviceLostReason::Unknown`, so this probe preserves the native HRESULT
    /// at the point where we need to distinguish an already-removed device from
    /// a failure caused by surface configuration.
    pub fn log_backend_device_status(&self, phase: &str) {
        #[cfg(target_os = "windows")]
        if self.info.backend.eq_ignore_ascii_case("Dx12") {
            use wgpu::hal::api::Dx12;

            unsafe {
                self.device
                    .as_hal::<Dx12, _, ()>(|hal_device| match hal_device {
                        Some(hal_device) => match hal_device.raw_device().GetDeviceRemovedReason() {
                            Ok(()) => log::info!(
                                "[gpu] DX12 device-removal probe: phase='{}' status=healthy",
                                phase
                            ),
                            Err(error) => log::error!(
                                "[gpu] DX12 device-removal probe: phase='{}' status=removed hresult=0x{:08x} error={:?}",
                                phase,
                                error.code().0 as u32,
                                error
                            ),
                        },
                        None => log::warn!(
                            "[gpu] DX12 device-removal probe: phase='{}' HAL device unavailable",
                            phase
                        ),
                    });
            }
        }

        #[cfg(not(target_os = "windows"))]
        let _ = phase;
    }

    /// Attach application-level status reporting after GPU initialization has
    /// completed. The low-level callback above is installed before any device
    /// resources are created, so early losses are never missed.
    pub fn install_runtime_diagnostics(&self, status: Arc<Mutex<NativeGpuRuntimeStatus>>) {
        let state = Arc::clone(&self.device_loss);
        let info = self.info.clone();
        let callback_status = Arc::clone(&status);
        self.device.set_device_lost_callback(move |reason, message| {
            state.record_loss(format!("{reason:?}"), message.clone());
            let diagnostic = state.diagnostic();
            let callback_phase = diagnostic
                .as_ref()
                .map(|value| value.phase.as_str())
                .unwrap_or("unknown");
            let rendered = diagnostic
                .as_ref()
                .map(|value| {
                    format!(
                        "GPU device lost during {}: reason={} message={}",
                        value.phase, value.reason, value.message
                    )
                })
                .unwrap_or_else(|| format!("GPU device lost: {reason:?} {message}"));
            log::error!(
                "[gpu] DEVICE LOST: adapter='{}' vendor=0x{:04x} device=0x{:04x} backend={} callback_phase='{}' callback_reason={reason:?} message={message}; {}",
                info.name,
                info.vendor_id,
                info.device_id,
                info.backend,
                callback_phase,
                rendered,
            );
            if let Ok(mut current) = callback_status.lock() {
                current.mark_failed(rendered);
            }
        });

        let error_info = self.info.clone();
        let error_state = Arc::clone(&self.device_loss);
        let error_status = Arc::clone(&status);
        self.device.on_uncaptured_error(Box::new(move |error| {
            error_state.record_error(format!("{error:?}"));
            log::error!(
                "[gpu] runtime uncaptured error at phase='{}': adapter='{}' backend={} error={error:?}",
                error_state.phase.lock().map(|phase| phase.clone()).unwrap_or_else(|_| "unknown".to_string()),
                error_info.name,
                error_info.backend,
            );
            if let wgpu::Error::OutOfMemory { .. } = error {
                error_state.record_loss("OutOfMemory".to_string(), format!("{error:?}"));
                if let Ok(mut current) = error_status.lock() {
                    current.mark_failed(format!("GPU ran out of memory: {error:?}"));
                }
            }
        }));

        if let Some(diagnostic) = self.device_loss_diagnostic() {
            if let Ok(mut current) = status.lock() {
                current.mark_failed(format!(
                    "GPU device lost during {}: reason={} message={}",
                    diagnostic.phase, diagnostic.reason, diagnostic.message
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_loss_state_records_phase_and_reason() {
        let state = DeviceLossState::default();
        state.set_phase("native_surface_configuration");
        state.record_loss("DXGI_ERROR_DEVICE_REMOVED".to_string(), "driver reset".to_string());

        assert!(state.is_lost());
        assert_eq!(
            state.diagnostic(),
            Some(DeviceLossDiagnostic {
                reason: "DXGI_ERROR_DEVICE_REMOVED".to_string(),
                message: "driver reset".to_string(),
                phase: "native_surface_configuration".to_string(),
            })
        );
    }

    #[test]
    fn device_loss_state_starts_healthy() {
        let state = DeviceLossState::default();
        assert!(!state.is_lost());
        assert_eq!(state.diagnostic(), None);
    }

    #[test]
    fn optional_preview_features_have_deterministic_fallback_order() {
        let available = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_FORMAT_NV12;
        let attempts = device_feature_attempts(available);

        assert_eq!(attempts[0], available);
        assert_eq!(
            attempts[1],
            wgpu::Features::TEXTURE_FORMAT_NV12,
            "HDR feature is removed before the zero-copy NV12 path"
        );
        assert_eq!(attempts[2], wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
        assert_eq!(attempts[3], wgpu::Features::empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_adapter_selection_scoring() {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let result = GpuContext::select_best_gpu(&instance).await;
        if let Ok(gpu_ctx) = result {
            assert!(!gpu_ctx.info.name.is_empty(), "GPU name must not be empty");
            gpu_ctx
                .checkpoint("test_after_device_initialization")
                .expect("device must survive the first diagnostic checkpoint");
            println!(
                "Selected GPU: {} | Backend: {} | Type: {} | Driver: {} {}",
                gpu_ctx.info.name,
                gpu_ctx.info.backend,
                gpu_ctx.info.device_type,
                gpu_ctx.info.driver,
                gpu_ctx.info.driver_info
            );
        }
    }
}
