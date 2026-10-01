# DX12/NV12 Preview Checkpoint

- DX12 zero-copy device-loss containment and the narrow NV12 decoder-array preflight are implemented; affected resources route to the healthy CPU-NV12 upload path before CreateSharedHandle.
- Playback capability probing now separates DXGI zero-copy transport from decode quality. CPU-NV12 fallback preserves the requested QualityTier::Full and source resolution; measured adaptive downgrades and timeout policy remain unchanged.
- Focused regression coverage was added for CPU-NV12 fallback preserving Full quality. Native/frontend checks passed; focused Windows test execution remains blocked by STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139).
