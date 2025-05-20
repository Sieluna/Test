use super::*;

impl Sampler {
    /// Creates a new [`Sampler`] using the given wrap and filter mode.
    pub fn new(fw: &Framework, wrap_mode: SamplerWrapMode, filter_mode: SamplerFilterMode) -> Self {
        let address_mode = match wrap_mode {
            SamplerWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            SamplerWrapMode::Repeat => wgpu::AddressMode::Repeat,
            SamplerWrapMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
            SamplerWrapMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
        };
        let wgpu_filter_mode = match filter_mode {
            SamplerFilterMode::Nearest => wgpu::FilterMode::Nearest,
            SamplerFilterMode::Linear => wgpu::FilterMode::Linear,
        };
        let sampler = fw.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: wgpu_filter_mode,
            min_filter: wgpu_filter_mode,
            mipmap_filter: wgpu_filter_mode,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        Self {
            sampler,
            filter_mode,
        }
    }
}
