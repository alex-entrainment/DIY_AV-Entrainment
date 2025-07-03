#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;
#[cfg(feature = "gpu")]
use pollster::block_on;

#[cfg(feature = "gpu")]
pub struct GpuMixer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl GpuMixer {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("no adapter available");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).expect("failed to create device");
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/mix.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("mix"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        Self { device, queue, pipeline }
    }

    /// Mix the given input buffers into `output` using the GPU when possible.
    /// Currently this falls back to a CPU implementation under the hood.
    pub fn mix(&self, inputs: &[&[f32]], output: &mut [f32]) {
        if inputs.is_empty() {
            output.fill(0.0);
            return;
        }
        // TODO: implement GPU compute shader dispatch
        output.fill(0.0);
        let gain = 1.0 / inputs.len() as f32;
        for buf in inputs {
            for (o, &v) in output.iter_mut().zip(buf.iter()) {
                *o += v * gain;
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuMixer;

#[cfg(not(feature = "gpu"))]
impl GpuMixer {
    pub fn new() -> Self { Self }
    pub fn mix(&self, inputs: &[&[f32]], output: &mut [f32]) {
        if inputs.is_empty() {
            output.fill(0.0);
            return;
        }
        let gain = 1.0 / inputs.len() as f32;
        output.fill(0.0);
        for buf in inputs {
            for (o, &v) in output.iter_mut().zip(buf.iter()) {
                *o += v * gain;
            }
        }
    }
}
