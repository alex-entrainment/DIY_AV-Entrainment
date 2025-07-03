#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;
#[cfg(feature = "gpu")]
use pollster::block_on;
#[cfg(feature = "gpu")]
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};

#[cfg(feature = "gpu")]
pub struct GpuMixer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    frames: u32,
    voices: u32,
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
        let frames = output.len() as u32;
        let voices = inputs.len() as u32;

        // Flatten input buffers into contiguous array
        let mut interleaved: Vec<f32> = Vec::with_capacity((frames * voices) as usize);
        for buf in inputs {
            interleaved.extend_from_slice(buf);
        }

        let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mix_input"),
            contents: cast_slice(&interleaved),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_output"),
            size: (frames as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params { frames, voices };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mix_params"),
            contents: bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mix_bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mix_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mix_pass"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (frames + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_readback"),
            size: (frames as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback, 0, (frames as u64) * 4);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        output.copy_from_slice(cast_slice(&data));
        drop(data);
        readback.unmap();
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
