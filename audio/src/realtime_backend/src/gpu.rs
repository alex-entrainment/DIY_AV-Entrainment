#[cfg(feature = "gpu")]
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
#[cfg(feature = "gpu")]
use pollster::block_on;

#[cfg(feature = "gpu")]
pub struct GpuMixer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    // Persistent resources reused across mix calls to avoid per-call allocations
    input_buf: Option<wgpu::Buffer>,
    output_buf: Option<wgpu::Buffer>,
    params_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    max_frames: u32,
    max_voices: u32,
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
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no adapter available");
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("failed to create device");
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/mix.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mix"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        Self {
            device,
            queue,
            pipeline,
            input_buf: None,
            output_buf: None,
            params_buf: None,
            readback_buf: None,
            bind_group: None,
            max_frames: 0,
            max_voices: 0,
        }
    }

    /// Mix the given input buffers into `output` using the GPU when possible.
    /// Currently this falls back to a CPU implementation under the hood.
    pub fn mix(&mut self, inputs: &[&[f32]], output: &mut [f32]) {
        if inputs.is_empty() {
            output.fill(0.0);
            return;
        }
        let frames = output.len() as u32;
        let voices = inputs.len() as u32;

        self.ensure_resources(frames, voices);

        // Flatten input buffers into contiguous array
        let mut interleaved: Vec<f32> = Vec::with_capacity((frames * voices) as usize);
        for buf in inputs {
            interleaved.extend_from_slice(buf);
        }

        if let Some(buf) = &self.input_buf {
            self.queue.write_buffer(buf, 0, cast_slice(&interleaved));
        }
        let params = Params { frames, voices };
        if let Some(pbuf) = &self.params_buf {
            self.queue.write_buffer(pbuf, 0, bytes_of(&params));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mix_encoder"),
            });
        if let (Some(bind_group), Some(output_buf), Some(readback)) =
            (&self.bind_group, &self.output_buf, &self.readback_buf)
        {
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mix_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                let workgroups = (frames + 63) / 64;
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(output_buf, 0, readback, 0, (frames as u64) * 4);
            self.queue.submit(Some(encoder.finish()));

            // Only map the portion of the readback buffer that was written to
            // for this mix call. Using the entire buffer can result in a
            // mismatch between the GPU output size and the destination slice
            // when the internal buffers are larger than the requested number
            // of frames.
            let buffer_slice = readback.slice(..(frames as u64 * 4));
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

    fn ensure_resources(&mut self, frames: u32, voices: u32) {
        if frames <= self.max_frames && voices <= self.max_voices {
            return;
        }
        self.max_frames = self.max_frames.max(frames);
        self.max_voices = self.max_voices.max(voices);

        let input_size = (self.max_frames * self.max_voices) as u64 * 4;
        let output_size = (self.max_frames as u64) * 4;

        self.input_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_input"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.output_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.params_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.readback_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mix_bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuMixer;

#[cfg(not(feature = "gpu"))]
impl GpuMixer {
    pub fn new() -> Self {
        Self
    }
    pub fn mix(&mut self, inputs: &[&[f32]], output: &mut [f32]) {
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
