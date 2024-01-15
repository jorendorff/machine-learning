//! GPU-based word2vec.

use std::collections::HashMap;
use std::sync::Arc;
use std::{collections::hash_map::Entry, sync::mpsc};

use anyhow::{Context, Result};
use bytemuck::Pod;

/// Handle to a device that can run computations.
pub struct Gpu {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,

    /// Queue submissions we should poll for.
    submissions: mpsc::Sender<wgpu::SubmissionIndex>,
}

pub struct Runner<'gpu> {
    gpu: &'gpu Gpu,
    module: &'gpu wgpu::ShaderModule,
    entry_point: &'gpu str,
    buffers: HashMap<u32, wgpu::Buffer>,
}

impl Gpu {
    pub async fn new(label: &str) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
            flags: wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION,
        });

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("Couldn't create webgpu adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(label),
                    ..wgpu::DeviceDescriptor::default()
                },
                None,
            )
            .await
            .context("failed to create device")?;
        let device = Arc::new(device);

        // Start a thread to poll the device for submissions someone cares about.
        let (submissions, rx_submissions) = std::sync::mpsc::channel();
        std::thread::spawn({
            let device = Arc::clone(&device);
            move || {
                for index in rx_submissions {
                    device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
                }
            }
        });

        Ok(Gpu {
            device,
            queue,
            submissions,
        })
    }

    pub async fn load_wgsl_module(
        &self,
        label: Option<&str>,
        source: &str,
    ) -> Result<wgpu::ShaderModule> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(source.to_string().into()),
            });
        if let Some(err) = self.device.pop_error_scope().await {
            anyhow::bail!("bad: {err}");
        }
        Ok(module)
    }

    pub async fn submit(&self, command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let index = self.queue.submit(command_buffers);
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.submissions.send(index).unwrap();
        receiver.await.unwrap();
    }
}

impl<'gpu> Runner<'gpu> {
    pub fn new(
        gpu: &'gpu Gpu,
        module: &'gpu wgpu::ShaderModule,
        entry_point: &'gpu str,
    ) -> Runner<'gpu> {
        Runner {
            gpu,
            module,
            entry_point,
            buffers: HashMap::new(),
        }
    }

    fn bind(&mut self, binding: u32, label: &str, usage: wgpu::BufferUsages, size: usize) {
        let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        assert!(self.buffers.insert(binding, buffer).is_none());
    }

    pub fn bind_in<T>(&mut self, binding: u32, label: &str, data: &T)
    where
        T: bytemuck::NoUninit,
    {
        let bytes = bytemuck::bytes_of(data);
        self.bind(
            binding,
            label,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );

        self.gpu
            .queue
            .write_buffer(&self.buffers[&binding], 0, bytes);
    }

    pub fn bind_in_slice<T>(&mut self, binding: u32, label: &str, data: &[T])
    where
        T: bytemuck::NoUninit,
    {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        self.bind(
            binding,
            label,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );

        self.gpu
            .queue
            .write_buffer(&self.buffers[&binding], 0, bytes);
    }

    pub fn bind_out(&mut self, binding: u32, label: &str, size: usize) -> &wgpu::Buffer {
        let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.try_into().unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        match self.buffers.entry(binding) {
            Entry::Occupied(_) => panic!("binding {binding} bound twice"),
            Entry::Vacant(e) => e.insert(buffer),
        }
    }

    pub fn bind_in_out<T>(&mut self, binding: u32, label: &str, data: &T)
    where
        T: Pod,
    {
        let bytes = bytemuck::bytes_of(data);
        self.bind(
            binding,
            label,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );
        self.gpu
            .queue
            .write_buffer(&self.buffers[&binding], 0, bytes);
    }

    pub fn bind_in_out_slice<T>(&mut self, binding: u32, label: &str, data: &[T])
    where
        T: Pod,
    {
        //let bytes = bytemuck::bytes_of(data);
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        self.bind(
            binding,
            label,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );
        self.gpu
            .queue
            .write_buffer(&self.buffers[&binding], 0, bytes);
    }

    pub async fn run(&self, size: (u32, u32, u32)) -> Result<()> {
        self.gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        self.run_inner(size).await;
        if let Some(err) = self.gpu.device.pop_error_scope().await {
            anyhow::bail!("run failed: {err}");
        }
        Ok(())
    }

    async fn run_inner(&self, size: (u32, u32, u32)) {
        let pipeline = self
            .gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} compute pipeline", self.entry_point)),
                layout: None,
                module: self.module,
                entry_point: self.entry_point,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entries = self
            .buffers
            .iter()
            .map(|(&binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            })
            .collect::<Vec<_>>();
        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pixel bind group"),
                layout: &bind_group_layout,
                entries: &entries,
            });

        let command_buffer = {
            let mut command_encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("{} command encoder", self.entry_point)),
                    });

            {
                let mut compute_pass =
                    command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("{} compute pass", self.entry_point)),
                        timestamp_writes: None,
                    });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                compute_pass.dispatch_workgroups(size.0, size.1, size.2);
            }
            command_encoder.finish()
        };

        self.gpu.submit([command_buffer]).await;
    }

    pub async fn copy_to_mapped_buffer<'buf>(
        &self,
        binding: u32,
        readback_buffer: &'buf wgpu::Buffer,
    ) -> wgpu::BufferView<'buf> {
        let buf = &self.buffers[&binding];

        let command_buffer = {
            let mut command_encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("copy_out command encoder"),
                    });

            command_encoder.copy_buffer_to_buffer(buf, 0, readback_buffer, 0, buf.size());
            command_encoder.finish()
        };

        self.gpu.submit([command_buffer]).await;
        map_slice(&self.gpu.device, &readback_buffer.slice(..)).await
    }

    pub async fn copy_slice_out<T>(&self, binding: u32, out: &mut [T])
    where
        T: Pod,
    {
        let readback_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback buffer"),
            size: self.buffers[&binding].size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let view = self.copy_to_mapped_buffer(binding, &readback_buffer).await;
        out.copy_from_slice(bytemuck::cast_slice::<u8, T>(&view));
    }
}

async fn map_slice<'a>(
    device: &wgpu::Device,
    slice: &wgpu::BufferSlice<'a>,
) -> wgpu::BufferView<'a> {
    let (sender, receiver) = tokio::sync::oneshot::channel();

    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    assert!(matches!(
        device.poll(wgpu::Maintain::Wait),
        wgpu::MaintainResult::Ok
    ));
    receiver.await.unwrap().expect("map failed");
    slice.get_mapped_range()
}
