//! GPU-based word2vec.

#![allow(unused_variables)]

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};

/// Handle to a device that can run computations.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct Runner<'gpu> {
    gpu: &'gpu Gpu,
    module: &'gpu wgpu::ShaderModule,
    entry_point: &'gpu str,
    buffers: HashMap<u32, wgpu::Buffer>,
}

impl Gpu {
    async fn new(label: &str) -> Result<Self> {
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

        Ok(Gpu { device, queue })
    }

    async fn load_wgsl_module(&self, path: &Path) -> Result<wgpu::ShaderModule> {
        let source = fs::read_to_string(path)?;
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{path:?}")),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        if let Some(err) = self.device.pop_error_scope().await {
            anyhow::bail!("bad: {err}");
        }
        Ok(module)
    }

    async fn submit(&self, command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission_index = self.queue.submit(command_buffers);
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        assert!(self.device.poll(wgpu::Maintain::Wait));
        receiver.await.unwrap();
    }
}

impl<'gpu> Runner<'gpu> {
    fn new(
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

    fn bind(
        &mut self,
        binding: u32,
        label: &str,
        usage: wgpu::BufferUsages,
        size: usize,
    ) -> &wgpu::Buffer {
        let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        match self.buffers.entry(binding) {
            Entry::Occupied(_) => panic!("binding {binding} bound twice"),
            Entry::Vacant(e) => e.insert(buffer),
        }
    }

    fn bind_in<T>(&mut self, binding: u32, label: &str, data: &T)
    where
        T: bytemuck::NoUninit,
    {
        let bytes = bytemuck::bytes_of(data);
        let buffer = self.bind(
            binding,
            label,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );

        self.gpu.queue.write_buffer(&buffer, 0, bytes);
    }

    fn bind_in_slice<T>(&mut self, binding: u32, label: &str, data: &[T])
    where
        T: bytemuck::NoUninit,
    {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let buffer = self.bind(
            binding,
            label,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );

        self.gpu.queue.write_buffer(&buffer, 0, bytes);
    }

    fn bind_out(&mut self, binding: u32, label: &str, size: usize) -> &wgpu::Buffer {
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

    fn bind_in_out<T>(&mut self, binding: u32, label: &str, data: &'gpu T)
    where
        T: Pod,
    {
        let bytes = bytemuck::bytes_of(data);
        let buffer = self.bind(
            binding,
            label,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );
        self.gpu.queue.write_buffer(buffer, 0, bytes);
    }

    fn bind_in_out_slice<T>(&mut self, binding: u32, label: &str, data: &'gpu [T])
    where
        T: Pod,
    {
        //let bytes = bytemuck::bytes_of(data);
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let buffer = self.bind(
            binding,
            label,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            bytes.len(),
        );
        self.gpu.queue.write_buffer(buffer, 0, bytes);
    }

    async fn run(&self) -> Result<()> {
        self.gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        self.run_inner().await;
        if let Some(err) = self.gpu.device.pop_error_scope().await {
            anyhow::bail!("run failed: {err}");
        }
        Ok(())
    }

    async fn run_inner(&self) {
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

                // TODO - make this code generic across workloads :-\
                // compute_pass.dispatch_workgroups(
                //     (WIDTH / PACK_FACTOR / WORKGROUP_WIDTH) as u32,
                //     (HEIGHT / WORKGROUP_HEIGHT) as u32,
                //     1,
                // );
            }
            command_encoder.finish()
        };

        self.gpu.submit([command_buffer]).await;
    }

    async fn copy_to_mapped_buffer<'buf>(
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

            command_encoder.copy_buffer_to_buffer(buf, 0, &readback_buffer, 0, buf.size());
            command_encoder.finish()
        };

        self.gpu.submit([command_buffer]).await;
        map_slice(&self.gpu.device, &readback_buffer.slice(..)).await
    }

    async fn copy_slice_out<T>(&self, binding: u32, out: &mut [T])
    where
        T: Pod,
    {
        let readback_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback buffer"),
            size: self.buffers[&binding].size().try_into().unwrap(),
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
    assert!(device.poll(wgpu::Maintain::Wait));
    receiver.await.unwrap().expect("map failed");
    slice.get_mapped_range()
}

async fn train(gpu: &Gpu, model: &mut crate::Model) -> anyhow::Result<()> {
    let module = gpu.load_wgsl_module(Path::new("learn.wgsl")).await?;

    let runner = Runner::new(gpu, &module, "adjust");
    runner.bind_in_out_slice(0, "embeddings", &model.embeddings);
    runner.bind_in_out_slice(1, "weights", &model.weights);
    runner.bind_in_slice(2, "paths", &model.paths);
    runner.bind_in_slice(3, "ranges", &model.ranges);
    runner.bind_in_slice(4, "tasks", &model.tasks);
    runner.run().await?;

    runner.copy_slice_out(0, &mut model.embeddings).await;
    runner.copy_slice_out(1, &mut model.weights).await;

    Ok(())
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct Rectangle {
    top_left: [f32; 2],
    bottom_right: [f32; 2],
}

#[tokio::main]
async fn mandelbrot_example() -> anyhow::Result<()> {
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;
    const PACK_FACTOR: usize = 4;

    const WORKGROUP_WIDTH: usize = 16;
    const WORKGROUP_HEIGHT: usize = 16;

    let gpu = Gpu::new("mandelbrotter").await?;
    let module = gpu.load_wgsl_module(Path::new("learn.wgsl")).await?;
    let runner = Runner::new(&gpu, &module, "mandelbrot");

    runner.bind_out(0, "mandelbrot pixels", WIDTH * HEIGHT);
    runner.bind_in(
        1,
        "viewport bounds",
        &Rectangle {
            top_left: [-1.0, -1.0],
            bottom_right: [0.0, 0.0],
        },
    );

    runner.run().await?;

    let mut pixels = [0u8; WIDTH * HEIGHT];
    runner.copy_slice_out(0, &mut pixels).await;

    image::save_buffer_with_format(
        "mandelbrot.png",
        &pixels,
        WIDTH.try_into().unwrap(),
        HEIGHT.try_into().unwrap(),
        image::ColorType::L8,
        image::ImageFormat::Png,
    )
    .context("error writing image")?;

    Ok(())
}
