//! GPU-based word2vec.
#![allow(unused_variables)]

use std::path::Path;
use std::fs;
use std::collections::HashMap;

use anyhow::{Context, Result};

struct Wgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

enum RunnerBuffer<'gpu> {
    ReadOnly(wgpu::Buffer),
    ReadWrite(&'gpu mut [u8]),
    Out {
        storage_buffer: wgpu::Buffer,
        readback_buffer: wgpu::Buffer,
    },
}

struct Runner<'gpu> {
    wgpu: &'gpu Wgpu,
    module: &'gpu wgpu::ShaderModule,
    entry_point: &'gpu str,
    buffers: HashMap<u32, RunnerBuffer<'gpu>>,
}

impl Wgpu {
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

        Ok(Wgpu { device, queue })
    }


    async fn load_wgsl_module(&self, path: &Path) -> Result<wgpu::ShaderModule> {
        let source = fs::read_to_string(path)?;
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{path:?}")),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        if let Some(err) = self.device.pop_error_scope().await {
           anyhow::bail!("bad: {err}");
        }
        Ok(module)
    }

    async fn submit(&self, command_buffers: impl IntoIterator<Item=wgpu::CommandBuffer>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission_index = self.queue.submit(command_buffers);
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        assert!(self.device.poll(wgpu::Maintain::Wait));
        receiver.await.unwrap();
    }
}

impl<'gpu> RunnerBuffer<'gpu> {
    fn binding_resource(&self) -> wgpu::BindingResource {
        // TODO: finish and polish this
        match self {
            RunnerBuffer::ReadOnly(buffer) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
            RunnerBuffer::Out { storage_buffer, .. } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: storage_buffer,
                offset: 0,
                size: None,
            }),
            RunnerBuffer::ReadWrite(_) => panic!("not implemented"),
        }
    }
}

impl<'gpu> Runner<'gpu> {
    fn new(wgpu: &'gpu Wgpu, module: &'gpu wgpu::ShaderModule, entry_point: &'gpu str) -> Runner<'gpu> {
        Runner { wgpu, module, entry_point, buffers: HashMap::new() }
    }

    fn read_only<T>(&self, binding: u32, label: &str, data: &T)
    where
        T: bytemuck::NoUninit,
    {
        assert!(!self.buffers.contains_key(&binding));

        let bytes = bytemuck::bytes_of(data);
        let buffer = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len().try_into().unwrap(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.buffers.insert(binding, RunnerBuffer::ReadOnly(buffer));
    }

    fn uniform_in<T>(&self, binding: u32, label: &str, data: &T)
    where
        T: bytemuck::NoUninit,
    {
        assert!(!self.buffers.contains_key(&binding));

        let bytes = bytemuck::bytes_of(data);
        let buffer = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len().try_into().unwrap(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu.queue.write_buffer(&buffer, 0, bytes);

        self.buffers.insert(binding, RunnerBuffer::ReadOnly(buffer));
    }

    fn out(&self, binding: u32, label: &str, size: usize)
    {
        assert!(!self.buffers.contains_key(&binding));

        let storage_buffer = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.try_into().unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} readback buffer")),
            size: size.try_into().unwrap(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.buffers.insert(binding, RunnerBuffer::Out { storage_buffer, readback_buffer });
    }

    fn read_write<T>(&self, binding: u32, label: &str, data: &'gpu mut T)
    where
        T: bytemuck::Pod,
    {
        assert!(!self.buffers.contains_key(&binding));

        let bytes = bytemuck::bytes_of_mut(data);
        let buffer = self.wgpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len().try_into().unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffers.insert(binding, RunnerBuffer::ReadWrite(bytes));
    }

    async fn run(&self) -> Result<()> {
        self.wgpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.run_inner().await;
        if let Some(err) = self.wgpu.device.pop_error_scope().await {
            anyhow::bail!("run failed: {err}");
        }
        Ok(())
    }

    async fn run_inner(&self) {
        let pipeline = self.wgpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} compute pipeline", self.entry_point)),
            layout: None,
            module: self.module,
            entry_point: self.entry_point,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entries = self.buffers.iter().map(|(&binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.binding_resource(),
        }).collect::<Vec<_>>();
        let bind_group = self.wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel bind group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let command_buffer = {
            let mut command_encoder = self.wgpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} command encoder", self.entry_point)),
            });

            {
                let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{} compute pass", self.entry_point)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // TODO - make this code generic across workloads :-\
                compute_pass.dispatch_workgroups(
                    (WIDTH / PACK_FACTOR / WORKGROUP_WIDTH) as u32,
                    (HEIGHT / WORKGROUP_HEIGHT) as u32,
                    1,
                );
            }

            // TODO - post-compute buffer copies per self.buffers
            command_encoder.copy_buffer_to_buffer(
                &storage_buffer,
                0,
                &readback_buffer,
                0,
                (WIDTH * HEIGHT) as u64,
            );

            command_encoder.finish()
        };

        self.wgpu.submit([command_buffer]).await;
    }
}

async fn train(wgpu: &Wgpu, model: &mut crate::Model) -> anyhow::Result<()> {
    let module = wgpu.load_wgsl_module(Path::new("learn.wgsl")).await?;

    let runner = Runner::new(wgpu, &module, "adjust");
    runner.read_write(0, "embeddings", model.embeddings.as_mut());
    runner.read_write(1, "weights", model.weights.as_mut());
    runner.read_only(2, "paths", &model.paths);
    runner.read_only(3, "ranges", &model.ranges);
    runner.read_only(4, "tasks", &model.tasks);
    runner.run().await?;
    Ok(())
}

#[tokio::main]
async fn mandelbrot_example() -> anyhow::Result<()> {
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;
    const PACK_FACTOR: usize = 4;

    const WORKGROUP_WIDTH: usize = 16;
    const WORKGROUP_HEIGHT: usize = 16;

    let wgpu = Wgpu::new("mandelbrotter").await?;
    let module = wgpu.load_wgsl_module(Path::new("learn.wgsl")).await?;
    let runner = Runner::new(wgpu, &module, "mandelbrot");

    runner.out(0, "mandelbrot pixels", WIDTH * HEIGHT);
    runner.uniform_in(1, "viewport bounds", &Rectangle {
        top_left: [-1.0, -1.0],
        bottom_right: [0.0, 0.0],
    });

    runner.run().await?;

    let view = map_slice(&wgpu.device, &readback_buffer.slice(..)).await;

    image::save_buffer_with_format(
        "mandelbrot.png",
        &view,
        WIDTH.try_into().unwrap(),
        HEIGHT.try_into().unwrap(),
        image::ColorType::L8,
        image::ImageFormat::Png,
    )
    .context("error writing image")?;

    Ok(())
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
