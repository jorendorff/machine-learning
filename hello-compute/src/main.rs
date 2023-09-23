#![allow(unused_variables)]

use anyhow::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
        gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
    });

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .expect("Couldn't create webgpu adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("machine-learning compute device"),
                ..wgpu::DeviceDescriptor::default()
            },
            None,
        )
        .await
        .context("failed to create device")?;

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;
    const PACK_FACTOR: usize = 4;

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mandelbrot pixels"),
        size: (WIDTH * HEIGHT).try_into().unwrap(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: (WIDTH * HEIGHT).try_into().unwrap(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    const WORKGROUP_WIDTH: usize = 16;
    const WORKGROUP_HEIGHT: usize = 16;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot plotter"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
    });

    if let Some(err) = device.pop_error_scope().await {
        panic!("error scope found something: {}", err);
    }
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mandelbrot compute pipeline"),
        layout: None,
        module: &module,
        entry_point: "mandelbrot",
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pixel bind group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &storage_buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mandelbrot command encoder"),
    });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mandelbrot compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (WIDTH / PACK_FACTOR / WORKGROUP_WIDTH) as u32,
            (HEIGHT / WORKGROUP_HEIGHT) as u32,
            1,
        );
    }

    command_encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &readback_buffer,
        0,
        (WIDTH * HEIGHT) as u64,
    );

    let command_buffer = command_encoder.finish();

    let submission_index = queue.submit([command_buffer]);
    wait_for_submitted_work(&device, &queue).await;

    if let Some(err) = device.pop_error_scope().await {
        panic!("error scope found something: {}", err);
    }

    let view = map_slice(&device, &readback_buffer.slice(..)).await;

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

async fn wait_for_submitted_work(device: &wgpu::Device, queue: &wgpu::Queue) {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    assert!(device.poll(wgpu::Maintain::Wait));
    receiver.await.unwrap();
}
