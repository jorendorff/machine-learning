#![allow(unused_variables)]

#[tokio::main]
async fn main() {
    //let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    //    backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
    //    dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
    //    gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
    //});

    let instance = wgpu::Instance::default();

    //let adapter = wgpu::util::initialize_adapter_from_env(&instance, None)
    //    .expect("Couldn't create webgpu adapter");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("failed to create wgpu adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("machine-learning compute device"),
                ..wgpu::DeviceDescriptor::default()
            },
            None,
        )
        .await
        .expect("failed to create device");

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;

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

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot plotter"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
    });

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
    .expect("error writing image");
}

async fn map_slice<'a>(device: &wgpu::Device, slice: &wgpu::BufferSlice<'a>) -> wgpu::BufferView<'a> {
    let (sender, receiver) = tokio::sync::oneshot::channel();

    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    assert!(device.poll(wgpu::Maintain::Wait));
    receiver.await.unwrap().expect("map failed");
    slice.get_mapped_range()
}
