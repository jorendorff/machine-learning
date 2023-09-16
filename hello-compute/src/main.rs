#![allow(unused_variables)]

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
        gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
    });

    let adapter = wgpu::util::initialize_adapter_from_env(&instance, None)
        .expect("Couldn't create webgpu adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("machine-learning compute device"),
            .. wgpu::DeviceDescriptor::default()
        },
        None
    ))
        .expect("failed to create device");

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mandelbrot pixels"),
        size: 1024 * 1024 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: 1024 * 1024 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot plotter"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
    });
    
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        panic!("error scope found something: {}", err);
    }
}
