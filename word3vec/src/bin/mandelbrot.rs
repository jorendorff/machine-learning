use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use word3vec::gpu::{Gpu, Runner};

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct Rectangle {
    top_left: [f32; 2],
    bottom_right: [f32; 2],
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1024;
    const PACK_FACTOR: usize = 4;

    const WORKGROUP_WIDTH: usize = 16;
    const WORKGROUP_HEIGHT: usize = 16;

    let gpu = Gpu::new("mandelbrotter").await?;
    let module = gpu.load_wgsl_module(Some("mandelbrot"), include_str!("mandelbrot.wgsl")).await?;
    let mut runner = Runner::new(&gpu, &module, "mandelbrot");

    runner.bind_out(0, "mandelbrot pixels", WIDTH * HEIGHT);
    runner.bind_in(
        1,
        "viewport bounds",
        &Rectangle {
            top_left: [-1.0, -1.0],
            bottom_right: [0.0, 0.0],
        },
    );

    let size = (
        (WIDTH / PACK_FACTOR / WORKGROUP_WIDTH) as u32,
        (HEIGHT / WORKGROUP_HEIGHT) as u32,
        1,
    );
    runner.run(size).await?;

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
