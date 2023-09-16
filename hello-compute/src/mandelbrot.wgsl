@group(0) @binding(0)
var<storage> pixels: array<u32, (1024 * 1024)>;

@compute
@workgroup_size(16,16)
fn mandelbrot(@builtin(global_invocation_id) id: vec3<u32>) {
   
}
