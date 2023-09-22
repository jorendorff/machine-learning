@group(0) @binding(0)
var<storage, read_write> pixels: array<u32, (1024 * 1024 / 4)>;

@compute
@workgroup_size(16,16)
fn mandelbrot(@builtin(global_invocation_id) id: vec3<u32>) {
  pixels[id.y * 1024u + id.x] = id.x + 256 * id.y;
}
