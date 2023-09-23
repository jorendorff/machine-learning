@group(0) @binding(0)
var<storage, read_write> pixels: array<u32, (1024 * 1024 / 4)>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2(a.x * b.x - a.y * b.y,
              a.x * b.y + a.y * b.x);
}

fn iterations(c: vec2<f32>) -> u32 {
  var n: u32;
  var z = vec2(0.0);
  for (n = 0u; n < 255u && length(z) < 2.0; n++) {
    z = cmul(z, z) + c;
  }
  return n;
}

@compute
@workgroup_size(16,16)
fn mandelbrot(@builtin(global_invocation_id) id: vec3<u32>) {
  let x: u32 = id.x * 4u;
  var value: u32 = 0u;

  for (var dx: u32 = 0u; dx < 4u; dx++) {
    let loc: vec2<f32> = vec2(f32(4u * id.x + dx) / 1024.0, f32(id.y) / 1024.0);
    let c: vec2<f32> = mix(vec2(-2.0, -2.0), vec2(2.0, 2.0), loc);
    let pixel: u32 = iterations(c);

    value = (value >> 8u) | (pixel << 24u);
  }

  pixels[id.y * (1024u / 4u) + id.x] = value;
}
