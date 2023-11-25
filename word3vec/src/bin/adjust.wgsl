const DIM: u32 = 200;
const WORKGROUP_SIZE: u32 = 256;
const ROOT: u32 = 0xffffffff;

@group(0) @binding(0)
var<storage, read_write> embeddings: array<array<f32, DIM>>;

@group(0) @binding(1)
var<storage, read_write> weights: array<array<f32, DIM>>;

struct Decision {
  weight: u32,

  // 1 for left, 0 for right
  direction: u32,
};
@group(0) @binding(2)
var<uniform> paths: array<Decision>;

/// `ranges[i]` is the index in `paths` of the start of the path from
/// root to leaf of embedding `i`.
@group(0) @binding(3)
var<uniform> ranges: array<u32>;

// Indices in embeddings of the given word, and the word we wish we
// predicted.
struct Pair {
  given: u32,
  predicted: u32,
}

@group(0) @binding(4)
var<uniform> tasks: array<Pair>;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn adjust(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let task = tasks[invocation.x];

  var path = ranges[task.predicted];
  var path_end = ranges[task.predicted + 1];
  for (var i: u32 = path; i != path_end; i++) {
    let decision = paths[i];
    
  }
}
