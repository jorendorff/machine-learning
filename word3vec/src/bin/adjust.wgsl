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
  direction: f32,
}

@group(0) @binding(2)
var<storage, read> paths: array<Decision>;

/// `ranges[i]` is the index in `paths` of the start of the path from
/// root to leaf of embedding `i`.
@group(0) @binding(3)
var<storage, read> ranges: array<u32>;

// Indices in embeddings of the given word, and the word we wish we
// predicted.
struct Task {
  given: u32,
  predicted: u32,
  alpha: f32,
}

@group(0) @binding(4)
var<storage, read> tasks: array<Task>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn adjust(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let task = tasks[invocation.x];

  // The given word's embedding.
  let embedding = &embeddings[task.given];

  // The range within `paths` holding the path from the root to
  // `task.predicted`.
  var path = ranges[task.predicted];
  var path_end = ranges[task.predicted + 1u];

  // The adjustments we will apply to `embedding` when we're done.
  // WGSL initializes this to zero.
  var adjust: array<f32, DIM>;

  for (var i = path; i != path_end; i++) {
    let decision = paths[i];
    let weights = &weights[decision.weight];
    var dot = 0.0;
    for (var j = 0u; j < DIM; j++) {
      dot += (*embedding)[j] * (*weights)[j];
    }

    let prediction = sigmoid(dot);
    let gradient = (1.0 - decision.direction - prediction) * task.alpha;

    for (var j = 0u; j < DIM; j++) {
      adjust[j] += gradient * (*weights)[j];
      (*weights)[j] += gradient * (*embedding)[j];
    }
  }

  for (var j = 0u; j < DIM; j++) {
    (*embedding)[j] += adjust[j];
  }
}
