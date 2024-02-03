const DIM: i32 = 200;
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

// `ranges[i]` is the index in `paths` of the start of the path from
// root to leaf of embedding `i`.
//
// Thus the path from the root to word `i` is a slice of `paths`,
// `paths[ranges[i] .. ranges[i + 1]]`.
//
// The length of this array is NWORDS + 1 so that this formula always works.
@group(0) @binding(3)
var<storage, read> ranges: array<u32>;

struct Task {
  // index in embeddings of the given word
  given: u32,
  // index in embeddings of the word we should predict
  predicted: u32,
}

@group(0) @binding(4)
var<storage, read> tasks: array<Task>;

struct Dispatch {
  alpha: f32,
  // index into tasks. number of tasks is implicit in the workgroup size
  first_task: u32,
}

var<push_constant> dispatch: Dispatch;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// One invocation of this function takes a single task and updates
// - The embedding for the `given` word
// - The weights of the decision nodes along the path to the `predicted` word.
@compute
@workgroup_size(WORKGROUP_SIZE)
fn adjust(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let task = tasks[dispatch.first_task + invocation.x];

  // The given word's embedding.
  let embedding = &embeddings[task.given];

  // The range within `paths` holding the path from the root to
  // `task.predicted`.
  var path = ranges[task.predicted];
  var path_end = ranges[task.predicted + 1];

  // The adjustments we will apply to `embedding` when we're done.
  // WGSL initializes this to zero.
  var adjust: array<f32, DIM>;

  for (var i = path; i != path_end; i++) {
    let decision = paths[i];
    let weights = &weights[decision.weight];
    var dot = 0.0;
    for (var j = 0; j < DIM; j++) {
      dot += (*embedding)[j] * (*weights)[j];
    }

    let prediction = sigmoid(dot);
    let gradient = (1.0 - decision.direction - prediction) * dispatch.alpha;

    for (var j = 0; j < DIM; j++) {
      adjust[j] += gradient * (*weights)[j];
      (*weights)[j] += gradient * (*embedding)[j];
    }
  }

  for (var j = 0; j < DIM; j++) {
    (*embedding)[j] += adjust[j];
  }
}
