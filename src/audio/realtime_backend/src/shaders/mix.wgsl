struct Params {
    frames: u32,
    voices: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if i >= params.frames { return; }
    var sum: f32 = 0.0;
    for(var v: u32 = 0u; v < params.voices; v = v + 1u) {
        sum = sum + input[v * params.frames + i];
    }
    output[i] = sum / f32(params.voices);
}
