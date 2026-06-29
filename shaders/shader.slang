struct VSInput {
  float2 inPosition;
  float3 inColor;
};

struct VSOutput {
  float4 pos : SV_Position;
  float3 color;
};

//static float2 positions[3] = float2[](
//  float2(0.0, -0.5),
//  float2(0.5, 0.5),
//  float2(-0.5, 0.5)
//);
//
//static float3 colors[3] = float3[](
//  float3(1.0, 0.0, 0.0),
//  float3(0.0, 1.0, 0.0),
//  float3(0.0, 0.0, 1.0)
//);


[shader("vertex")]
VSOutput vertMain(VSInput input) {
  VSOutput output;
  output.pos = float4(input.inPosition, 0.0, 1.0);
  output.color = input.inColor;
  return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
  return float4(vertIn.color, 1.0);
}
