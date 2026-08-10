struct ComputeProperties
{
  float deltaTime;
  uint objectCount;
  float2 padding;
  float4 cameraPositionRecycleDistance;
  float4 driftDirectionSpawnDistance;
};

[[vk::push_constant]]
ComputeProperties computeProperties;

struct GameObjectCompute
{
  float4 positionScale;
  float4 velocityMass;
  float4 orientation;
  float4 angularVelocity;
};

float4 multiplyQuaternions(float4 left, float4 right)
{
  float3 xyz = left.w * right.xyz + right.w * left.xyz + cross(left.xyz, right.xyz);
  float w = left.w * right.w - dot(left.xyz, right.xyz);
  return float4(xyz, w);
}

float3 rotateVectorByQuaternion(float3 vector, float4 quaternion)
{
  float3 rotationAxis = quaternion.xyz;
  float3 crossProduct = cross(rotationAxis, vector);
  float3 twiceCrossProduct = 2.0f * crossProduct;
  return vector + quaternion.w * twiceCrossProduct + cross(rotationAxis, twiceCrossProduct);
}

[[vk::binding(0, 0)]]
StructuredBuffer<GameObjectCompute> gameObjectsIn;

[[vk::binding(1, 0)]]
RWStructuredBuffer<GameObjectCompute> gameObjectsOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void compMain(uint3 threadId : SV_DispatchThreadID)
{
  uint index = threadId.x;

  if (index >= computeProperties.objectCount) {
    return;
  }

  GameObjectCompute gameObject = gameObjectsIn[index];
  gameObject.positionScale.xyz += gameObject.velocityMass.xyz * computeProperties.deltaTime;

  float distanceAlongDrift = dot(gameObject.positionScale.xyz - computeProperties.cameraPositionRecycleDistance.xyz,
                                 computeProperties.driftDirectionSpawnDistance.xyz);
  if (distanceAlongDrift > computeProperties.cameraPositionRecycleDistance.w)
  {
    gameObject.positionScale.xyz -= computeProperties.driftDirectionSpawnDistance.xyz *
                                    (computeProperties.driftDirectionSpawnDistance.w +
                                     computeProperties.cameraPositionRecycleDistance.w);
  }

  float4 angularVelocityQuaternion = float4(gameObject.angularVelocity.xyz, 0.0f);
  float4 orientationDerivative = multiplyQuaternions(angularVelocityQuaternion, gameObject.orientation);
  gameObject.orientation += 0.5f * orientationDerivative * computeProperties.deltaTime;
  gameObject.orientation = normalize(gameObject.orientation);
  gameObjectsOut[index] = gameObject;
}

struct UniformBuffer {
  float4x4 view;
  float4x4 proj;
}

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;
[[vk::binding(1, 0)]]
StructuredBuffer<GameObjectCompute> gameObjects;

struct GraphicsProperties
{
  uint objectIndex;
};

[[vk::push_constant]]
GraphicsProperties graphicsProperties;

struct VSInput {
  float3 inPosition;
  float3 inColor;
  float2 inTexCoord;
};

struct VSOutput {
  float4 pos : SV_Position;
  float3 fragColor;
  float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
  VSOutput output;
  GameObjectCompute gameObject = gameObjects[graphicsProperties.objectIndex];
  float3 scaledPosition = input.inPosition * gameObject.positionScale.w;
  float3 rotatedPosition = rotateVectorByQuaternion(scaledPosition, gameObject.orientation);
  float3 objectPosition = rotatedPosition + gameObject.positionScale.xyz;
  output.pos = mul(ubo.proj, mul(ubo.view, float4(objectPosition, 1.0)));
  output.fragColor = input.inColor;
  output.fragTexCoord = input.inTexCoord;
  return output;
}

[[vk::binding(0, 1)]]
Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
  return float4(vertIn.fragColor * texture.Sample(vertIn.fragTexCoord).rgb, 1.0);
}
