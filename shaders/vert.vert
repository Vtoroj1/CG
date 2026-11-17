#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambient;
    ivec4 lightCounts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 specColor;
    vec4 params;
} model;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;

void main() {
    vec4 worldPos = model.model * vec4(inPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(mat3(model.normalMatrix) * inNormal);
    vUV = vec2(inUV.x, 1.0 - inUV.y);
    gl_Position = scene.proj * scene.view * worldPos;
}