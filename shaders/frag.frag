#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambient;
    ivec4 lightCounts;
} scene;

layout(set = 0, binding = 3, std140) uniform DirectionalLight {
    vec4 direction;
    vec4 color;
} dirLight;

struct PointLight { vec4 position; vec4 color; };
struct SpotLight  { vec4 position; vec4 direction; vec4 color; vec4 params; };

layout(set = 0, binding = 1, std430) buffer PointLights { PointLight points[]; } pointLights;
layout(set = 0, binding = 2, std430) buffer SpotLights  { SpotLight  spots[];  } spotLights;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 specColor;
    vec4 params;
} model;

layout(set = 1, binding = 1) uniform sampler2D albedoTex;

layout(push_constant) uniform PushConstants {
    int useTrippySampling;
    float time;
} pc;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity, float attenuation, vec3 albedo, vec3 specCol) {
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * intensity * albedo;

    vec3 H = normalize(L + V);
    float spec = 0.0;
    if (diff > 0.0) spec = pow(max(dot(N, H), 0.0), model.params.x);
    vec3 specular = spec * lightColor * intensity * specCol;

    return (diffuse + specular) * attenuation;
}

void main() {
    vec3 albedo = texture(albedoTex, vUV).rgb;

    if (pc.useTrippySampling != 0) {
        vec3 acc = vec3(0.0);
        for (int i = 0; i < 3; ++i) {
            float fi = float(i);
            vec2 uv = vUV + vec2(sin(pc.time * 1.5 + fi * 2.1), cos(pc.time * 1.3 + fi * 2.1)) * 0.1;
            acc += texture(albedoTex, uv).rgb;
        }
        albedo = acc / 3.0;
        vec3 tint = vec3(0.5 + 0.5 * sin(pc.time), 0.5 + 0.5 * sin(pc.time + 2.094), 0.5 + 0.5 * sin(pc.time + 4.188));
        albedo *= tint;
    }

    vec3 N = normalize(vNormal);
    vec3 V = normalize(scene.cameraPos.xyz - vWorldPos);

    vec3 result = scene.ambient.rgb * scene.ambient.a * albedo;

    {
        vec3 L = normalize(-dirLight.direction.xyz);
        result += blinnPhong(N, V, L, dirLight.color.rgb, dirLight.direction.w, 1.0, albedo, model.specColor.rgb);
    }

    for (int i = 0; i < scene.lightCounts.x; ++i) {
        PointLight pl = pointLights.points[i];
        vec3 toLight = pl.position.xyz - vWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;
        float atten = 1.0 / (dist * dist + 0.01);
        result += blinnPhong(N, V, L, pl.color.rgb, pl.position.w, atten, albedo, model.specColor.rgb);
    }

    for (int i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight sl = spotLights.spots[i];
        vec3 toLight = sl.position.xyz - vWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;
        vec3 L = toLight / dist;
        float cosTheta = dot(normalize(sl.direction.xyz), -L);
        float coneFactor = smoothstep(sl.params.y, sl.params.x, cosTheta);
        if (coneFactor <= 0.0) continue;
        float atten = (1.0 / (dist * dist + 0.01)) * coneFactor;
        result += blinnPhong(N, V, L, sl.color.rgb, sl.position.w, atten, albedo, model.specColor.rgb);
    }

    outColor = vec4(result, 1.0);
}