#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace {

using namespace graphics::internal;

//Вершина и uniform-структуры

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct alignas(16) SceneUniforms {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 cameraPos;
    glm::vec4 ambient;
    glm::ivec4 lightCounts;
};

struct alignas(16) DirectionalLight {
    glm::vec4 direction;
    glm::vec4 color;
};

struct alignas(16) PointLight {
    glm::vec4 position;
    glm::vec4 color;
};

struct alignas(16) SpotLight {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color;
    glm::vec4 params;
};

struct alignas(16) ModelData {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 specColor;
    glm::vec4 params;
};

struct PushConstants {
    int useTrippySampling;
    float time;
};

struct RenderableObject {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    std::string name;
};


//Геометрия и ресурсы

std::vector<Vertex> gVertices;
std::vector<uint32_t> gIndices;

VkBuffer gVertexBuffer = VK_NULL_HANDLE;
VmaAllocation gVertexAllocation = VK_NULL_HANDLE;
VkBuffer gIndexBuffer = VK_NULL_HANDLE;
VmaAllocation gIndexAllocation = VK_NULL_HANDLE;

VkDescriptorSetLayout gSceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSet gSceneSet = VK_NULL_HANDLE;

VkBuffer gSceneUboBuffer = VK_NULL_HANDLE;
VmaAllocation gSceneUboAllocation = VK_NULL_HANDLE;

VkBuffer gDirLightUboBuffer = VK_NULL_HANDLE;
VmaAllocation gDirLightUboAllocation = VK_NULL_HANDLE;

SceneUniforms gSceneUniforms{};
DirectionalLight gDirLight{
    .direction = glm::vec4(glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f)), 1.0f),
    .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
};

constexpr int MAX_LIGHTS = 16;

std::vector<PointLight> gPointLights;
std::vector<SpotLight>  gSpotLights;

VkBuffer gPointSSBO = VK_NULL_HANDLE;
VmaAllocation gPointSSBOAlloc = VK_NULL_HANDLE;

VkBuffer gSpotSSBO = VK_NULL_HANDLE;
VmaAllocation gSpotSSBOAlloc = VK_NULL_HANDLE;

VkDescriptorSetLayout gModelSetLayout = VK_NULL_HANDLE;
VkDescriptorSet gModelSet = VK_NULL_HANDLE;

constexpr uint32_t MAX_OBJECTS = 8;

VkBuffer gModelDynamicUBO = VK_NULL_HANDLE;
VmaAllocation gModelDynamicUBOAlloc = VK_NULL_HANDLE;
uint32_t gModelDynamicStride  = 0;

ModelData gModelData{};

std::vector<Texture> gTextures;
std::vector<VkDescriptorSet> gModelSets;
int gSelectedTexture = 0;

PushConstants gPushConstants{ .useTrippySampling = 0, .time = 0.0f };

VkPipelineLayout gPipelineLayout = VK_NULL_HANDLE;
VkPipeline gGraphicsPipeline = VK_NULL_HANDLE;
VkDescriptorPool gDescriptorPool = VK_NULL_HANDLE;

RenderableObject gPyramid;

struct Camera {
    glm::vec3 position = {0.0f, 2.5f, 7.0f};
    float yaw = -90.0f;
    float pitch = -20.0f;
    float fovDeg = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    glm::mat4 view() const {
        glm::vec3 dir;
        dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        dir.y = sin(glm::radians(pitch));
        dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        dir = glm::normalize(dir);

        return glm::lookAt(position, position + dir, glm::vec3(0, 1, 0));
    }
};
Camera gCamera;

struct AnimationState {
    bool  playing = true;
    float spinDegPerSec = 30.0f;
    float spinAngleDeg = 0.0f;
    bool  orbitEnabled   = true;
    float orbitSpeed     = 1.0f;
    float orbitRadius    = 2.0f;
    float orbitAngle     = 0.0f;
    float orbitAngularRate = 2.0f;
};
AnimationState gAnim;

//Вспомогательные функции

std::vector<char> loadBinary(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Не удалось открыть файл шейдера: " << path << '\n';
        return {};
    }
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), static_cast<std::streamsize>(size));
    return data;
}

VkShaderModule makeShaderModule(const std::vector<char>& spirv) {
    if (spirv.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size(),
        .pCode = reinterpret_cast<const uint32_t*>(spirv.data()),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

void allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkBuffer& outBuffer, VmaAllocation& outAllocation) {
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                         | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                    &outBuffer, &outAllocation, nullptr);
}

template<typename T>
void uploadToAllocation(VmaAllocation allocation, const T& value) {
    void* mapped = nullptr;
    if (vmaMapMemory(context.allocator, allocation, &mapped) == VK_SUCCESS) {
        std::memcpy(mapped, &value, sizeof(T));
        vmaUnmapMemory(context.allocator, allocation);
    }
}

void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = graphics::internal::getCommandPool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(context.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        std::cerr << "Неподдерживаемый переход layout'а\n";
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &cmd};
    vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphics_queue);
    vkFreeCommandBuffers(context.device, graphics::internal::getCommandPool(), 1, &cmd);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = graphics::internal::getCommandPool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(context.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &cmd};
    vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphics_queue);
    vkFreeCommandBuffers(context.device, graphics::internal::getCommandPool(), 1, &cmd);
}

void copyBufferToDevice(VkBuffer srcStaging, VkBuffer dstDevice, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = graphics::internal::getCommandPool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(context.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, srcStaging, dstDevice, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphics_queue);
    vkFreeCommandBuffers(context.device, graphics::internal::getCommandPool(), 1, &cmd);
}

bool loadTexture(const std::string& path, const std::string& debugName, Texture& out) {
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Не удалось загрузить текстуру: " << path << '\n';
        return false;
    }
    const VkDeviceSize imageSize = VkDeviceSize(width) * height * 4;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = imageSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo allocInfo{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        vmaCreateBuffer(context.allocator, &bufInfo, &allocInfo, &stagingBuffer, &stagingAlloc, nullptr);
    }
    void* mapped = nullptr;
    vmaMapMemory(context.allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, pixels, imageSize);
    vmaUnmapMemory(context.allocator, stagingAlloc);
    stbi_image_free(pixels);

    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = { uint32_t(width), uint32_t(height), 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    if (vmaCreateImage(context.allocator, &imageInfo, &allocInfo, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAlloc);
        return false;
    }

    transitionImageLayout(out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, out.image, uint32_t(width), uint32_t(height));
    transitionImageLayout(out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAlloc);

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    if (vkCreateImageView(context.device, &viewInfo, nullptr, &out.view) != VK_SUCCESS) {
        return false;
    }

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    };
    if (vkCreateSampler(context.device, &samplerInfo, nullptr, &out.sampler) != VK_SUCCESS) {
        return false;
    }

    out.name = debugName;
    return true;
}

void destroyTexture(Texture& tex) {
    if (tex.sampler) vkDestroySampler(context.device, tex.sampler, nullptr);
    if (tex.view) vkDestroyImageView(context.device, tex.view, nullptr);
    if (tex.image) vmaDestroyImage(context.allocator, tex.image, tex.allocation);
    tex = {};
}

//Генерация пирамиды

void buildPyramidMesh(float baseRadius, float height, int sideCount) {
    gVertices.clear();
    gIndices.clear();

    const float halfHeight = height * 0.5f;
    const glm::vec3 tip{0.0f, halfHeight, 0.0f};

    std::vector<glm::vec3> ring;
    ring.reserve(sideCount);
    for (int i = 0; i < sideCount; ++i) {
        const float angle = 2.0f * glm::pi<float>() * static_cast<float>(i) / static_cast<float>(sideCount);
        ring.emplace_back(baseRadius * std::cos(angle), -halfHeight, baseRadius * std::sin(angle));
    }

    auto pushTriangle = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec2& ua, const glm::vec2& ub, const glm::vec2& uc) {
        const glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        const uint32_t base = static_cast<uint32_t>(gVertices.size());
        gVertices.push_back({a, n, ua});
        gVertices.push_back({b, n, ub});
        gVertices.push_back({c, n, uc});
        gIndices.insert(gIndices.end(), {base, base + 1, base + 2});
    };

    const glm::vec2 uvLeft {0.0f, 0.0f};
    const glm::vec2 uvTop {0.5f, 1.0f};
    const glm::vec2 uvRight {1.0f, 0.0f};
    for (int i = 0; i < sideCount; ++i) {
        const int next = (i + 1) % sideCount;
        pushTriangle(ring[i], tip, ring[next], uvRight, uvTop, uvLeft);
    }

    const glm::vec2 uvBase[4] = {{0,0}, {1,0}, {1,1}, {0,1}};
    for (int i = 1; i + 1 < sideCount; ++i) {
        pushTriangle(ring[0], ring[i], ring[i + 1], uvBase[0], uvBase[(sideCount - i) % sideCount], uvBase[(sideCount - i - 1) % sideCount]);
    }
}

bool createSceneResources() {
    allocateBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, gSceneUboBuffer, gSceneUboAllocation);
    allocateBuffer(sizeof(DirectionalLight), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, gDirLightUboBuffer, gDirLightUboAllocation);
    allocateBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gPointSSBO, gPointSSBOAlloc);
    allocateBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gSpotSSBO, gSpotSSBOAlloc);

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4, .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &gSceneSetLayout) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gDescriptorPool,
        .descriptorSetCount = 1, .pSetLayouts = &gSceneSetLayout,
    };
    if (vkAllocateDescriptorSets(context.device, &allocInfo, &gSceneSet) != VK_SUCCESS)
        return false;

    VkDescriptorBufferInfo sceneInfo {gSceneUboBuffer, 0, sizeof(SceneUniforms)};
    VkDescriptorBufferInfo dirInfo   {gDirLightUboBuffer, 0, sizeof(DirectionalLight)};
    VkDescriptorBufferInfo pointInfo {gPointSSBO, 0, sizeof(PointLight) * MAX_LIGHTS};
    VkDescriptorBufferInfo spotInfo  {gSpotSSBO,  0, sizeof(SpotLight)  * MAX_LIGHTS};

    VkWriteDescriptorSet writes[4]{};
    writes[0] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = gSceneSet, .dstBinding = 0,
                 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 .pBufferInfo = &sceneInfo};
    writes[1] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = gSceneSet, .dstBinding = 1,
                 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 .pBufferInfo = &pointInfo};
    writes[2] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = gSceneSet, .dstBinding = 2,
                 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 .pBufferInfo = &spotInfo};
    writes[3] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = gSceneSet, .dstBinding = 3,
                 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 .pBufferInfo = &dirInfo};
    vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);

    return true;
}

bool createModelResources() {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(context.physical_device, &props);
    const uint32_t align = props.limits.minUniformBufferOffsetAlignment;
    gModelDynamicStride = (sizeof(ModelData) + align - 1) / align * align;

    allocateBuffer(gModelDynamicStride * MAX_OBJECTS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, gModelDynamicUBO, gModelDynamicUBOAlloc);

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &gModelSetLayout) != VK_SUCCESS) return false;

    gModelSets.clear();
    for (size_t i = 0; i < gTextures.size(); ++i) {
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = gDescriptorPool,
            .descriptorSetCount = 1, .pSetLayouts = &gModelSetLayout,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(context.device, &allocInfo, &set) != VK_SUCCESS)
            return false;

        VkDescriptorBufferInfo bufInfo{
            .buffer = gModelDynamicUBO, .offset = 0, .range = sizeof(ModelData),
        };
        VkDescriptorImageInfo imgInfo{
            .sampler = gTextures[i].sampler,
            .imageView = gTextures[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet writes[2]{};
        writes[0] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     .dstSet = set, .dstBinding = 0,
                     .descriptorCount = 1,
                     .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                     .pBufferInfo = &bufInfo};
        writes[1] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     .dstSet = set, .dstBinding = 1,
                     .descriptorCount = 1,
                     .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                     .pImageInfo = &imgInfo};
        vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
        gModelSets.push_back(set);
    }
    return true;
}

//Создание графического пайплайна

bool createPipeline(VkShaderModule vertexShader, VkShaderModule fragmentShader) {
    VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragmentShader,
            .pName = "main",
        },
    };

    VkVertexInputBindingDescription binding{
        0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX
    };
    std::array<VkVertexInputAttributeDescription, 3> attributes{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)};

    VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depth{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
    };

    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    VkDescriptorSetLayout setLayouts[] = {gSceneSetLayout, gModelSetLayout};
    VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };

    if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &gPipelineLayout) != VK_SUCCESS) {
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &assembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = gPipelineLayout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };
    return vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &gGraphicsPipeline) == VK_SUCCESS;
}

//Трансформации

//1.2
glm::mat4 buildModelMatrix(const RenderableObject& obj, const glm::vec3& worldPos) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), worldPos);
    model = glm::rotate(model, glm::radians(obj.rotation.x), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(obj.rotation.y), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(obj.rotation.z), glm::vec3(0, 0, 1));
    return glm::scale(model, obj.scale);
}

//1.4
void uploadScene() {
    gSceneUniforms.lightCounts.x = static_cast<int>(gPointLights.size());
    gSceneUniforms.lightCounts.y = static_cast<int>(gSpotLights.size());

    uploadToAllocation(gSceneUboAllocation, gSceneUniforms);
    uploadToAllocation(gDirLightUboAllocation, gDirLight);

    if (!gPointLights.empty()) {
        void* data = nullptr;
        if (vmaMapMemory(context.allocator, gPointSSBOAlloc, &data) == VK_SUCCESS) {
            std::memcpy(data, gPointLights.data(), gPointLights.size() * sizeof(PointLight));
            vmaUnmapMemory(context.allocator, gPointSSBOAlloc);
        }
    }
    if (!gSpotLights.empty()) {
        void* data = nullptr;
        if (vmaMapMemory(context.allocator, gSpotSSBOAlloc, &data) == VK_SUCCESS) {
            std::memcpy(data, gSpotLights.data(), gSpotLights.size() * sizeof(SpotLight));
            vmaUnmapMemory(context.allocator, gSpotSSBOAlloc);
        }
    }
}

void uploadModel() {
    void* data = nullptr;
    if (vmaMapMemory(context.allocator, gModelDynamicUBOAlloc, &data) != VK_SUCCESS)
        return;
    std::memcpy(data, &gModelData, sizeof(ModelData));
    vmaUnmapMemory(context.allocator, gModelDynamicUBOAlloc);
}

} // namespace

//  Публичный API приложения

bool application::initialize() {
    buildPyramidMesh(1.0f, 2.0f, 4);

    if (gVertices.empty() || gIndices.empty()) {
        std::cerr << "Не удалось построить геометрию пирамиды\n";
        return false;
    }

    const VkDeviceSize vertexBytes = sizeof(Vertex) * gVertices.size();
    const VkDeviceSize indexBytes = sizeof(uint32_t) * gIndices.size();

    {
        VkBuffer stagingBuf; VmaAllocation stagingAlloc;
        VkBufferCreateInfo bufInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = vertexBytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo allocInfo{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO};
        vmaCreateBuffer(context.allocator, &bufInfo, &allocInfo, &stagingBuf, &stagingAlloc, nullptr);

        void* mapped = nullptr;
        vmaMapMemory(context.allocator, stagingAlloc, &mapped);
        std::memcpy(mapped, gVertices.data(), vertexBytes);
        vmaUnmapMemory(context.allocator, stagingAlloc);

        VkBufferCreateInfo devBufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = vertexBytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo devAllocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        vmaCreateBuffer(context.allocator, &devBufInfo, &devAllocInfo, &gVertexBuffer, &gVertexAllocation, nullptr);

        copyBufferToDevice(stagingBuf, gVertexBuffer, vertexBytes);
        vmaDestroyBuffer(context.allocator, stagingBuf, stagingAlloc);
    }

    {
        VkBuffer stagingBuf; VmaAllocation stagingAlloc;
        VkBufferCreateInfo bufInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = indexBytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo allocInfo{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO};
        vmaCreateBuffer(context.allocator, &bufInfo, &allocInfo, &stagingBuf, &stagingAlloc, nullptr);

        void* mapped = nullptr;
        vmaMapMemory(context.allocator, stagingAlloc, &mapped);
        std::memcpy(mapped, gIndices.data(), indexBytes);
        vmaUnmapMemory(context.allocator, stagingAlloc);

        VkBufferCreateInfo devBufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = indexBytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo devAllocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        vmaCreateBuffer(context.allocator, &devBufInfo, &devAllocInfo, &gIndexBuffer, &gIndexAllocation, nullptr);

        copyBufferToDevice(stagingBuf, gIndexBuffer, indexBytes);
        vmaDestroyBuffer(context.allocator, stagingBuf, stagingAlloc);
    }

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 10,
        .poolSizeCount = 4,
        .pPoolSizes = poolSizes,
    };
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &gDescriptorPool) != VK_SUCCESS) return false;

    if (!createSceneResources()) { std::cerr << "Scene resources failed\n"; return false; }

    if (gTextures.empty()) {
        Texture tex;
        if (!loadTexture("textures/concrete.png", "Concrete", tex)) {
            std::cerr << "Не удалось загрузить concrete.png\n"; return false;
        }
        gTextures.push_back(std::move(tex));

        if (!loadTexture("textures/king.png", "King", tex)) {
            std::cerr << "Не удалось загрузить king.png\n"; return false;
        }
        gTextures.push_back(std::move(tex));

        if (!loadTexture("textures/huang.png", "Huang", tex)) {
            std::cerr << "Не удалось загрузить huang.png\n"; return false;
        }
        gTextures.push_back(std::move(tex));
    }

    if (!createModelResources()) { std::cerr << "Model resources failed\n"; return false; }

    gPointLights.push_back({
        .position = glm::vec4(2.0f, 1.5f, 1.0f, 3.0f),
        .color    = glm::vec4(1.0f, 0.6f, 0.2f, 0.0f),
    });
    gSpotLights.push_back({
        .position  = glm::vec4(-2.0f, 3.0f, 2.0f, 5.0f),
        .direction = glm::vec4(0.0f, -1.0f, -0.5f, 0.0f),
        .color     = glm::vec4(0.4f, 0.8f, 1.0f, 0.0f),
        .params    = glm::vec4(std::cos(glm::radians(15.0f)), std::cos(glm::radians(25.0f)), 0.0f, 0.0f),
    });

    gSceneUniforms.lightCounts.x = static_cast<int>(gPointLights.size());
    gSceneUniforms.lightCounts.y = static_cast<int>(gSpotLights.size());

    const auto vertexSpirv = loadBinary("shaders/vert.spv");
    const auto fragmentSpirv = loadBinary("shaders/frag.spv");
    const auto vertexShader = makeShaderModule(vertexSpirv);
    const auto fragmentShader = makeShaderModule(fragmentSpirv);

    if (vertexShader == VK_NULL_HANDLE || fragmentShader == VK_NULL_HANDLE) {
        std::cerr << "Не удалось загрузить модули шейдеров\n";
        return false;
    }

    const bool pipelineOk = createPipeline(vertexShader, fragmentShader);
    vkDestroyShaderModule(context.device, vertexShader, nullptr);
    vkDestroyShaderModule(context.device, fragmentShader, nullptr);
    if (!pipelineOk) {
        std::cerr << "Не удалось создать графический пайплайн\n";
        return false;
    }

    return true;
}

void application::update(double time) {
    static double lastTimestamp = -1.0;
    float delta = 0.0f;
    if (lastTimestamp >= 0.0) {
        delta = static_cast<float>(time - lastTimestamp);
        if (delta > 0.1f) delta = 0.1f;
    }
    lastTimestamp = time;

    //1.3
    if (gAnim.playing) {
        gAnim.spinAngleDeg += gAnim.spinDegPerSec * delta;
        if (gAnim.spinAngleDeg >= 360.0f) gAnim.spinAngleDeg -= 360.0f;

        if (gAnim.orbitEnabled) {
            gAnim.orbitAngle += delta * gAnim.orbitSpeed;
        }
    }
    gPyramid.rotation.y = gAnim.spinAngleDeg;

    //Панель управления
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 130), ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera");
    ImGui::DragFloat3("Position", glm::value_ptr(gCamera.position), 0.1f);
    ImGui::SliderFloat("Yaw", &gCamera.yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &gCamera.pitch, -89.0f, 89.0f);
    ImGui::SliderFloat("FOV", &gCamera.fovDeg, 1.0f, 120.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 150), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 110), ImGuiCond_FirstUseEver);
    ImGui::Begin("Animation");
    ImGui::Checkbox("Spin", &gAnim.playing);
    ImGui::SliderFloat("Speed (deg/s)", &gAnim.spinDegPerSec, 0.0f, 180.0f);
    ImGui::Separator();
    ImGui::Checkbox   ("Orbit", &gAnim.orbitEnabled);
    ImGui::SliderFloat("Orbit Speed", &gAnim.orbitSpeed,  0.1f, 3.0f);
    ImGui::SliderFloat("Orbit Radius", &gAnim.orbitRadius, 0.5f, 5.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 270), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 110), ImGuiCond_FirstUseEver);
    ImGui::Begin("Material");
    ImGui::Text("Texture");
    if (ImGui::BeginCombo("##tex", gTextures[gSelectedTexture].name.c_str())) {
        for (int i = 0; i < (int)gTextures.size(); ++i) {
            const bool selected = (i == gSelectedTexture);
            if (ImGui::Selectable(gTextures[i].name.c_str(), selected))
                gSelectedTexture = i;
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::Checkbox("Trippy Sampling", (bool*)&gPushConstants.useTrippySampling);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 390), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 90), ImGuiCond_FirstUseEver);
    ImGui::Begin("Directional Light");
    ImGui::SliderFloat3("Direction", glm::value_ptr(gDirLight.direction), -1.0f, 1.0f);
    ImGui::ColorEdit3 ("Color", glm::value_ptr(gDirLight.color));
    ImGui::SliderFloat("Intensity", &gDirLight.direction.w, 0.0f, 10.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 490), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 110), ImGuiCond_FirstUseEver);
    ImGui::Begin("Ambient");
    ImGui::ColorEdit3("Color", glm::value_ptr(gSceneUniforms.ambient));
    ImGui::SliderFloat("Intensity", &gSceneUniforms.ambient.w, 0.0f, 1.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(360, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 360), ImGuiCond_FirstUseEver);
    ImGui::Begin("Point Lights");

    if (ImGui::Button("Add Point Light") &&
        gPointLights.size() < static_cast<size_t>(MAX_LIGHTS)) {
        gPointLights.push_back({
            .position = glm::vec4(1.0f, 1.0f, 0.0f, 2.0f),
            .color    = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
        });
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear") && !gPointLights.empty()) {
        gPointLights.clear();
    }

    for (size_t i = 0; i < gPointLights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::Separator();
        ImGui::Text("Light #%zu", i);
        ImGui::DragFloat3("Position", glm::value_ptr(gPointLights[i].position), 0.1f);
        ImGui::ColorEdit3("Color", glm::value_ptr(gPointLights[i].color));
        ImGui::SliderFloat("Intensity", &gPointLights[i].position.w, 0.0f, 20.0f);
        if (ImGui::Button("Remove")) {
            gPointLights.erase(gPointLights.begin() + i);
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(360, 380), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 360), ImGuiCond_FirstUseEver);
    ImGui::Begin("Spot Lights");

    if (ImGui::Button("Add Spot Light") &&
        gSpotLights.size() < static_cast<size_t>(MAX_LIGHTS)) {
        gSpotLights.push_back({
            .position  = glm::vec4(0.0f, 5.0f, 0.0f, 3.0f),
            .direction = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f),
            .color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
            .params = glm::vec4(std::cos(glm::radians(15.0f)), std::cos(glm::radians(25.0f)), 0.0f, 0.0f),
        });
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear") && !gSpotLights.empty()) {
        gSpotLights.clear();
    }

    for (size_t i = 0; i < gSpotLights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::Separator();
        ImGui::Text("Spot #%zu", i);
        ImGui::DragFloat3("Position", glm::value_ptr(gSpotLights[i].position), 0.1f);
        ImGui::DragFloat3("Direction", glm::value_ptr(gSpotLights[i].direction), 0.05f);
        ImGui::ColorEdit3("Color", glm::value_ptr(gSpotLights[i].color));
        ImGui::SliderFloat("Intensity", &gSpotLights[i].position.w, 0.0f, 20.0f);

        float innerDeg = glm::degrees(std::acos(gSpotLights[i].params.x));
        float outerDeg = glm::degrees(std::acos(gSpotLights[i].params.y));
        if (ImGui::SliderFloat("Inner Angle", &innerDeg, 1.0f, 89.0f)) {
            gSpotLights[i].params.x = std::cos(glm::radians(innerDeg));
        }
        if (ImGui::SliderFloat("Outer Angle", &outerDeg, 1.0f, 89.0f)) {
            gSpotLights[i].params.y = std::cos(glm::radians(outerDeg));
        }

        if (ImGui::Button("Remove")) {
            gSpotLights.erase(gSpotLights.begin() + i);
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }
    ImGui::End();

    const float aspect = static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height);

    gSceneUniforms.view = gCamera.view();
    gSceneUniforms.proj = glm::perspective(glm::radians(gCamera.fovDeg), aspect, gCamera.nearPlane, gCamera.farPlane);
    gSceneUniforms.proj[1][1] *= -1.0f;
    gSceneUniforms.cameraPos = glm::vec4(gCamera.position, 1.0f);

    uploadScene();

    glm::vec3 worldPos = gPyramid.position;
    if (gAnim.orbitEnabled && gAnim.playing) {
        worldPos.x += std::cos(gAnim.orbitAngle * gAnim.orbitAngularRate) * gAnim.orbitRadius;
        worldPos.z += std::sin(gAnim.orbitAngle * gAnim.orbitAngularRate) * gAnim.orbitRadius;
    }

    gModelData.model = buildModelMatrix(gPyramid, worldPos);
    gModelData.normalMatrix = glm::transpose(
        glm::inverse(glm::mat4(glm::mat3(gModelData.model))));
    gModelData.specColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    gModelData.params = glm::vec4(32.0f, 0.0f, 0.0f, 0.0f);
    uploadModel();
}

void application::render(const graphics::internal::FrameData& fd) {
    VkCommandBuffer cmd = fd.command_buffer;

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo passBegin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {.extent = context.swapchain_extent},
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(cmd, &passBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gGraphicsPipeline);

    VkViewport viewport{
        0.0f, 0.0f,
        static_cast<float>(context.swapchain_extent.width),
        static_cast<float>(context.swapchain_extent.height),
        0.0f, 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, context.swapchain_extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &gVertexBuffer, offsets);
    vkCmdBindIndexBuffer(cmd, gIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gPipelineLayout, 0, 1, &gSceneSet, 0, nullptr);
    gPushConstants.time = static_cast<float>(glfwGetTime());
    vkCmdPushConstants(cmd, gPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(gPushConstants), &gPushConstants);

    VkDescriptorSet modelSet = gModelSets[gSelectedTexture];
    uint32_t dynamicOffset = 0;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gPipelineLayout, 1, 1, &modelSet, 1, &dynamicOffset);
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(gIndices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void application::shutdown() {
    vmaDestroyBuffer(context.allocator, gSceneUboBuffer, gSceneUboAllocation);
    vmaDestroyBuffer(context.allocator, gDirLightUboBuffer, gDirLightUboAllocation);
    vmaDestroyBuffer(context.allocator, gModelDynamicUBO, gModelDynamicUBOAlloc);
    for (auto& tex : gTextures) destroyTexture(tex);
    gTextures.clear();
    gModelSets.clear();
    vmaDestroyBuffer(context.allocator, gPointSSBO, gPointSSBOAlloc);
    vmaDestroyBuffer(context.allocator, gSpotSSBO, gSpotSSBOAlloc);
    vmaDestroyBuffer(context.allocator, gVertexBuffer, gVertexAllocation);
    vmaDestroyBuffer(context.allocator, gIndexBuffer, gIndexAllocation);
    vkDestroyDescriptorPool(context.device, gDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, gModelSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, gSceneSetLayout, nullptr);
    vkDestroyPipeline(context.device, gGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, gPipelineLayout, nullptr);
}