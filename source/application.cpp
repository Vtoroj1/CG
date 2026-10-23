#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using namespace graphics::internal;

//Вершина и uniform-структуры

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct CameraUniforms {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

struct ColorUniform {
    glm::vec4 rgba;
};

//Объект сцены

struct RenderableObject {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec4 tint{1.0f};

    VkBuffer mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation mvpAllocation = VK_NULL_HANDLE;
    VkBuffer tintBuffer = VK_NULL_HANDLE;
    VmaAllocation tintAllocation = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
};

//Геометрия и ресурсы

std::vector<Vertex> gVertices;
std::vector<uint32_t> gIndices;

VkBuffer gVertexBuffer = VK_NULL_HANDLE;
VmaAllocation gVertexAllocation = VK_NULL_HANDLE;
VkBuffer gIndexBuffer = VK_NULL_HANDLE;
VmaAllocation gIndexAllocation = VK_NULL_HANDLE;

VkDescriptorSetLayout gDescriptorLayout = VK_NULL_HANDLE;
VkPipelineLayout gPipelineLayout = VK_NULL_HANDLE;
VkPipeline gGraphicsPipeline = VK_NULL_HANDLE;
VkDescriptorPool gDescriptorPool = VK_NULL_HANDLE;

RenderableObject gPyramid;

//Состояние сцены и анимации

struct SceneState {
    bool perspective = true;
    bool playing = true;
    float orbitSpeed = 1.0f;
    float orbitRadius = 2.0f;
    float orbitAngle = 0.0f;
    float orbitAngularRate = 2.0f;
    float spinDegPerSecond = 30.0f;
};

SceneState gScene;

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

    // Процедурный цвет по локальной позиции (1.5)
    auto vertexTint = [](const glm::vec3& p) {
        return p * 0.5f + 0.5f;
    };

    auto pushTriangle = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        const uint32_t base = static_cast<uint32_t>(gVertices.size());
        gVertices.push_back({a, vertexTint(a)});
        gVertices.push_back({b, vertexTint(b)});
        gVertices.push_back({c, vertexTint(c)});
        gIndices.insert(gIndices.end(), {base, base + 1, base + 2});
    };

    for (int i = 0; i < sideCount; ++i) {
        const int next = (i + 1) % sideCount;
        pushTriangle(ring[i], tip, ring[next]);
    }
    for (int i = 1; i + 1 < sideCount; ++i) {
        pushTriangle(ring[0], ring[i], ring[i + 1]);
    }
}

//Настройка объекта сцены

void createObjectUniforms(RenderableObject& obj) {
    allocateBuffer(sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   obj.mvpBuffer, obj.mvpAllocation);
    allocateBuffer(sizeof(ColorUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   obj.tintBuffer, obj.tintAllocation);

    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gDescriptorLayout,
    };
    vkAllocateDescriptorSets(context.device, &allocateInfo, &obj.descriptor);

    VkDescriptorBufferInfo mvpBinding{obj.mvpBuffer, 0, sizeof(CameraUniforms)};
    VkDescriptorBufferInfo tintBinding{obj.tintBuffer, 0, sizeof(ColorUniform)};

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptor,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &mvpBinding,
    };
    writes[1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptor,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &tintBinding,
    };
    vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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
    std::array<VkVertexInputAttributeDescription, 2> attributes{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};

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

    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &gDescriptorLayout,
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
void uploadObjectUniforms(RenderableObject& obj, const CameraUniforms& mvp) {
    uploadToAllocation(obj.mvpAllocation, mvp);
    uploadToAllocation(obj.tintAllocation, ColorUniform{obj.tint});
}

} // namespace

//  Публичный API приложения

bool application::initialize() {
    buildPyramidMesh(1.0f, 2.0f, 4);

    if (gVertices.empty() || gIndices.empty()) {
        std::cerr << "Не удалось построить геометрию пирамиды\n";
        return false;
    }

    allocateBuffer(sizeof(Vertex) * gVertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gVertexBuffer, gVertexAllocation);
    allocateBuffer(sizeof(uint32_t) * gIndices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gIndexBuffer, gIndexAllocation);

    void* mapped = nullptr;
    if (vmaMapMemory(context.allocator, gVertexAllocation, &mapped) == VK_SUCCESS) {
        std::memcpy(mapped, gVertices.data(), sizeof(Vertex) * gVertices.size());
        vmaUnmapMemory(context.allocator, gVertexAllocation);
    }
    if (vmaMapMemory(context.allocator, gIndexAllocation, &mapped) == VK_SUCCESS) {
        std::memcpy(mapped, gIndices.data(), sizeof(uint32_t) * gIndices.size());
        vmaUnmapMemory(context.allocator, gIndexAllocation);
    }

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &gDescriptorLayout) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &gDescriptorPool) != VK_SUCCESS) {
        return false;
    }

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

    createObjectUniforms(gPyramid);
    gPyramid.tint = {1.0f, 1.0f, 0.1f, 1.0f};
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
    if (gScene.playing) {
        gScene.orbitAngle += delta * gScene.orbitSpeed;
        gPyramid.rotation.y += gScene.spinDegPerSecond * delta;
        if (gPyramid.rotation.y >= 360.0f) gPyramid.rotation.y -= 360.0f;
    }

    //Панель управления

    ImGui::Begin("Lab #1 - Pyramid");

    ImGui::TextUnformatted("Projection");
    int mode = gScene.perspective ? 0 : 1;
    ImGui::RadioButton("Perspective", &mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Orthographic", &mode, 1);
    gScene.perspective = (mode == 0);

    ImGui::Separator();
    ImGui::TextUnformatted("Transform");
    ImGui::DragFloat3("Position", glm::value_ptr(gPyramid.position), 0.05f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(gPyramid.rotation), 0.5f);
    ImGui::DragFloat3("Scale", glm::value_ptr(gPyramid.scale),    0.05f, 0.1f, 10.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Orbit animation");
    ImGui::Checkbox("Playing", &gScene.playing);
    ImGui::SliderFloat("Speed",  &gScene.orbitSpeed,  0.1f, 3.0f);
    ImGui::SliderFloat("Radius", &gScene.orbitRadius, 0.5f, 5.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Color");
    ImGui::ColorEdit4("Tint", glm::value_ptr(gPyramid.tint));

    ImGui::End();

    //Матрицы камеры

    const float aspect = static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height);

    const glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 2.5f, 7.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));

    //1.1
    glm::mat4 projection = gScene.perspective
        ? glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f)
        : glm::ortho(-aspect * 3.5f, aspect * 3.5f, -3.5f, 3.5f, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;

    //Позиция с учётом орбиты

    //1.3
    glm::vec3 worldPos = gPyramid.position;
    if (gScene.playing) {
        worldPos.x += std::cos(gScene.orbitAngle * gScene.orbitAngularRate) * gScene.orbitRadius;
        worldPos.z += std::sin(gScene.orbitAngle * gScene.orbitAngularRate) * gScene.orbitRadius;
    } else {
        worldPos.x += gScene.orbitRadius;
    }

    CameraUniforms mvp{
        .model = buildModelMatrix(gPyramid, worldPos),
        .view = view,
        .projection = projection,
    };
    uploadObjectUniforms(gPyramid, mvp);
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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gPipelineLayout, 0, 1, &gPyramid.descriptor, 0, nullptr);
    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(gIndices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void application::shutdown() {
    vmaDestroyBuffer(context.allocator, gPyramid.mvpBuffer, gPyramid.mvpAllocation);
    vmaDestroyBuffer(context.allocator, gPyramid.tintBuffer, gPyramid.tintAllocation);
    vmaDestroyBuffer(context.allocator, gVertexBuffer, gVertexAllocation);
    vmaDestroyBuffer(context.allocator, gIndexBuffer, gIndexAllocation);
    vkDestroyDescriptorPool(context.device, gDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, gDescriptorLayout, nullptr);
    vkDestroyPipeline(context.device, gGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, gPipelineLayout, nullptr);
}