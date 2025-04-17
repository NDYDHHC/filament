/*
* Copyright (C) 2025 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TNT_FILAMENT_BACKEND_WEBGPUPIPELINECACHE_H
#define TNT_FILAMENT_BACKEND_WEBGPUPIPELINECACHE_H

#include <backend/DriverEnums.h>

#include <utils/FixedCapacityVector.h>

#include <tsl/robin_map.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <array>

namespace filament::backend {

/**
 * The render pipeline requirements is a POD that represents all currently bound states that
 * form the immutable wgpu::RenderPipeline object.
 */
struct WebGPURenderPipelineRequirements final {
    wgpu::ShaderModule vertexShaderModule = nullptr;
    wgpu::ShaderModule fragmentShaderModule = nullptr;
    // using array instead of vertex here to avoid more heap allocations
    // (but could be refactored as vertex otherwise)
    std::array<wgpu::VertexAttribute, MAX_VERTEX_ATTRIBUTE_COUNT> vertexAttributes = {};
    std::array<wgpu::VertexBufferLayout, MAX_VERTEX_BUFFER_COUNT> vertexBufferLayouts = {};
    size_t vertexBufferCount = 0;
    // max constants unknown
    // (otherwise could possibly be an array to avoid heap allocations if small enough)
    utils::FixedCapacityVector<wgpu::ConstantEntry> constants = {};
    wgpu::PrimitiveTopology topology = wgpu::PrimitiveTopology::Undefined;
    wgpu::CullMode cullMode = wgpu::CullMode::Undefined;
    wgpu::FrontFace frontFace = wgpu::FrontFace::Undefined;
    bool blendEnable = false;
    wgpu::OptionalBool depthWriteEnabled = wgpu::OptionalBool::Undefined;
    bool alphaToCoverageEnabled = false;
    wgpu::BlendState blendState = {};
    wgpu::ColorWriteMask colorWriteMask = wgpu::ColorWriteMask::All;
    uint32_t multisampleCount = 0;
    bool unclippedDepth = false;
    uint8_t colorTargetCount = 0;
    wgpu::CompareFunction depthCompare = wgpu::CompareFunction::Undefined;
    int32_t depthBias = 0;
    float depthBiasSlopeScale = 0.0f;
    wgpu::PipelineLayout layout = nullptr;
    wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
    wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined;

    struct Hash {
        [[nodiscard]] size_t operator()(WebGPURenderPipelineRequirements const&) const;
    };

    struct Equal {
        [[nodiscard]] bool operator()(WebGPURenderPipelineRequirements const&,
                WebGPURenderPipelineRequirements const&) const;
    };
};

class WebGPUPipelineCache final {
public:
    WebGPUPipelineCache() = default;
    WebGPUPipelineCache(WebGPUPipelineCache&) = delete;
    WebGPUPipelineCache& operator=(WebGPUPipelineCache&) = delete;
    WebGPUPipelineCache(WebGPUPipelineCache&&) = delete;
    WebGPUPipelineCache& operator=(WebGPUPipelineCache&&) = delete;
    ~WebGPUPipelineCache();

    [[nodiscard]] wgpu::RenderPipeline const& getOrCreateRenderPipeline(wgpu::Device&,
            WebGPURenderPipelineRequirements&);

    void gc();

private:
    [[nodiscard]] static wgpu::RenderPipeline createRenderPipeline(wgpu::Device&,
            WebGPURenderPipelineRequirements&);

    uint64_t mGcCount = 0;

    struct RenderPipelineCacheEntry final {
        wgpu::RenderPipeline pipeline = nullptr;
        uint64_t lastGcCountWhenUsed = 0;
    };

    tsl::robin_map<WebGPURenderPipelineRequirements, RenderPipelineCacheEntry,
            WebGPURenderPipelineRequirements::Hash, WebGPURenderPipelineRequirements::Equal>
            mRenderPipelines = {};
};

}// namespace filament::backend

#endif// TNT_FILAMENT_BACKEND_WEBGPUPIPELINECACHE_H
