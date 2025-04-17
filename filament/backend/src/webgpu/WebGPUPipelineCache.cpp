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

#include "WebGPUPipelineCache.h"

#include "WebGPUConstants.h"

#include "backend/TargetBufferInfo.h"

#include <utils/debug.h>
#include <utils/Hash.h>

#include <webgpu/webgpu_cpp.h>

#include <string_view>

namespace filament::backend {

WebGPUPipelineCache::~WebGPUPipelineCache() { mRenderPipelines.clear(); }

// TODO this seems WAY too slow AND error-prone to maintain over time
// consider murmer3, which may need some refactoring of the requirements stuct a bit
size_t WebGPURenderPipelineRequirements::Hash::operator()(
        WebGPURenderPipelineRequirements const& r) const {
    using namespace utils;
    size_t h = 0;
    // hash scalar values
    hash::combine(h,
            r.vertexShaderModule.Get(),
            r.fragmentShaderModule.Get(),
            r.vertexBufferCount,
            r.topology,
            r.cullMode,
            r.frontFace,
            r.blendEnable,
            r.depthWriteEnabled,
            r.alphaToCoverageEnabled,
            r.blendState.color.operation,
            r.blendState.color.srcFactor,
            r.blendState.color.dstFactor,
            r.blendState.alpha.operation,
            r.blendState.alpha.srcFactor,
            r.blendState.alpha.dstFactor,
            r.colorWriteMask,
            r.multisampleCount,
            r.unclippedDepth,
            r.colorTargetCount,
            r.depthCompare,
            r.depthBias,
            r.depthBiasSlopeScale,
            r.layout.Get(),
            r.colorFormat,
            r.depthFormat);
    // hash collections
    for (auto const& vAttribute: r.vertexAttributes) {
        hash::combine(h, vAttribute.format, vAttribute.offset, vAttribute.shaderLocation);
    }
    for (size_t i = 0; i < r.vertexBufferCount; i++) {
        wgpu::VertexBufferLayout const& vbLayout = r.vertexBufferLayouts[i];
        hash::combine(h, vbLayout.stepMode, vbLayout.arrayStride, vbLayout.attributeCount);
        for (size_t attrIndex = 0; attrIndex < vbLayout.attributeCount; attrIndex++) {
            hash::combine(h,
                    vbLayout.attributes[attrIndex].format,
                    vbLayout.attributes[attrIndex].offset,
                    vbLayout.attributes[attrIndex].shaderLocation);
        }
    }
    for (auto const& constant: r.constants) {
        hash::combine(h, std::string_view(constant.key), constant.value);
    }
    return h;
}

// TODO this is WAY too slow for fast comparison checks AND error-prone to maintain over time
// consider memcmp, which may need some refactoring of the requirements stuct a bit
bool WebGPURenderPipelineRequirements::Equal::operator()(
        WebGPURenderPipelineRequirements const& first,
        WebGPURenderPipelineRequirements const& second) const {
    const bool scalarsMatch =
            first.vertexShaderModule.Get() == second.vertexShaderModule.Get() &&
            first.fragmentShaderModule.Get() == second.fragmentShaderModule.Get() &&
            first.vertexBufferCount == second.vertexBufferCount &&
            first.topology == second.topology &&
            first.cullMode == second.cullMode &&
            first.frontFace == second.frontFace &&
            first.blendEnable == second.blendEnable &&
            first.depthWriteEnabled == second.depthWriteEnabled &&
            first.alphaToCoverageEnabled == second.alphaToCoverageEnabled &&
            first.blendState.color.operation == second.blendState.color.operation &&
            first.blendState.color.srcFactor == second.blendState.color.srcFactor &&
            first.blendState.color.dstFactor == second.blendState.color.dstFactor &&
            first.blendState.alpha.operation == second.blendState.alpha.operation &&
            first.blendState.alpha.srcFactor == second.blendState.alpha.srcFactor &&
            first.blendState.alpha.dstFactor == second.blendState.alpha.dstFactor &&
            first.colorWriteMask == second.colorWriteMask &&
            first.multisampleCount == second.multisampleCount &&
            first.unclippedDepth == second.unclippedDepth &&
            first.colorTargetCount == second.colorTargetCount &&
            first.depthCompare == second.depthCompare &&
            first.depthBias == second.depthBias &&
            first.depthBiasSlopeScale == second.depthBiasSlopeScale &&
            first.layout.Get() == second.layout.Get() &&
            first.colorFormat == second.colorFormat &&
            first.depthFormat == second.depthFormat;
    if (!scalarsMatch) {
        return false;
    }
    // compare collections
    for (size_t vAttrIndex = 0; vAttrIndex < first.vertexAttributes.size(); vAttrIndex++) {
        wgpu::VertexAttribute const& firstVAttr = first.vertexAttributes[vAttrIndex];
        wgpu::VertexAttribute const& secondVAttr = second.vertexAttributes[vAttrIndex];
        if (firstVAttr.format != secondVAttr.format) {
            return false;
        }
        if (firstVAttr.offset != secondVAttr.offset) {
            return false;
        }
        if (firstVAttr.shaderLocation != secondVAttr.shaderLocation) {
            return false;
        }
    }
    for (size_t vblIndex = 0; vblIndex < first.vertexBufferCount; vblIndex++) {
        wgpu::VertexBufferLayout const& firstVBLayout = first.vertexBufferLayouts[vblIndex];
        wgpu::VertexBufferLayout const& secondVBLayout = second.vertexBufferLayouts[vblIndex];
        if (firstVBLayout.stepMode != secondVBLayout.stepMode) {
            return false;
        }
        if (firstVBLayout.arrayStride != secondVBLayout.arrayStride) {
            return false;
        }
        if (firstVBLayout.attributeCount != secondVBLayout.attributeCount) {
            return false;
        }
        for (size_t vAttrIndex = 0; vAttrIndex < firstVBLayout.attributeCount; vAttrIndex++) {
            wgpu::VertexAttribute const& firstVAttr = firstVBLayout.attributes[vAttrIndex];
            wgpu::VertexAttribute const& secondVAttr = secondVBLayout.attributes[vAttrIndex];
            if (firstVAttr.format != secondVAttr.format) {
                return false;
            }
            if (firstVAttr.offset != secondVAttr.offset) {
                return false;
            }
            if (firstVAttr.shaderLocation != secondVAttr.shaderLocation) {
                return false;
            }
        }
        if (first.constants.size() != second.constants.size()) {
            return false;
        }
        for (size_t constIndex = 0; constIndex < first.constants.size(); constIndex++) {
            wgpu::ConstantEntry const& firstConst = first.constants[constIndex];
            wgpu::ConstantEntry const& secondConst = second.constants[constIndex];
            if (std::string_view(firstConst.key) != std::string_view(secondConst.key)) {
                return false;
            }
            if (firstConst.value != secondConst.value) {
                return false;
            }
        }
    }
    return true;
}

wgpu::RenderPipeline const& WebGPUPipelineCache::getOrCreateRenderPipeline(wgpu::Device& device,
        WebGPURenderPipelineRequirements& reqs) {
    if (auto pipelineIter = mRenderPipelines.find(reqs); pipelineIter != mRenderPipelines.end()) {
        // pipeline is already in the cache!
        auto& cacheEntry = pipelineIter.value();
        cacheEntry.lastGcCountWhenUsed = mGcCount;
        return cacheEntry.pipeline;
    }
    // pipeline is not already in the cache...
    RenderPipelineCacheEntry cacheEntry = {
        .pipeline = createRenderPipeline(device, reqs),
        .lastGcCountWhenUsed = 0
    };
    if (cacheEntry.pipeline == nullptr) {
        FWGPU_LOGE << "Failed to create render pipeline!" << utils::io::endl;
    }
    auto [iter, emplaced] = mRenderPipelines.emplace(reqs, cacheEntry);
    if (!emplaced) {
        FWGPU_LOGE << "Failed to emplace the render pipeline into the rabin_map cache?"
                   << utils::io::endl;
    }
    return iter->second.pipeline;
}

void WebGPUPipelineCache::gc() {
    ++mGcCount;
    for (decltype(mRenderPipelines)::const_iterator iter = mRenderPipelines.begin();
            iter != mRenderPipelines.end();) {
        const RenderPipelineCacheEntry& cacheEntry = iter.value();
        if (mGcCount > (cacheEntry.lastGcCountWhenUsed + FWGPU_PIPELINE_MAX_AGE)) {
            iter = mRenderPipelines.erase(iter);
        } else {
            ++iter;
        }
    }
}

wgpu::RenderPipeline WebGPUPipelineCache::createRenderPipeline(wgpu::Device& device,
        filament::backend::WebGPURenderPipelineRequirements& r) {
    assert_invariant(r.layout);
    assert_invariant(r.vertexShaderModule);
    // TODO no idea about stencilFront, stencilBack, stencil read/write masks
    wgpu::DepthStencilState depthStencilState {
        .format = r.depthFormat,
        .depthWriteEnabled = r.depthWriteEnabled,
        .depthCompare = r.depthCompare,
        .depthBias = r.depthBias,
        .depthBiasSlopeScale = r.depthBiasSlopeScale,
        .depthBiasClamp = 0.0f
    };
    // TODO no idea at this point about label
    wgpu::RenderPipelineDescriptor descriptor{
        .layout = r.layout,
        .vertex = {
            .module = r.vertexShaderModule,
            .entryPoint = "main",
            .constantCount = r.constants.size(),
            .constants = r.constants.data(),
            .bufferCount = r.vertexBufferCount,
            .buffers = r.vertexBufferLayouts.data()
        },
        .primitive = {
            .topology = r.topology,
            .stripIndexFormat = wgpu::IndexFormat::Undefined,
            .frontFace = r.frontFace,
            .cullMode = r.cullMode,
            .unclippedDepth = r.unclippedDepth
        },
        .depthStencil = &depthStencilState,
        .multisample = {
            .count = r.multisampleCount,
            .mask = ~0u,
            .alphaToCoverageEnabled = r.alphaToCoverageEnabled
        },
        .fragment = nullptr// will add this below if a fragment shader is provided
    };
    if (r.fragmentShaderModule != nullptr) {
        // avoiding heap allocation with array (although using more space)
        // could alternatively use FixedSizedVector with heap allocation but more efficient space
        std::array<wgpu::ColorTargetState, MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT>
                colorTargets = {};
        // Filament assumes consistent blend state across all color attachments.
        colorTargets[0] = {
            .format = r.colorFormat,
            .blend = r.blendEnable ? &r.blendState : nullptr,
            .writeMask = r.colorWriteMask
        };
        for (size_t targetIndex = 1; targetIndex < colorTargets.size(); targetIndex++) {
            colorTargets[targetIndex] = colorTargets[0];
        }
        wgpu::FragmentState fragmentState{
            .module = r.fragmentShaderModule,
            .entryPoint = "main",
            .constantCount = r.constants.size(),
            .constants = r.constants.data(),
            .targetCount = r.colorTargetCount,
            .targets = colorTargets.data()
        };
        descriptor.fragment = &fragmentState;
    }
    return device.CreateRenderPipeline(&descriptor);
}

}// namespace filament::backend
