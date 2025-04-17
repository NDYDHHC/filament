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

#include "WebGPUHandles.h"

#include <backend/DriverEnums.h>
#include <backend/Handle.h>
#include <backend/Program.h>

#include <utils/BitmaskEnum.h>
#include <utils/compiler.h>
#include <utils/FixedCapacityVector.h>

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>
#include <sstream>
#include <string> // for std::to_string

namespace {

constexpr wgpu::BufferUsage getBufferObjectUsage(
        filament::backend::BufferObjectBinding bindingType) noexcept {
    switch (bindingType) {
        case filament::backend::BufferObjectBinding::VERTEX:
            return wgpu::BufferUsage::Vertex;
        case filament::backend::BufferObjectBinding::UNIFORM:
            return wgpu::BufferUsage::Uniform;
        case filament::backend::BufferObjectBinding::SHADER_STORAGE:
            return wgpu::BufferUsage::Storage;
    }
}

wgpu::Buffer createIndexBuffer(wgpu::Device const& device, uint8_t elementSize, uint32_t indexCount) {
    wgpu::BufferDescriptor descriptor{ .label = "index_buffer",
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index,
        .size = elementSize * indexCount,
        .mappedAtCreation = false };
    return device.CreateBuffer(&descriptor);
}

wgpu::VertexFormat getVertexFormat(filament::backend::ElementType type, bool normalized,
        bool integer) {
    using ElementType = filament::backend::ElementType;
    using VertexFormat = wgpu::VertexFormat;
    if (normalized) {
        switch (type) {
            // Single Component Types
            case ElementType::BYTE:
                return VertexFormat::Snorm8;
            case ElementType::UBYTE:
                return VertexFormat::Unorm8;
            case ElementType::SHORT:
                return VertexFormat::Snorm16;
            case ElementType::USHORT:
                return VertexFormat::Unorm16;
            // Two Component Types
            case ElementType::BYTE2:
                return VertexFormat::Snorm8x2;
            case ElementType::UBYTE2:
                return VertexFormat::Unorm8x2;
            case ElementType::SHORT2:
                return VertexFormat::Snorm16x2;
            case ElementType::USHORT2:
                return VertexFormat::Unorm16x2;
            // Three Component Types
            // There is no vertex format type for 3 byte data in webgpu. Use
            // 4 byte signed normalized type and ignore the last byte.
            // TODO: This is to be verified.
            case ElementType::BYTE3:
                return VertexFormat::Snorm8x4;// NOT MINSPEC
            case ElementType::UBYTE3:
                return VertexFormat::Unorm8x4;// NOT MINSPEC
            case ElementType::SHORT3:
                return VertexFormat::Snorm16x4;// NOT MINSPEC
            case ElementType::USHORT3:
                return VertexFormat::Unorm16x4;// NOT MINSPEC
            // Four Component Types
            case ElementType::BYTE4:
                return VertexFormat::Snorm8x4;
            case ElementType::UBYTE4:
                return VertexFormat::Unorm8x4;
            case ElementType::SHORT4:
                return VertexFormat::Snorm16x4;
            case ElementType::USHORT4:
                return VertexFormat::Unorm8x4;
            default:
                FILAMENT_CHECK_POSTCONDITION(false) << "Normalized format does not exist.";
                return VertexFormat::Float32x3;
        }
    }
    switch (type) {
        // Single Component Types
        // There is no direct alternative for SSCALED in webgpu. Convert them to Float32 directly.
        // This will result in increased memory on the cpu side.
        // TODO: Is Float16 acceptable instead with some potential accuracy errors?
        case ElementType::BYTE:
            return integer ? VertexFormat::Sint8 : VertexFormat::Float32;
        case ElementType::UBYTE:
            return integer ? VertexFormat::Uint8 : VertexFormat::Float32;
        case ElementType::SHORT:
            return integer ? VertexFormat::Sint16 : VertexFormat::Float32;
        case ElementType::USHORT:
            return integer ? VertexFormat::Uint16 : VertexFormat::Float32;
        case ElementType::HALF:
            return VertexFormat::Float16;
        case ElementType::INT:
            return VertexFormat::Sint32;
        case ElementType::UINT:
            return VertexFormat::Uint32;
        case ElementType::FLOAT:
            return VertexFormat::Float32;
        // Two Component Types
        case ElementType::BYTE2:
            return integer ? VertexFormat::Sint8x2 : VertexFormat::Float32x2;
        case ElementType::UBYTE2:
            return integer ? VertexFormat::Uint8x2 : VertexFormat::Float32x2;
        case ElementType::SHORT2:
            return integer ? VertexFormat::Sint16x2 : VertexFormat::Float32x2;
        case ElementType::USHORT2:
            return integer ? VertexFormat::Uint16x2 : VertexFormat::Float32x2;
        case ElementType::HALF2:
            return VertexFormat::Float16x2;
        case ElementType::FLOAT2:
            return VertexFormat::Float32x2;
        // Three Component Types
        case ElementType::BYTE3:
            return VertexFormat::Sint8x4;// NOT MINSPEC
        case ElementType::UBYTE3:
            return VertexFormat::Uint8x4;// NOT MINSPEC
        case ElementType::SHORT3:
            return VertexFormat::Sint16x4;// NOT MINSPEC
        case ElementType::USHORT3:
            return VertexFormat::Uint16x4;// NOT MINSPEC
        case ElementType::HALF3:
            return VertexFormat::Float16x4;// NOT MINSPEC
        case ElementType::FLOAT3:
            return VertexFormat::Float32x3;
        // Four Component Types
        case ElementType::BYTE4:
            return integer ? VertexFormat::Sint8x4 : VertexFormat::Float32x4;
        case ElementType::UBYTE4:
            return integer ? VertexFormat::Uint8x4 : VertexFormat::Float32x4;
        case ElementType::SHORT4:
            return integer ? VertexFormat::Sint16x4 : VertexFormat::Float32x4;
        case ElementType::USHORT4:
            return integer ? VertexFormat::Uint16x4 : VertexFormat::Float32x4;
        case ElementType::HALF4:
            return VertexFormat::Float16x4;
        case ElementType::FLOAT4:
            return VertexFormat::Float32x4;
    }
}

wgpu::ShaderModule createShaderModuleFromWgsl(wgpu::Device& device, const char* programName,
        std::string_view shaderType, utils::FixedCapacityVector<uint8_t> const& wgslSource) {
    wgpu::ShaderModuleWGSLDescriptor wgslDescriptor{};
    wgslDescriptor.code = wgpu::StringView(reinterpret_cast<const char*>(wgslSource.data()));
    std::stringstream labelStream;
    labelStream << programName << "_" << shaderType << "_shader";
    wgpu::ShaderModuleDescriptor descriptor{
        .nextInChain = &wgslDescriptor,
        .label = labelStream.str().data()
    };
    return device.CreateShaderModule(&descriptor);
}

wgpu::ShaderModule createVertexShaderModule(const char* programName, wgpu::Device& device,
        utils::FixedCapacityVector<uint8_t> const& source) {
    if (UTILS_UNLIKELY(source.empty())) {
        return nullptr;// null handle
    }
    return createShaderModuleFromWgsl(device, programName, "vertex", source);
}

wgpu::ShaderModule createFragmentShaderModule(const char* programName, wgpu::Device& device,
        utils::FixedCapacityVector<uint8_t> const& source) {
    if (source.empty()) {
        return nullptr;// null handle
    }
    return createShaderModuleFromWgsl(device, programName, "fragment", source);
}

wgpu::ShaderModule createComputeShaderModule(const char* programName, wgpu::Device& device,
        utils::FixedCapacityVector<uint8_t> const& source) {
    if (source.empty()) {
        return nullptr;// null handle
    }
    return createShaderModuleFromWgsl(device, programName, "compute", source);
}

utils::FixedCapacityVector<wgpu::ConstantEntry> convertConstants(
        utils::FixedCapacityVector<filament::backend::Program::SpecializationConstant> const&
                constantsInfo) {
    utils::FixedCapacityVector<wgpu::ConstantEntry> constants(constantsInfo.size());
    for (size_t i = 0; i < constantsInfo.size(); i++) {
        filament::backend::Program::SpecializationConstant const& specConstant = constantsInfo[i];
        wgpu::ConstantEntry& constantEntry = constants[i];
        constantEntry.key = wgpu::StringView(std::to_string(specConstant.id));
        if (auto* v = std::get_if<int32_t>(&specConstant.value)) {
            constantEntry.value = static_cast<double>(*v);
        } else if (auto* f = std::get_if<float>(&specConstant.value)) {
            constantEntry.value = static_cast<double>(*f);
        } else if (auto* b = std::get_if<bool>(&specConstant.value)) {
            constantEntry.value = *b ? 0.0 : 1.0;
        }
    }
    return constants;
}

} // namespace

namespace filament::backend {

WGPUVertexBufferInfo::WGPUVertexBufferInfo(uint8_t bufferCount, uint8_t attributeCount,
        AttributeArray const& attributes)
    : HwVertexBufferInfo(bufferCount, attributeCount) {
    if (bufferCount == 0 || attributeCount == 0) {
        HwVertexBufferInfo::bufferCount = 0;
        HwVertexBufferInfo::attributeCount = 0;
        return;
    }
    assert_invariant(attributes.size() >= attributeCount);
    assert_invariant(attributeCount <= MAX_VERTEX_ATTRIBUTE_COUNT);
    assert_invariant(bufferCount <= MAX_VERTEX_BUFFER_COUNT);
    // sort attributes first by buffer index, then by offset and keep original index
    std::array<std::pair<Attribute const*, uint8_t>, MAX_VERTEX_ATTRIBUTE_COUNT>
            attributesWithIndex{};
    for (size_t i = 0; i < attributes.size(); i++) {
        attributesWithIndex[i] = std::pair{ &attributes[i], i };
    }
    std::sort(attributesWithIndex.begin(), attributesWithIndex.end(),
            [](auto const& first, auto const& second) {
                auto firstAttr = first.first;
                auto secondAttr = second.first;
                if (firstAttr->buffer < secondAttr->buffer) {
                    return true;// buffer index in increasing order
                } else if (firstAttr->buffer > secondAttr->buffer) {
                    return false;
                }
                // same buffer index...
                if (firstAttr->offset < secondAttr->offset) {
                    return true;// offsets in increasing order by buffer
                } else if (firstAttr->offset > secondAttr->offset) {
                    return false;
                } else if (firstAttr->buffer == Attribute::BUFFER_UNUSED) {
                    return true;// don't care
                } else {
                    assert_invariant(false);// should not be possible to have multiple attributes
                                            // with the same value buffer index and offset
                    return true;
                }
            });
    uint8_t bufferIndex = 0;
    // make sure the first sorted attribute starts at buffer 0
    assert_invariant(attributesWithIndex[0].first->buffer == bufferIndex);
    for (size_t sortedIndex = 0; sortedIndex < attributes.size(); sortedIndex++) {
        const auto [attribute, attrIndex] = attributesWithIndex[sortedIndex];
        if (attribute->buffer == Attribute::BUFFER_UNUSED) {
            HwVertexBufferInfo::attributeCount = sortedIndex;
            break;
        }
        wgpu::VertexAttribute& vAttribute = vertexAttributes[sortedIndex];
        if (attribute->buffer > bufferIndex) {
            bufferIndex++;
            // make sure each buffer index increases by 1
            assert_invariant(bufferIndex == attribute->buffer);
        }
        vAttribute.format =
                getVertexFormat(attribute->type, attribute->flags & Attribute::FLAG_NORMALIZED,
                        attribute->flags & Attribute::FLAG_INTEGER_TARGET);
        vAttribute.offset = attribute->offset;
        vAttribute.shaderLocation = attrIndex;
        wgpu::VertexBufferLayout& vbLayout = vertexBufferLayouts[attribute->buffer];
        if (vbLayout.attributes == nullptr) {
            vbLayout.attributes = &vAttribute;
            vbLayout.attributeCount = 0;
            vbLayout.arrayStride = 0;
            vbLayout.stepMode = wgpu::VertexStepMode::Vertex;
        }
        vbLayout.attributeCount++;
        vbLayout.arrayStride += attribute->stride;
    }
    HwVertexBufferInfo::bufferCount = bufferIndex + 1;
}

WGPUProgram::WGPUProgram(wgpu::Device& device, Program& program)
    : HwProgram(program.getName()),
      vertexShaderModule(createVertexShaderModule(name.c_str_safe(), device,
              program.getShadersSource()[static_cast<size_t>(ShaderStage::VERTEX)])),
      fragmentShaderModule(createFragmentShaderModule(name.c_str_safe(), device,
              program.getShadersSource()[static_cast<size_t>(ShaderStage::FRAGMENT)])),
      computeShaderModule(createComputeShaderModule(name.c_str_safe(), device,
              program.getShadersSource()[static_cast<size_t>(ShaderStage::COMPUTE)])),
      constants(convertConstants(program.getSpecializationConstants())) {}

WGPUIndexBuffer::WGPUIndexBuffer(wgpu::Device const& device, uint8_t elementSize,
        uint32_t indexCount)
    : buffer(createIndexBuffer(device, elementSize, indexCount)),
      indexFormat(elementSize == 2 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32) {}

WGPUVertexBuffer::WGPUVertexBuffer(wgpu::Device const& device, uint32_t vextexCount,
        uint32_t bufferCount, Handle<HwVertexBufferInfo> vbih)
    : HwVertexBuffer(vextexCount),
          vbih(vbih),
          buffers(bufferCount) {
    wgpu::BufferDescriptor descriptor {
            .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex,
            .size = vextexCount * bufferCount,
            .mappedAtCreation = false };

    for (uint32_t i = 0; i < bufferCount; ++i) {
        descriptor.label = ("vertex_buffer_" + std::to_string(i)).c_str();
        buffers[i] = device.CreateBuffer(&descriptor);
    }
}

WGPUBufferObject::WGPUBufferObject(wgpu::Device const& device, BufferObjectBinding bindingType,
        uint32_t byteCount)
    : HwBufferObject(byteCount),
      bufferObjectBinding(bindingType) {
    wgpu::BufferDescriptor descriptor{ .usage = getBufferObjectUsage(bindingType),
        .size = byteCount,
        .mappedAtCreation = false };
    buffer = device.CreateBuffer(&descriptor);
}

wgpu::ShaderStage WebGPUDescriptorSetLayout::filamentStageToWGPUStage(ShaderStageFlags fFlags) {
    wgpu::ShaderStage retStages = wgpu::ShaderStage::None;
    if (any(ShaderStageFlags::VERTEX & fFlags)) {
        retStages |= wgpu::ShaderStage::Vertex;
    }
    if (any(ShaderStageFlags::FRAGMENT & fFlags)) {
        retStages |= wgpu::ShaderStage::Fragment;
    }
    if (any(ShaderStageFlags::COMPUTE & fFlags)) {
        retStages |= wgpu::ShaderStage::Compute;
    }
    return retStages;
}

WebGPUDescriptorSetLayout::WebGPUDescriptorSetLayout(DescriptorSetLayout const& layout,
        wgpu::Device const& device) {
    assert_invariant(device);

//    // TODO: layoutDescriptor has a "Label". Ideally we can get info on what this layout is for
//    // debugging. For now, hack an incrementing value.
//    static int layoutNum = 0;

    uint samplerCount =
            std::count_if(layout.bindings.begin(), layout.bindings.end(), [](auto& fEntry) {
                return fEntry.type == DescriptorType::SAMPLER ||
                       fEntry.type == DescriptorType::SAMPLER_EXTERNAL;
            });


    std::vector<wgpu::BindGroupLayoutEntry> wEntries;
    wEntries.reserve(layout.bindings.size() + samplerCount);

    for (auto fEntry: layout.bindings) {
        auto& wEntry = wEntries.emplace_back();
        wEntry.visibility = filamentStageToWGPUStage(fEntry.stageFlags);
        wEntry.binding = fEntry.binding * 2;

        switch (fEntry.type) {
            // TODO Metal treats these the same. Is this fine?
            case DescriptorType::SAMPLER_EXTERNAL:
            case DescriptorType::SAMPLER: {
                // Sampler binding is 2n+1 due to split.
                auto& samplerEntry = wEntries.emplace_back();
                samplerEntry.binding = fEntry.binding * 2 + 1;
                samplerEntry.visibility = wEntry.visibility;
                // We are simply hoping that undefined and defaults suffices here.
                samplerEntry.sampler.type = wgpu::SamplerBindingType::Undefined;
                wEntry.texture.sampleType = wgpu::TextureSampleType::Undefined;
                break;
            }
            case DescriptorType::UNIFORM_BUFFER: {
                wEntry.buffer.hasDynamicOffset =
                        any(fEntry.flags & DescriptorFlags::DYNAMIC_OFFSET);
                wEntry.buffer.type = wgpu::BufferBindingType::Uniform;
                // TODO: Ideally we fill minBindingSize
                break;
            }

            case DescriptorType::INPUT_ATTACHMENT: {
                // TODO: support INPUT_ATTACHMENT. Metal does not currently.
                PANIC_POSTCONDITION("Input Attachment is not supported");
                break;
            }

            case DescriptorType::SHADER_STORAGE_BUFFER: {
                // TODO: Vulkan does not support this, can we?
                PANIC_POSTCONDITION("Shader storage is not supported");
                break;
            }
        }

        // Currently flags are only used to specify dynamic offset.

        // UNUSED
        // fEntry.count
    }

//    wgpu::BindGroupLayoutDescriptor layoutDescriptor{
//        // TODO: layoutDescriptor has a "Label". Ideally we can get info on what this layout is for
//        // debugging. For now, hack an incrementing value.
//        .label{ "layout_" + std::to_string(++layoutNum) },
//        .entryCount = wEntries.size(),
//        .entries = wEntries.data()
//    };
//    // TODO Do we need to defer this until we have more info on textures and samplers??
//    mLayout = device.CreateBindGroupLayout(&layoutDescriptor);
}
WebGPUDescriptorSetLayout::~WebGPUDescriptorSetLayout() {}

void WGPURenderPrimitive::setBuffers(WGPUVertexBufferInfo const* const vbi, WGPUVertexBuffer*,
        WGPUIndexBuffer*) {}

}// namespace filament::backend
