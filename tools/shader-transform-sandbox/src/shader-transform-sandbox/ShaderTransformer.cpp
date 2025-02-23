/*
 * Copyright (C) 2017 The Android Open Source Project
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

#include "ShaderTransformer.h"

#include <iostream>
#include <string>
#include <vector>

#include <intermediate.h>
#include <resource_limits_c.h>
#include <GlslangToSpv.h>
#include "spirv_glsl.hpp"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

namespace {

//class SeparateCombinedImageSamplersPass : public spvtools::opt::Pass {
//public:
//    SeparateCombinedImageSamplersPass();
//
//    const char* name() const override { return "separate-combined-image-samplers"; }
//
//    spvtools::opt::Pass::Status Process() override {
//        return spvtools::opt::Pass::Status::SuccessWithoutChange;
//    }
//};

constexpr TBuiltInResource DefaultTBuiltInResource = {
        /* .MaxLights = */ 32,
        /* .MaxClipPlanes = */ 6,
        /* .MaxTextureUnits = */ 32,
        /* .MaxTextureCoords = */ 32,
        /* .MaxVertexAttribs = */ 64,
        /* .MaxVertexUniformComponents = */ 4096,
        /* .MaxVaryingFloats = */ 64,
        /* .MaxVertexTextureImageUnits = */ 32,
        /* .MaxCombinedTextureImageUnits = */ 80,
        /* .MaxTextureImageUnits = */ 32,
        /* .MaxFragmentUniformComponents = */ 4096,
        /* .MaxDrawBuffers = */ 32,
        /* .MaxVertexUniformVectors = */ 128,
        /* .MaxVaryingVectors = */ 8,
        /* .MaxFragmentUniformVectors = */ 16,
        /* .MaxVertexOutputVectors = */ 16,
        /* .MaxFragmentInputVectors = */ 15,
        /* .MinProgramTexelOffset = */ -8,
        /* .MaxProgramTexelOffset = */ 7,
        /* .MaxClipDistances = */ 8,
        /* .MaxComputeWorkGroupCountX = */ 65535,
        /* .MaxComputeWorkGroupCountY = */ 65535,
        /* .MaxComputeWorkGroupCountZ = */ 65535,
        /* .MaxComputeWorkGroupSizeX = */ 1024,
        /* .MaxComputeWorkGroupSizeY = */ 1024,
        /* .MaxComputeWorkGroupSizeZ = */ 64,
        /* .MaxComputeUniformComponents = */ 1024,
        /* .MaxComputeTextureImageUnits = */ 16,
        /* .MaxComputeImageUniforms = */ 8,
        /* .MaxComputeAtomicCounters = */ 8,
        /* .MaxComputeAtomicCounterBuffers = */ 1,
        /* .MaxVaryingComponents = */ 60,
        /* .MaxVertexOutputComponents = */ 64,
        /* .MaxGeometryInputComponents = */ 64,
        /* .MaxGeometryOutputComponents = */ 128,
        /* .MaxFragmentInputComponents = */ 128,
        /* .MaxImageUnits = */ 8,
        /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
        /* .MaxCombinedShaderOutputResources = */ 8,
        /* .MaxImageSamples = */ 0,
        /* .MaxVertexImageUniforms = */ 0,
        /* .MaxTessControlImageUniforms = */ 0,
        /* .MaxTessEvaluationImageUniforms = */ 0,
        /* .MaxGeometryImageUniforms = */ 0,
        /* .MaxFragmentImageUniforms = */ 8,
        /* .MaxCombinedImageUniforms = */ 8,
        /* .MaxGeometryTextureImageUnits = */ 16,
        /* .MaxGeometryOutputVertices = */ 256,
        /* .MaxGeometryTotalOutputComponents = */ 1024,
        /* .MaxGeometryUniformComponents = */ 1024,
        /* .MaxGeometryVaryingComponents = */ 64,
        /* .MaxTessControlInputComponents = */ 128,
        /* .MaxTessControlOutputComponents = */ 128,
        /* .MaxTessControlTextureImageUnits = */ 16,
        /* .MaxTessControlUniformComponents = */ 1024,
        /* .MaxTessControlTotalOutputComponents = */ 4096,
        /* .MaxTessEvaluationInputComponents = */ 128,
        /* .MaxTessEvaluationOutputComponents = */ 128,
        /* .MaxTessEvaluationTextureImageUnits = */ 16,
        /* .MaxTessEvaluationUniformComponents = */ 1024,
        /* .MaxTessPatchComponents = */ 120,
        /* .MaxPatchVertices = */ 32,
        /* .MaxTessGenLevel = */ 64,
        /* .MaxViewports = */ 16,
        /* .MaxVertexAtomicCounters = */ 0,
        /* .MaxTessControlAtomicCounters = */ 0,
        /* .MaxTessEvaluationAtomicCounters = */ 0,
        /* .MaxGeometryAtomicCounters = */ 0,
        /* .MaxFragmentAtomicCounters = */ 8,
        /* .MaxCombinedAtomicCounters = */ 8,
        /* .MaxAtomicCounterBindings = */ 1,
        /* .MaxVertexAtomicCounterBuffers = */ 0,
        /* .MaxTessControlAtomicCounterBuffers = */ 0,
        /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
        /* .MaxGeometryAtomicCounterBuffers = */ 0,
        /* .MaxFragmentAtomicCounterBuffers = */ 1,
        /* .MaxCombinedAtomicCounterBuffers = */ 1,
        /* .MaxAtomicCounterBufferSize = */ 16384,
        /* .MaxTransformFeedbackBuffers = */ 4,
        /* .MaxTransformFeedbackInterleavedComponents = */ 64,
        /* .MaxCullDistances = */ 8,
        /* .MaxCombinedClipAndCullDistances = */ 8,
        /* .MaxSamples = */ 4,
        /* .maxMeshOutputVerticesNV = */ 256,
        /* .maxMeshOutputPrimitivesNV = */ 512,
        /* .maxMeshWorkGroupSizeX_NV = */ 32,
        /* .maxMeshWorkGroupSizeY_NV = */ 1,
        /* .maxMeshWorkGroupSizeZ_NV = */ 1,
        /* .maxTaskWorkGroupSizeX_NV = */ 32,
        /* .maxTaskWorkGroupSizeY_NV = */ 1,
        /* .maxTaskWorkGroupSizeZ_NV = */ 1,
        /* .maxMeshViewCountNV = */ 4,
        /* .max_mesh_output_vertices_ext = */ 256,
        /* .max_mesh_output_primitives_ext = */ 256,
        /* .max_mesh_work_group_size_x_ext = */ 128,
        /* .max_mesh_work_group_size_y_ext = */ 128,
        /* .max_mesh_work_group_size_z_ext = */ 128,
        /* .max_task_work_group_size_x_ext = */ 128,
        /* .max_task_work_group_size_y_ext = */ 128,
        /* .max_task_work_group_size_z_ext = */ 128,
        /* .max_mesh_view_count_ext = */ 4,
        /* .maxDualSourceDrawBuffersEXT = */ 1,

        /* .limits = */ {
                /* .nonInductiveForLoops = */ 1,
                /* .whileLoops = */ 1,
                /* .doWhileLoops = */ 1,
                /* .generalUniformIndexing = */ 1,
                /* .generalAttributeMatrixVectorIndexing = */ 1,
                /* .generalVaryingIndexing = */ 1,
                /* .generalSamplerIndexing = */ 1,
                /* .generalVariableIndexing = */ 1,
                /* .generalConstantMatrixVectorIndexing = */ 1,
        }
};

std::string readText(std::basic_istream<char>& in) {
    std::string text {};
    std::string line {};
    while (std::getline(in, line)) {
        text.append(line);
        text.append("\n");
    }
    return text;
}

std::vector<uint32_t> glslToSpirv(std::string const& glsl) {
    const char *glslStr{glsl.c_str()};
    int length{static_cast<int>(glsl.size())};

    glslang::InitializeProcess();
    glslang::TShader shader{EShLangFragment};
    shader.setDebugInfo(true);
    shader.setStringsWithLengths(&glslStr, &length, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, EShLangFragment, glslang::EShClientOpenGL,
                       glslang::EShTargetClientVersion::EShTargetOpenGL_450);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_3);
    auto includer{glslang::TShader::ForbidIncluder{}};
    EShMessages message {static_cast<EShMessages>(EShMessages::EShMsgVulkanRules | EShMessages::EShMsgSpvRules)};
    if (!shader.parse(&DefaultTBuiltInResource,
                      glslang::EShTargetClientVersion::EShTargetOpenGL_450,
                      false,
                      EShMessages::EShMsgDefault,
                      includer)) {
        std::cerr << "Failed to parse GLSL!" << std::endl;
        return {};
    }
    glslang::TProgram program {};
	program.addShader(&shader);
    if (!program.link(message)) {
        std::cerr << "WARNING linking shader: " << program.getInfoDebugLog() << std::endl;
        std::cerr << "ERROR linking shader: " << program.getInfoLog() << std::endl;
        return {};
    }
    glslang::SpvOptions options{
        .generateDebugInfo = true,
        .stripDebugInfo = false,
        .disableOptimizer = true
    };
    std::vector<uint32_t> spirv {};
    glslang::GlslangToSpv(*program.getIntermediate(EShLangFragment), spirv, &options);
    glslang::FinalizeProcess();
    return spirv;
}

std::vector<uint32_t> modifySpirv(std::vector<uint32_t> const &spirv) {
    spvtools::Optimizer optimizer{SPV_ENV_UNIVERSAL_1_3};
    std::vector<uint32_t> transformedSpirv{};
    if (!optimizer.Run(spirv.data(), spirv.size(), &transformedSpirv)) {
        std::cerr << "Failed to transform SPIR-V!" << std::endl;
        return {};
    }
    return transformedSpirv;
}

std::string spirvToGlsl(std::vector<uint32_t> const & spirv) {
    spirv_cross::CompilerGLSL compiler{spirv};
    spirv_cross::CompilerGLSL::Options options {
        .version = glslang::EShTargetClientVersion::EShTargetOpenGL_450,
        .es = false
    };
    compiler.set_common_options(options);
	return compiler.compile();
}

}  // namespace

namespace shadertransformsandbox {

bool ShaderTransformer::transform(std::basic_istream<char> &in, std::basic_ostream<char> &out) {
    std::string originalGlsl{readText(in)};
    std::vector<uint32_t> spirv {glslToSpirv(originalGlsl)};
    if (spirv.empty()) {
        return false;
    }
    std::cerr << "INFO: initial SPIR-V. length(" << spirv.size() << ")" << std::endl;
    std::vector<uint32_t> modifiedSpirv {modifySpirv(spirv)};
    if (spirv.empty()) {
        return false;
    }
    std::cerr << "INFO: transformed SPIR-V. length(" << modifiedSpirv.size() << ")" << std::endl;
    std::string outputGlsl{spirvToGlsl(modifiedSpirv)};
    out << outputGlsl;
    return true;
}

} // shadertransformsandbox
