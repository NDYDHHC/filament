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

#include "WebGPUConstants.h"
#include "WebGPUStrings.h"

#include "DriverBase.h"
#include <backend/DriverEnums.h>
#include <backend/Program.h>

#include <utils/Panic.h>
#include <utils/ostream.h>

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace filament::backend {

namespace {

[[nodiscard]] std::string replaceSpecConstants(std::string_view shaderLabel,
        std::string_view shaderSource,
        std::unordered_map<uint32_t, std::variant<int32_t, float, bool>> const&
                specConstants) {
    assert_invariant(!specConstants.empty());
    constexpr std::string_view specConstantPrefix = "FILAMENT_SPEC_CONST_";
    constexpr size_t specConstantPrefixSize = specConstantPrefix.size();
    const char* const sourceData = shaderSource.data();
    std::stringstream processedShaderSourceStr{};
    size_t pos = 0;
    while (pos < shaderSource.size()) {
        const size_t posOfNextSpecConstant = shaderSource.find(specConstantPrefix, pos);
        if (posOfNextSpecConstant == std::string::npos) {
            // no more spec constants, so just stream the rest of the source code string
            processedShaderSourceStr
                    << std::string_view(sourceData + pos, shaderSource.size() - pos);
            break;
        }
        const size_t posOfId = posOfNextSpecConstant + specConstantPrefixSize;
        const size_t posAfterId = shaderSource.find('_', posOfId);
        FILAMENT_CHECK_POSTCONDITION(posAfterId != std::string::npos)
                << "malformed " << shaderLabel << ". Found spec constant prefix '"
                << specConstantPrefix << "' without an id or '_' after it.";
        const std::string_view idStr =
                std::string_view(sourceData + posOfId, posAfterId - posOfId);
        const size_t posEndOfStatement = shaderSource.find(';', posAfterId);
        FILAMENT_CHECK_POSTCONDITION(posEndOfStatement != std::string::npos)
                << "malformed " << shaderLabel << ". Found spec constant assignment with id "
                << idStr << " without a terminating ';' character?";
        const std::string_view statementSegment =
                std::string_view(sourceData + posAfterId, posEndOfStatement - posAfterId);
        size_t posOfEqual = statementSegment.find('=');
        if (posOfEqual == std::string::npos) {
            // not an assignment statement, so stream to the end of the statement and continue...
            processedShaderSourceStr
                    << std::string_view(sourceData + pos, posEndOfStatement + 1 - pos);
            pos = posEndOfStatement + 1;
            continue;
        }
        posOfEqual += posAfterId; // position in original source, not just the segment
        int id = 0;
        try {
            id = std::stoi(idStr.data());
        } catch (const std::invalid_argument& e) {
            PANIC_POSTCONDITION("Invalid spec constant id '%s' in %s (not a valid integer?): %s",
                    idStr.data(), shaderLabel.data(), e.what());
        } catch (const std::out_of_range& e) {
            PANIC_POSTCONDITION(
                    "Invalid spec constant id '%s' in %s (not an integer? out of range?): %s",
                    idStr.data(), shaderLabel.data(), e.what());
        }
        const auto newValueItr = specConstants.find(static_cast<uint32_t>(id));
        if (newValueItr == specConstants.end()) {
            // not going to override the constant (stream to the end of the statement)...
            processedShaderSourceStr
                    << std::string_view(sourceData + pos, posEndOfStatement + 1 - pos);
            pos = posEndOfStatement + 1;
            continue;
        }
        // need to override the constant...
        const std::variant<int32_t, float, bool> newValue = newValueItr->second;
        // stream up to the equal sign...
        processedShaderSourceStr << std::string_view(sourceData + pos, posOfEqual + 1 - pos);
        // stream the new value...
        if (auto* v = std::get_if<int32_t>(&newValue)) {
            processedShaderSourceStr << " " << *v << "i";
        } else if (auto* f = std::get_if<float>(&newValue)) {
            processedShaderSourceStr << " " << *f << "f";
        } else if (auto* b = std::get_if<bool>(&newValue)) {
            processedShaderSourceStr << " " << ((*b) ? "true" : "false");
        }
        processedShaderSourceStr << ";";
        // and skip to after the end of the statement in the original source...
        pos = posEndOfStatement + 1;
    }
    return processedShaderSourceStr.str();
}

[[nodiscard]] wgpu::ShaderModule createShaderModule(wgpu::Device& device, Program& program,
        ShaderStage stage,
        std::unordered_map<uint32_t, std::variant<int32_t, float, bool>> const& specConstants) {
    const char* const programName = program.getName().c_str_safe();
    std::array<utils::FixedCapacityVector<uint8_t>, Program::SHADER_TYPE_COUNT> const&
            shaderSource = program.getShadersSource();
    utils::FixedCapacityVector<uint8_t> const& sourceBytes =
            shaderSource[static_cast<size_t>(stage)];
    if (sourceBytes.empty()) {
        return nullptr;// nothing to compile, the shader was not provided
    }

    bool testing = false;
    int32_t oldNumber = 0;
    int32_t newNumber = 0;
    const std::string patternStr = std::string("FILAMENT_SPEC_CONST_0_") + R"(\w*\s*=\s*([^;]+);)";
    const std::regex regexPattern(patternStr);
    const std::string originalSource = reinterpret_cast<const char*>(sourceBytes.data());
    if (originalSource.find("FILAMENT_SPEC_CONST_0_") != std::string::npos &&
            specConstants.find(0) != specConstants.end()) {
        FWGPU_LOGD << "spec constant 0 is in the shader AND the constants to override"
                   << utils::io::endl;
        std::smatch match;
        if (std::regex_search(originalSource, match, regexPattern)) {
            const std::string oldValue = match[1].str();
            oldNumber = static_cast<int32_t>(std::atoi(oldValue.data()));
            const std::variant<int32_t, float, bool> v = specConstants.at(0);
            newNumber = std::get<int32_t>(v);
            if (oldNumber != newNumber) {
                FWGPU_LOGD << "shader has " << oldNumber << " value to override is " << newNumber
                           << utils::io::endl;
                testing = true;
            }
        }
        if (!testing) {
            FWGPU_LOGD << "shader has the same value as the one to be replaced. not checking it."
                       << utils::io::endl;
        }
    }

    std::stringstream labelStream;
    labelStream << programName << " " << filamentShaderStageToString(stage) << " shader";
    auto label = labelStream.str();
    const std::string processedShaderSource =
            specConstants.empty()
                    ? reinterpret_cast<const char*>(sourceBytes.data())
                    : replaceSpecConstants(label, reinterpret_cast<const char*>(sourceBytes.data()),
                              specConstants);
    wgpu::ShaderModuleWGSLDescriptor wgslDescriptor{};
    wgslDescriptor.code = wgpu::StringView(processedShaderSource);
    wgpu::ShaderModuleDescriptor descriptor{
        .nextInChain = &wgslDescriptor,
        .label = label.data()
    };

    if (testing) {
        const std::string originalSourceAgain = reinterpret_cast<const char*>(sourceBytes.data());
        std::smatch match;
        int checked = 0;
        if (std::regex_search(originalSourceAgain, match, regexPattern)) {
            const std::string oldValue = match[1].str();
            const int32_t oldNumberAgain = static_cast<int32_t>(std::atoi(oldValue.data()));
            assert_invariant(
                    oldNumber == oldNumberAgain &&
                    "the original source code value should not have changed! (constant 0)");
            checked++;
        }
        if (std::regex_search(processedShaderSource, match, regexPattern)) {
            const std::string newValue = match[1].str();
            const int32_t newNumberAgain = static_cast<int32_t>(std::atoi(newValue.data()));
            assert_invariant(newNumber == newNumberAgain &&
                             "the processed source code value should have changed! (constant 0)");
            checked++;
        }
        assert_invariant(
                checked == 2 &&
                "did not find old and new values in the old and processed shader source code?");
        FWGPU_LOGD << "The original source value was unchanged as expected (" << oldNumber
                   << ") and the processed source value changed as expected (" << newNumber << ")"
                   << utils::io::endl;
    }

    wgpu::ShaderModule module = device.CreateShaderModule(&descriptor);
    FILAMENT_CHECK_POSTCONDITION(module != nullptr) << "Failed to create " << descriptor.label;

    wgpu::Instance instance = device.GetAdapter().GetInstance();
    instance.WaitAny(
            module.GetCompilationInfo(wgpu::CallbackMode::WaitAnyOnly,
                    [&descriptor](auto const& status,
                            wgpu::CompilationInfo const* info) {
                        switch (status) {
                            case wgpu::CompilationInfoRequestStatus::CallbackCancelled:
                                FWGPU_LOGW << "Shader compilation info callback cancelled for "
                                           << descriptor.label << "?" << utils::io::endl;
                                return;
                            case wgpu::CompilationInfoRequestStatus::Success:
                                break;
                        }
                        if (info != nullptr) {
                            std::stringstream errorStream;
                            int errorCount = 0;
                            for (size_t msgIndex = 0; msgIndex < info->messageCount; msgIndex++) {
                                wgpu::CompilationMessage const& message = info->messages[msgIndex];
                                switch (message.type) {
                                    case wgpu::CompilationMessageType::Info:
                                        FWGPU_LOGI << descriptor.label << ": " << message.message
                                                   << " line#:" << message.lineNum
                                                   << " linePos:" << message.linePos
                                                   << " offset:" << message.offset
                                                   << " length:" << message.length
                                                   << utils::io::endl;
                                        break;
                                    case wgpu::CompilationMessageType::Warning:
                                        FWGPU_LOGW
                                                << "Warning compiling " << descriptor.label << ": "
                                                << message.message << " line#:" << message.lineNum
                                                << " linePos:" << message.linePos
                                                << " offset:" << message.offset
                                                << " length:" << message.length << utils::io::endl;
                                        break;
                                    case wgpu::CompilationMessageType::Error:
                                        errorCount++;
                                        errorStream << "Error " << errorCount << " : "
                                                    << std::string_view(message.message)
                                                    << " line#:" << message.lineNum
                                                    << " linePos:" << message.linePos
                                                    << " offset:" << message.offset
                                                    << " length:" << message.length << "\n";
                                        break;
                                }
                            }
                            FILAMENT_CHECK_POSTCONDITION(errorCount < 1)
                                    << errorCount << " error(s) compiling " << descriptor.label
                                    << ":\n"
                                    << errorStream.str();
                        }
#if FWGPU_ENABLED(FWGPU_DEBUG_VALIDATION)
                        FWGPU_LOGD << descriptor.label << " compiled successfully"
                                   << utils::io::endl;
#endif
                    }),
            SHADER_COMPILATION_TIMEOUT_NANOSECONDS);
    return module;
}

void toMap(utils::FixedCapacityVector<Program::SpecializationConstant> const& specConstants,
        std::unordered_map<uint32_t, std::variant<int32_t, float, bool>>& constantById) {
    constantById.reserve(specConstants.size());
    for (auto const& specConstant: specConstants) {
        constantById.emplace(specConstant.id, specConstant.value);
    }
}

}// namespace

WGPUProgram::WGPUProgram(wgpu::Device& device, Program& program)
    : HwProgram(program.getName()) {
    std::unordered_map<uint32_t, std::variant<int32_t, float, bool>> specConstants;
    toMap(program.getSpecializationConstants(), specConstants);
    if (specConstants.find(0) != specConstants.end()) {
        specConstants[0] = 42;
    }
    vertexShaderModule = createShaderModule(device, program, ShaderStage::VERTEX, specConstants);
    fragmentShaderModule =
            createShaderModule(device, program, ShaderStage::FRAGMENT, specConstants);
    computeShaderModule = createShaderModule(device, program, ShaderStage::COMPUTE, specConstants);
}

}// namespace filament::backend
