/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "OpenGLBlobCache.h"

#include "OpenGLContext.h"

#include <backend/Platform.h>
#include <backend/Program.h>

#include <private/utils/Tracing.h>

#include <utils/Logger.h>

namespace filament::backend {

OpenGLBlobCache::OpenGLBlobCache(OpenGLContext& gl) noexcept
    : mCachingSupported(gl.gets.num_program_binary_formats >= 1) {
}

OpenGLBlobCache::Blob::Ptr OpenGLBlobCache::retrieve(BlobCacheKey* outKey, size_t* outSize,
        Platform& platform, Program const& program) const noexcept {
    FILAMENT_TRACING_CALL(FILAMENT_TRACING_CATEGORY_FILAMENT);
    if (!mCachingSupported || !platform.hasRetrieveBlobFunc()) {
        // the key is never updated in that case
        return {nullptr, nullptr};
    }

    BlobCacheKey key{ program.getCacheId(), program.getSpecializationConstants() };

    // always attempt with 64 KiB
    constexpr size_t DEFAULT_BLOB_SIZE = 65536;
    Blob::Ptr blob = Blob::create(DEFAULT_BLOB_SIZE);

    size_t const blobSize = platform.retrieveBlob(
            key.data(), key.size(), blob.get(), DEFAULT_BLOB_SIZE);

    if (blobSize > 0) {
        if (blobSize > DEFAULT_BLOB_SIZE) {
            // our buffer was too small, retry with the correct size
            blob = Blob::create(blobSize);
            platform.retrieveBlob(
                    key.data(), key.size(), blob.get(), blobSize);
        }
    }

    if (UTILS_LIKELY(outKey)) {
        using std::swap;
        swap(*outKey, key);
    }

    if (UTILS_LIKELY(outSize)) {
        *outSize = blobSize;
    }

    return blob;
}

GLuint OpenGLBlobCache::createProgram(BlobCacheKey key, Program const& program,
        Blob const& blob, size_t blobSize) const noexcept {
    FILAMENT_TRACING_CALL(FILAMENT_TRACING_CATEGORY_FILAMENT);

    GLuint programId = 0;

#ifndef FILAMENT_SILENCE_NOT_SUPPORTED_BY_ES2
    GLsizei const programBinarySize = GLsizei(blobSize - sizeof(Blob));

    programId = glCreateProgram();

    { // scope for systrace
        FILAMENT_TRACING_NAME(FILAMENT_TRACING_CATEGORY_FILAMENT, "glProgramBinary");
        glProgramBinary(programId, blob.format, blob.data, programBinarySize);
    }

    // Verify the program retrieved from the blob cache. `glProgramBinary` can succeed but
    // result in an unlinked program, so we must check both `glGetError()` and the
    // `GL_LINK_STATUS`. This can happen if, for instance, the graphics driver has been updated.
    // If loading fails, we return 0 to fall back to a normal compilation and link.
    GLenum glError = glGetError();
    GLint linkStatus = GL_FALSE;
    if (glError == GL_NO_ERROR) {
        glGetProgramiv(programId, GL_LINK_STATUS, &linkStatus);
    }

    if (UTILS_UNLIKELY(glError != GL_NO_ERROR || linkStatus != GL_TRUE)) {
        LOG(WARNING) << "Failed to load program binary, name=" << program.getName().c_str_safe()
                     << ", size=" << blobSize << ", format=" << blob.format
                     << ", glError=" << glError << ", linkStatus=" << linkStatus;
        glDeleteProgram(programId);
        programId = 0;
    }
#endif

    return programId;
}

void OpenGLBlobCache::insert(Platform& platform,
        BlobCacheKey const& key, GLuint program) noexcept {
    FILAMENT_TRACING_CALL(FILAMENT_TRACING_CATEGORY_FILAMENT);
    if (!mCachingSupported || !platform.hasInsertBlobFunc()) {
        // the key is never updated in that case
        return;
    }

#ifndef FILAMENT_SILENCE_NOT_SUPPORTED_BY_ES2
    GLenum format;
    GLint programBinarySize = 0;
    { // scope for systrace
        FILAMENT_TRACING_NAME(FILAMENT_TRACING_CATEGORY_FILAMENT, "glGetProgramiv");
        glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &programBinarySize);
    }
    if (programBinarySize) {
        size_t const size = sizeof(Blob) + programBinarySize;
        std::unique_ptr<Blob, decltype(&::free)> blob{ (Blob*)malloc(size), &::free };
        if (UTILS_LIKELY(blob)) {
            { // scope for systrace
                FILAMENT_TRACING_NAME(FILAMENT_TRACING_CATEGORY_FILAMENT, "glGetProgramBinary");
                glGetProgramBinary(program, programBinarySize,
                        &programBinarySize, &format, blob->data);
            }
            GLenum const error = glGetError();
            if (error == GL_NO_ERROR) {
                blob->format = format;
                platform.insertBlob(key.data(), key.size(), blob.get(), size);
            }
        }
    }
#endif
}

} // namespace filament::backend
