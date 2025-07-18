/*
 * Copyright (C) 2015 The Android Open Source Project
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

#include "details/Texture.h"

#include "details/Engine.h"
#include "details/Stream.h"

#include "private/backend/BackendUtils.h"

#include "FilamentAPI-impl.h"

#include <filament/Texture.h>

#include <backend/DriverEnums.h>
#include <backend/Handle.h>

#include <math/half.h>
#include <math/scalar.h>
#include <math/vec3.h>

#include <utils/Allocator.h>
#include <utils/algorithm.h>
#include <utils/BitmaskEnum.h>
#include <utils/compiler.h>
#include <utils/CString.h>
#include <utils/StaticString.h>
#include <utils/debug.h>
#include <utils/FixedCapacityVector.h>
#include <utils/Panic.h>

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <stddef.h>
#include <stdint.h>

using namespace utils;

namespace filament {

using namespace backend;
using namespace math;

// this is a hack to be able to create a std::function<> with a non-copyable closure
template<class F>
static auto make_copyable_function(F&& f) {
    using dF = std::decay_t<F>;
    auto spf = std::make_shared<dF>(std::forward<F>(f));
    return [spf](auto&& ... args) -> decltype(auto) {
        return (*spf)(decltype(args)(args)...);
    };
}

struct Texture::BuilderDetails {
    intptr_t mImportedId = 0;
    uint32_t mWidth = 1;
    uint32_t mHeight = 1;
    uint32_t mDepth = 1;
    uint8_t mLevels = 1;
    uint8_t mSamples = 1;
    Sampler mTarget = Sampler::SAMPLER_2D;
    InternalFormat mFormat = InternalFormat::RGBA8;
    Usage mUsage = Usage::NONE;
    bool mHasBlitSrc = false;
    bool mTextureIsSwizzled = false;
    bool mExternal = false;
    std::array<Swizzle, 4> mSwizzle = {
           Swizzle::CHANNEL_0, Swizzle::CHANNEL_1,
           Swizzle::CHANNEL_2, Swizzle::CHANNEL_3 };
};

using BuilderType = Texture;
BuilderType::Builder::Builder() noexcept = default;
BuilderType::Builder::~Builder() noexcept = default;
BuilderType::Builder::Builder(Builder const& rhs) noexcept = default;
BuilderType::Builder::Builder(Builder&& rhs) noexcept = default;
BuilderType::Builder& BuilderType::Builder::operator=(Builder const& rhs) noexcept = default;
BuilderType::Builder& BuilderType::Builder::operator=(Builder&& rhs) noexcept = default;


Texture::Builder& Texture::Builder::width(uint32_t const width) noexcept {
    mImpl->mWidth = width;
    return *this;
}

Texture::Builder& Texture::Builder::height(uint32_t const height) noexcept {
    mImpl->mHeight = height;
    return *this;
}

Texture::Builder& Texture::Builder::depth(uint32_t const depth) noexcept {
    mImpl->mDepth = depth;
    return *this;
}

Texture::Builder& Texture::Builder::levels(uint8_t const levels) noexcept {
    mImpl->mLevels = std::max(uint8_t(1), levels);
    return *this;
}

Texture::Builder& Texture::Builder::samples(uint8_t const samples) noexcept {
    mImpl->mSamples = std::max(uint8_t(1), samples);
    return *this;
}

Texture::Builder& Texture::Builder::sampler(Sampler const target) noexcept {
    mImpl->mTarget = target;
    return *this;
}

Texture::Builder& Texture::Builder::format(InternalFormat const format) noexcept {
    mImpl->mFormat = format;
    return *this;
}

Texture::Builder& Texture::Builder::usage(Usage const usage) noexcept {
    mImpl->mUsage = Usage(usage);
    return *this;
}

Texture::Builder& Texture::Builder::import(intptr_t const id) noexcept {
    assert_invariant(id); // imported id can't be zero
    mImpl->mImportedId = id;
    return *this;
}

Texture::Builder& Texture::Builder::external() noexcept {
    mImpl->mExternal = true;
    return *this;
}

Texture::Builder& Texture::Builder::swizzle(Swizzle const r, Swizzle const g, Swizzle const b, Swizzle const a) noexcept {
    mImpl->mTextureIsSwizzled = true;
    mImpl->mSwizzle = { r, g, b, a };
    return *this;
}

Texture::Builder& Texture::Builder::name(const char* name, size_t const len) noexcept {
    return BuilderNameMixin::name(name, len);
}

Texture::Builder& Texture::Builder::name(StaticString const& name) noexcept {
    return BuilderNameMixin::name(name);
}

Texture* Texture::Builder::build(Engine& engine) {
    if (mImpl->mTarget != SamplerType::SAMPLER_EXTERNAL) {
        FILAMENT_CHECK_PRECONDITION(Texture::isTextureFormatSupported(engine, mImpl->mFormat))
                << "Texture format " << uint16_t(mImpl->mFormat)
                << " not supported on this platform, texture name="
                << getNameOrDefault().c_str_safe();

        FILAMENT_CHECK_PRECONDITION(mImpl->mWidth > 0 && mImpl->mHeight > 0)
                << "Texture has invalid dimensions: (" << mImpl->mWidth << ", " << mImpl->mHeight
                << "), texture name=" << getNameOrDefault().c_str_safe();
    }

    if (mImpl->mSamples > 1) {
        FILAMENT_CHECK_PRECONDITION(any(mImpl->mUsage & Texture::Usage::SAMPLEABLE))
                << "Multisample (" << unsigned(mImpl->mSamples)
                << ") texture is not sampleable, texture name=" << getNameOrDefault().c_str_safe();
    }

    const bool isProtectedTexturesSupported =
            downcast(engine).getDriverApi().isProtectedTexturesSupported();
    const bool useProtectedMemory = bool(mImpl->mUsage & TextureUsage::PROTECTED);

    FILAMENT_CHECK_PRECONDITION(
            (isProtectedTexturesSupported && useProtectedMemory) || !useProtectedMemory)
            << "Texture is PROTECTED but protected textures are not supported";

    size_t const maxTextureDimension = getMaxTextureSize(engine, mImpl->mTarget);
    size_t const maxTextureDepth = (mImpl->mTarget == Sampler::SAMPLER_2D_ARRAY ||
                                    mImpl->mTarget == Sampler::SAMPLER_CUBEMAP_ARRAY)
                                       ? getMaxArrayTextureLayers(engine)
                                       : maxTextureDimension;

    FILAMENT_CHECK_PRECONDITION(
            mImpl->mWidth <= maxTextureDimension &&
            mImpl->mHeight <= maxTextureDimension &&
            mImpl->mDepth <= maxTextureDepth) << "Texture dimensions out of range: "
                    << "width= " << mImpl->mWidth << " (>" << maxTextureDimension << ")"
                    <<", height= " << mImpl->mHeight << " (>" << maxTextureDimension << ")"
                    << ", depth= " << mImpl->mDepth << " (>" << maxTextureDepth << ")";

    const auto validateSamplerType = [&engine = downcast(engine)](SamplerType const sampler) -> bool {
        switch (sampler) {
            case SamplerType::SAMPLER_2D:
            case SamplerType::SAMPLER_CUBEMAP:
            case SamplerType::SAMPLER_EXTERNAL:
                return true;
            case SamplerType::SAMPLER_3D:
            case SamplerType::SAMPLER_2D_ARRAY:
                return engine.hasFeatureLevel(FeatureLevel::FEATURE_LEVEL_1);
            case SamplerType::SAMPLER_CUBEMAP_ARRAY:
                return engine.hasFeatureLevel(FeatureLevel::FEATURE_LEVEL_2);
        }
        return false;
    };

    // Validate sampler before any further interaction with it.
    FILAMENT_CHECK_PRECONDITION(validateSamplerType(mImpl->mTarget))
            << "SamplerType " << uint8_t(mImpl->mTarget) << " not support at feature level "
            << uint8_t(engine.getActiveFeatureLevel());

    // SAMPLER_EXTERNAL implies imported.
    if (mImpl->mTarget == SamplerType::SAMPLER_EXTERNAL) {
        mImpl->mExternal = true;
    }

    uint8_t maxLevelCount;
    switch (mImpl->mTarget) {
        case SamplerType::SAMPLER_2D:
        case SamplerType::SAMPLER_2D_ARRAY:
        case SamplerType::SAMPLER_CUBEMAP:
        case SamplerType::SAMPLER_EXTERNAL:
        case SamplerType::SAMPLER_CUBEMAP_ARRAY:
            maxLevelCount = FTexture::maxLevelCount(mImpl->mWidth, mImpl->mHeight);
            break;
        case SamplerType::SAMPLER_3D:
            maxLevelCount = FTexture::maxLevelCount(std::max(
                    { mImpl->mWidth, mImpl->mHeight, mImpl->mDepth }));
            break;
    }
    mImpl->mLevels = std::min(mImpl->mLevels, maxLevelCount);

    if (mImpl->mUsage == TextureUsage::NONE) {
        mImpl->mUsage = TextureUsage::DEFAULT;
        if (mImpl->mLevels > 1 &&
            (mImpl->mWidth > 1 || mImpl->mHeight > 1) &&
            !mImpl->mExternal) {
            const bool formatMipmappable =
                    downcast(engine).getDriverApi().isTextureFormatMipmappable(mImpl->mFormat);
            if (formatMipmappable) {
                // by default mipmappable textures have the BLIT usage bits set
                mImpl->mUsage |= TextureUsage::BLIT_SRC | TextureUsage::BLIT_DST;
            }
        }
    }

    // TODO: remove in a future filament release.
    // Clients might not have known that textures that are read need to have BLIT_SRC as usages. For
    // now, we workaround the issue by making sure any color attachment can be the source of a copy
    // for readPixels().
    mImpl->mHasBlitSrc = any(mImpl->mUsage & TextureUsage::BLIT_SRC);
    if (!mImpl->mHasBlitSrc && any(mImpl->mUsage & TextureUsage::COLOR_ATTACHMENT)) {
        mImpl->mUsage |= TextureUsage::BLIT_SRC;
    }

    const bool sampleable = bool(mImpl->mUsage & TextureUsage::SAMPLEABLE);
    const bool swizzled = mImpl->mTextureIsSwizzled;
    const bool imported = mImpl->mImportedId;

    #if defined(__EMSCRIPTEN__)
    FILAMENT_CHECK_PRECONDITION(!swizzled) << "WebGL does not support texture swizzling.";
    #endif

    FILAMENT_CHECK_PRECONDITION((swizzled && sampleable) || !swizzled)
            << "Swizzled texture must be SAMPLEABLE";

    FILAMENT_CHECK_PRECONDITION((imported && sampleable) || !imported)
            << "Imported texture must be SAMPLEABLE";

    return downcast(engine).createTexture(*this);
}

// ------------------------------------------------------------------------------------------------

FTexture::FTexture(FEngine& engine, const Builder& builder)
    : mHasBlitSrc(false),
      mExternal(false) {
    FEngine::DriverApi& driver = engine.getDriverApi();
    mDriver = &driver; // this is unfortunately needed for getHwHandleForSampling()
    mWidth  = static_cast<uint32_t>(builder->mWidth);
    mHeight = static_cast<uint32_t>(builder->mHeight);
    mDepth  = static_cast<uint32_t>(builder->mDepth);
    mFormat = builder->mFormat;
    mUsage = builder->mUsage;
    mTarget = builder->mTarget;
    mLevelCount = builder->mLevels;
    mSampleCount = builder->mSamples;
    mSwizzle = builder->mSwizzle;
    mTextureIsSwizzled = builder->mTextureIsSwizzled;
    mHasBlitSrc = builder->mHasBlitSrc;
    mExternal = builder->mExternal;
    mTextureType = backend::getTextureType(mFormat);

    bool const isImported = builder->mImportedId != 0;
    if (mExternal && !isImported) {
        // mHandle and mHandleForSampling will be created in setExternalImage()
        // If this Texture is used for sampling before setExternalImage() is called,
        // we'll lazily create a 1x1 placeholder texture.
        return;
    }

    if (UTILS_LIKELY(!isImported)) {
        mHandle = driver.createTexture(
                mTarget, mLevelCount, mFormat, mSampleCount, mWidth, mHeight, mDepth, mUsage);
    } else {
        mHandle = driver.importTexture(builder->mImportedId,
                mTarget, mLevelCount, mFormat, mSampleCount, mWidth, mHeight, mDepth, mUsage);
    }

    if (UTILS_UNLIKELY(builder->mTextureIsSwizzled)) {
        auto const& s = builder->mSwizzle;
        auto swizzleView = driver.createTextureViewSwizzle(mHandle, s[0], s[1], s[2], s[3]);
        driver.destroyTexture(mHandle);
        mHandle = swizzleView;
    }

    mHandleForSampling = mHandle;

    if (auto name = builder.getName(); !name.empty()) {
        driver.setDebugTag(mHandle.getId(), std::move(name));
    } else {
        driver.setDebugTag(mHandle.getId(), CString{"FTexture"});
    }
}

// frees driver resources, object becomes invalid
void FTexture::terminate(FEngine& engine) {
    setHandles({});
}

size_t FTexture::getWidth(size_t const level) const noexcept {
    return valueForLevel(level, mWidth);
}

size_t FTexture::getHeight(size_t const level) const noexcept {
    return valueForLevel(level, mHeight);
}

size_t FTexture::getDepth(size_t const level) const noexcept {
    return valueForLevel(level, mDepth);
}

void FTexture::setImage(FEngine& engine, size_t const level,
        uint32_t const xoffset, uint32_t const yoffset, uint32_t const zoffset,
        uint32_t const width, uint32_t const height, uint32_t const depth,
        PixelBufferDescriptor&& p) const {

    if (UTILS_UNLIKELY(!engine.hasFeatureLevel(FeatureLevel::FEATURE_LEVEL_1))) {
        FILAMENT_CHECK_PRECONDITION(p.stride == 0 || p.stride == width)
                << "PixelBufferDescriptor stride must be 0 (or width) at FEATURE_LEVEL_0";
    }

    // this should have been validated already
    assert_invariant(isTextureFormatSupported(engine, mFormat));

    FILAMENT_CHECK_PRECONDITION(p.type == PixelDataType::COMPRESSED ||
            validatePixelFormatAndType(mFormat, p.format, p.type))
            << "The combination of internal format=" << unsigned(mFormat)
            << " and {format=" << unsigned(p.format) << ", type=" << unsigned(p.type)
            << "} is not supported.";

    FILAMENT_CHECK_PRECONDITION(!mStream) << "setImage() called on a Stream texture.";

    FILAMENT_CHECK_PRECONDITION(level < mLevelCount)
            << "level=" << unsigned(level) << " is >= to levelCount=" << unsigned(mLevelCount)
            << ".";

    FILAMENT_CHECK_PRECONDITION(!mExternal)
            << "External Texture not supported for this operation.";

    FILAMENT_CHECK_PRECONDITION(any(mUsage & Texture::Usage::UPLOADABLE))
            << "Texture is not uploadable.";

    FILAMENT_CHECK_PRECONDITION(mSampleCount <= 1)
            << "Operation not supported with multisample ("
            << unsigned(mSampleCount) << ") texture.";

    FILAMENT_CHECK_PRECONDITION(xoffset + width <= valueForLevel(level, mWidth))
            << "xoffset (" << unsigned(xoffset) << ") + width (" << unsigned(width)
            << ") > texture width (" << valueForLevel(level, mWidth) << ") at level ("
            << unsigned(level) << ")";

    FILAMENT_CHECK_PRECONDITION(yoffset + height <= valueForLevel(level, mHeight))
            << "yoffset (" << unsigned(yoffset) << ") + height (" << unsigned(height)
            << ") > texture height (" << valueForLevel(level, mHeight) << ") at level ("
            << unsigned(level) << ")";

    FILAMENT_CHECK_PRECONDITION(p.buffer) << "Data buffer is nullptr.";

    uint32_t effectiveTextureDepthOrLayers = 0;
    switch (mTarget) {
        case SamplerType::SAMPLER_EXTERNAL:
            // can't happen by construction, fallthrough...
        case SamplerType::SAMPLER_2D:
            assert_invariant(mDepth == 1);
            effectiveTextureDepthOrLayers = 1;
            break;
        case SamplerType::SAMPLER_3D:
            effectiveTextureDepthOrLayers = valueForLevel(level, mDepth);
            break;
        case SamplerType::SAMPLER_2D_ARRAY:
            effectiveTextureDepthOrLayers = mDepth;
            break;
        case SamplerType::SAMPLER_CUBEMAP:
            effectiveTextureDepthOrLayers = 6;
            break;
        case SamplerType::SAMPLER_CUBEMAP_ARRAY:
            effectiveTextureDepthOrLayers = mDepth * 6;
            break;
    }

    FILAMENT_CHECK_PRECONDITION(zoffset + depth <= effectiveTextureDepthOrLayers)
            << "zoffset (" << unsigned(zoffset) << ") + depth (" << unsigned(depth)
            << ") > texture depth (" << effectiveTextureDepthOrLayers << ") at level ("
            << unsigned(level) << ")";

    if (UTILS_UNLIKELY(!width || !height || !depth)) {
        // The operation is a no-op, return immediately. The PixelBufferDescriptor callback
        // should be called automatically when the object is destroyed.
        // The precondition check below assumes width, height, depth non null.
        return;
    }

    if (p.type != PixelDataType::COMPRESSED) {
        using PBD = PixelBufferDescriptor;
        size_t const stride = p.stride ? p.stride : width;
        size_t const bpp = PBD::computeDataSize(p.format, p.type, 1, 1, 1);
        size_t const bpr = PBD::computeDataSize(p.format, p.type, stride, 1, p.alignment);
        size_t const bpl = bpr * height; // TODO: PBD should have a "layer stride"
        // TODO: PBD should have a p.depth (# layers to skip)

        /* Calculates the byte offset of the last pixel in a 3D sub-region. */
        auto const calculateLastPixelOffset = [bpp, bpr, bpl](
                size_t xoff, size_t yoff, size_t zoff,
                size_t width, size_t height, size_t depth) {
            // The 0-indexed coordinates of the last pixel are:
            // x = xoff + width - 1
            // y = yoff + height - 1
            // z = zoff + depth - 1
            // The offset is calculated as: (z * bpl) + (y * bpr) + (x * bpp)
            return ((zoff + depth  - 1) * bpl) +
                   ((yoff + height - 1) * bpr) +
                   ((xoff + width  - 1) * bpp);
        };

        size_t const lastPixelOffset = calculateLastPixelOffset(
                p.left, p.top, 0, width, height, depth);

        // make sure the whole last pixel is in the buffer
        FILAMENT_CHECK_PRECONDITION(lastPixelOffset + bpp <= p.size)
                << "buffer overflow: (size=" << size_t(p.size) << ", stride=" << size_t(p.stride)
                << ", left=" << unsigned(p.left) << ", top=" << unsigned(p.top)
                << ") smaller than specified region "
                   "{{"
                << unsigned(xoffset) << "," << unsigned(yoffset) << "," << unsigned(zoffset) << "},{"
                << unsigned(width) << "," << unsigned(height) << "," << unsigned(depth) << ")}}";
    }

    engine.getDriverApi().update3DImage(mHandle, uint8_t(level), xoffset, yoffset, zoffset, width,
            height, depth, std::move(p));

    // this method shouldn't have been const
    const_cast<FTexture*>(this)->updateLodRange(level);
}

// deprecated
void FTexture::setImage(FEngine& engine, size_t const level,
        PixelBufferDescriptor&& buffer, const FaceOffsets& faceOffsets) const {

    auto validateTarget = [](SamplerType const sampler) -> bool {
        switch (sampler) {
            case SamplerType::SAMPLER_CUBEMAP:
                return true;
            case SamplerType::SAMPLER_2D:
            case SamplerType::SAMPLER_3D:
            case SamplerType::SAMPLER_2D_ARRAY:
            case SamplerType::SAMPLER_CUBEMAP_ARRAY:
            case SamplerType::SAMPLER_EXTERNAL:
                return false;
        }
        return false;
    };

    // this should have been validated already
    assert_invariant(isTextureFormatSupported(engine, mFormat));

    FILAMENT_CHECK_PRECONDITION(buffer.type == PixelDataType::COMPRESSED ||
            validatePixelFormatAndType(mFormat, buffer.format, buffer.type))
            << "The combination of internal format=" << unsigned(mFormat)
            << " and {format=" << unsigned(buffer.format) << ", type=" << unsigned(buffer.type)
            << "} is not supported.";

    FILAMENT_CHECK_PRECONDITION(!mStream) << "setImage() called on a Stream texture.";

    FILAMENT_CHECK_PRECONDITION(level < mLevelCount)
            << "level=" << unsigned(level) << " is >= to levelCount=" << unsigned(mLevelCount)
            << ".";

    FILAMENT_CHECK_PRECONDITION(validateTarget(mTarget))
            << "Texture Sampler type (" << unsigned(mTarget)
            << ") not supported for this operation.";

    FILAMENT_CHECK_PRECONDITION(buffer.buffer) << "Data buffer is nullptr.";

    auto w = std::max(1u, mWidth >> level);
    auto h = std::max(1u, mHeight >> level);
    assert_invariant(w == h);
    const size_t faceSize = PixelBufferDescriptor::computeDataSize(buffer.format, buffer.type,
            buffer.stride ? buffer.stride : w, h, buffer.alignment);

    if (faceOffsets[0] == 0 &&
        faceOffsets[1] == 1 * faceSize &&
        faceOffsets[2] == 2 * faceSize &&
        faceOffsets[3] == 3 * faceSize &&
        faceOffsets[4] == 4 * faceSize &&
        faceOffsets[5] == 5 * faceSize) {
        // in this special case, we can upload all 6 faces in one call
        engine.getDriverApi().update3DImage(mHandle, uint8_t(level),
                0, 0, 0, w, h, 6, std::move(buffer));
    } else {
        UTILS_NOUNROLL
        for (size_t face = 0; face < 6; face++) {
            engine.getDriverApi().update3DImage(mHandle, uint8_t(level), 0, 0, face, w, h, 1, {
                    (char*)buffer.buffer + faceOffsets[face],
                    faceSize, buffer.format, buffer.type, buffer.alignment,
                    buffer.left, buffer.top, buffer.stride });
        }
        engine.getDriverApi().queueCommand(
                make_copyable_function([buffer = std::move(buffer)]() {}));
    }

    // this method shouldn't been const
    const_cast<FTexture*>(this)->updateLodRange(level);
}

void FTexture::setExternalImage(FEngine& engine, ExternalImageHandleRef image) noexcept {
    FILAMENT_CHECK_PRECONDITION(mExternal) << "The texture must be external.";

    // The call to setupExternalImage2 is synchronous, and allows the driver to take ownership of the
    // external image on this thread, if necessary.
    auto& api = engine.getDriverApi();
    api.setupExternalImage2(image);

    auto texture = api.createTextureExternalImage2(mTarget, mFormat, mWidth, mHeight, mUsage, image);

    if (mTextureIsSwizzled) {
        auto const& s = mSwizzle;
        auto swizzleView = api.createTextureViewSwizzle(texture, s[0], s[1], s[2], s[3]);
        api.destroyTexture(texture);
        texture = swizzleView;
    }

    setHandles(texture);
}

void FTexture::setExternalImage(FEngine& engine, void* image) noexcept {
    FILAMENT_CHECK_PRECONDITION(mExternal) << "The texture must be external.";

    // The call to setupExternalImage is synchronous, and allows the driver to take ownership of the
    // external image on this thread, if necessary.
    auto& api = engine.getDriverApi();
    api.setupExternalImage(image);

    auto texture = api.createTextureExternalImage(mTarget, mFormat, mWidth, mHeight, mUsage, image);

    if (mTextureIsSwizzled) {
        auto const& s = mSwizzle;
        auto swizzleView = api.createTextureViewSwizzle(texture, s[0], s[1], s[2], s[3]);
        api.destroyTexture(texture);
        texture = swizzleView;
    }

    setHandles(texture);
}

void FTexture::setExternalImage(FEngine& engine, void* image, size_t const plane) noexcept {
    FILAMENT_CHECK_PRECONDITION(mExternal) << "The texture must be external.";

    // The call to setupExternalImage is synchronous, and allows the driver to take ownership of
    // the external image on this thread, if necessary.
    auto& api = engine.getDriverApi();
    api.setupExternalImage(image);

    auto texture =
            api.createTextureExternalImagePlane(mFormat, mWidth, mHeight, mUsage, image, plane);

    if (mTextureIsSwizzled) {
        auto const& s = mSwizzle;
        auto swizzleView = api.createTextureViewSwizzle(texture, s[0], s[1], s[2], s[3]);
        api.destroyTexture(texture);
        texture = swizzleView;
    }

    setHandles(texture);
}

void FTexture::setExternalStream(FEngine& engine, FStream* stream) noexcept {
    FILAMENT_CHECK_PRECONDITION(mExternal) << "The texture must be external.";

    auto& api = engine.getDriverApi();
    auto texture = api.createTexture(
            mTarget, mLevelCount, mFormat, mSampleCount, mWidth, mHeight, mDepth, mUsage);

    if (mTextureIsSwizzled) {
        auto const& s = mSwizzle;
        auto swizzleView = api.createTextureViewSwizzle(texture, s[0], s[1], s[2], s[3]);
        api.destroyTexture(texture);
        texture = swizzleView;
    }

    setHandles(texture);

    if (stream) {
        mStream = stream;
        api.setExternalStream(mHandle, stream->getHandle());
    } else {
        mStream = nullptr;
        api.setExternalStream(mHandle, backend::Handle<HwStream>());
    }
}

void FTexture::generateMipmaps(FEngine& engine) const noexcept {
    FILAMENT_CHECK_PRECONDITION(!mExternal)
            << "External Textures are not mipmappable.";

    FILAMENT_CHECK_PRECONDITION(mTarget != SamplerType::SAMPLER_3D)
            << "3D Textures are not mipmappable.";

    const bool formatMipmappable = engine.getDriverApi().isTextureFormatMipmappable(mFormat);
    FILAMENT_CHECK_PRECONDITION(formatMipmappable)
            << "Texture format " << (unsigned)mFormat << " is not mipmappable.";

    if (mLevelCount < 2 || (mWidth == 1 && mHeight == 1)) {
        return;
    }

    engine.getDriverApi().generateMipmaps(mHandle);
    // this method shouldn't have been const
    const_cast<FTexture*>(this)->updateLodRange(0, mLevelCount);
}

bool FTexture::textureHandleCanMutate() const noexcept {
    return (any(mUsage & Usage::SAMPLEABLE) && mLevelCount > 1) || mExternal;
}

void FTexture::updateLodRange(uint8_t const baseLevel, uint8_t const levelCount) noexcept {
    assert_invariant(!mExternal);
    if (any(mUsage & Usage::SAMPLEABLE) && mLevelCount > 1) {
        auto& range = mLodRange;
        uint8_t const last = int8_t(baseLevel + levelCount);
        if (range.first > baseLevel || range.last < last) {
            if (range.empty()) {
                range = { baseLevel, last };
            } else {
                range.first = std::min(range.first, baseLevel);
                range.last = std::max(range.last, last);
            }
            // We defer the creation of the texture view to getHwHandleForSampling() because it
            // is a common case that by then, the view won't be needed. Creating the first view on a
            // texture has a backend cost.
        }
    }
}

void FTexture::setHandles(Handle<HwTexture> handle) noexcept {
    assert_invariant(!mHandle || mHandleForSampling);
    if (mHandle) {
        mDriver->destroyTexture(mHandle);
    }
    if (mHandleForSampling != mHandle) {
        mDriver->destroyTexture(mHandleForSampling);
    }
    mHandle = handle;
    mHandleForSampling = handle;
}

Handle<HwTexture> FTexture::setHandleForSampling(
        Handle<HwTexture> handle) const noexcept {
    assert_invariant(!mHandle || mHandleForSampling);
    if (mHandleForSampling && mHandleForSampling != mHandle) {
        mDriver->destroyTexture(mHandleForSampling);
    }
    return mHandleForSampling = handle;
}

Handle<HwTexture> FTexture::createPlaceholderTexture(
        DriverApi& driver) noexcept {
    auto handle = driver.createTexture(
            Sampler::SAMPLER_2D, 1, InternalFormat::RGBA8, 1, 1, 1, 1, Usage::DEFAULT);
    static uint8_t pixels[4] = { 0, 0, 0, 0 };
    driver.update3DImage(handle, 0, 0, 0, 0, 1, 1, 1,
            { (char*)&pixels[0], sizeof(pixels),
                    PixelBufferDescriptor::PixelDataFormat::RGBA,
                    PixelBufferDescriptor::PixelDataType::UBYTE });
    return handle;
}

Handle<HwTexture> FTexture::getHwHandleForSampling() const noexcept {
    if (UTILS_UNLIKELY(mExternal && !mHandleForSampling)) {
        return setHandleForSampling(createPlaceholderTexture(*mDriver));
    }
    auto const& range = mLodRange;
    auto& activeRange = mActiveLodRange;
    bool const lodRangeChanged = activeRange.first != range.first || activeRange.last != range.last;
    if (UTILS_UNLIKELY(lodRangeChanged)) {
        activeRange = range;
        if (range.empty() || hasAllLods(range)) {
            std::ignore = setHandleForSampling(mHandle);
        } else {
            std::ignore = setHandleForSampling(mDriver->createTextureView(
                mHandle, range.first, range.size()));
        }
    }
    return mHandleForSampling;
}

void FTexture::updateLodRange(uint8_t const level) noexcept {
    updateLodRange(level, 1);
}

bool FTexture::isTextureFormatSupported(FEngine& engine, InternalFormat const format) noexcept {
    return engine.getDriverApi().isTextureFormatSupported(format);
}

bool FTexture::isTextureFormatMipmappable(FEngine& engine, InternalFormat const format) noexcept {
    return engine.getDriverApi().isTextureFormatMipmappable(format);
}

bool FTexture::isTextureFormatCompressed(InternalFormat const format) noexcept {
    return isCompressedFormat(format);
}

bool FTexture::isProtectedTexturesSupported(FEngine& engine) noexcept {
    return engine.getDriverApi().isProtectedTexturesSupported();
}

bool FTexture::isTextureSwizzleSupported(FEngine& engine) noexcept {
    return engine.getDriverApi().isTextureSwizzleSupported();
}

size_t FTexture::getMaxTextureSize(FEngine& engine, Sampler type) noexcept {
    return engine.getDriverApi().getMaxTextureSize(type);
}

size_t FTexture::getMaxArrayTextureLayers(FEngine& engine) noexcept {
    return engine.getDriverApi().getMaxArrayTextureLayers();
}

size_t FTexture::computeTextureDataSize(Format const format, Type const type,
        size_t const stride, size_t const height, size_t const alignment) noexcept {
    return PixelBufferDescriptor::computeDataSize(format, type, stride, height, alignment);
}

size_t FTexture::getFormatSize(InternalFormat const format) noexcept {
    return backend::getFormatSize(format);
}

TextureType FTexture::getTextureType() const noexcept {
    return mTextureType;
}

bool FTexture::validatePixelFormatAndType(TextureFormat const internalFormat,
        PixelDataFormat const format, PixelDataType const type) noexcept {

    switch (internalFormat) {
        case TextureFormat::R8:
        case TextureFormat::R8_SNORM:
        case TextureFormat::R16F:
        case TextureFormat::R32F:
            if (format != PixelDataFormat::R) {
                return false;
            }
            break;

        case TextureFormat::R8UI:
        case TextureFormat::R8I:
        case TextureFormat::R16UI:
        case TextureFormat::R16I:
        case TextureFormat::R32UI:
        case TextureFormat::R32I:
            if (format != PixelDataFormat::R_INTEGER) {
                return false;
            }
            break;

        case TextureFormat::RG8:
        case TextureFormat::RG8_SNORM:
        case TextureFormat::RG16F:
        case TextureFormat::RG32F:
            if (format != PixelDataFormat::RG) {
                return false;
            }
            break;

        case TextureFormat::RG8UI:
        case TextureFormat::RG8I:
        case TextureFormat::RG16UI:
        case TextureFormat::RG16I:
        case TextureFormat::RG32UI:
        case TextureFormat::RG32I:
            if (format != PixelDataFormat::RG_INTEGER) {
                return false;
            }
            break;

        case TextureFormat::RGB565:
        case TextureFormat::RGB9_E5:
        case TextureFormat::RGB5_A1:
        case TextureFormat::RGBA4:
        case TextureFormat::RGB8:
        case TextureFormat::SRGB8:
        case TextureFormat::RGB8_SNORM:
        case TextureFormat::R11F_G11F_B10F:
        case TextureFormat::RGB16F:
        case TextureFormat::RGB32F:
            if (format != PixelDataFormat::RGB) {
                return false;
            }
            break;

        case TextureFormat::RGB8UI:
        case TextureFormat::RGB8I:
        case TextureFormat::RGB16UI:
        case TextureFormat::RGB16I:
        case TextureFormat::RGB32UI:
        case TextureFormat::RGB32I:
            if (format != PixelDataFormat::RGB_INTEGER) {
                return false;
            }
            break;

        case TextureFormat::RGBA8:
        case TextureFormat::SRGB8_A8:
        case TextureFormat::RGBA8_SNORM:
        case TextureFormat::RGB10_A2:
        case TextureFormat::RGBA16F:
        case TextureFormat::RGBA32F:
            if (format != PixelDataFormat::RGBA) {
                return false;
            }
            break;

        case TextureFormat::RGBA8UI:
        case TextureFormat::RGBA8I:
        case TextureFormat::RGBA16UI:
        case TextureFormat::RGBA16I:
        case TextureFormat::RGBA32UI:
        case TextureFormat::RGBA32I:
            if (format != PixelDataFormat::RGBA_INTEGER) {
                return false;
            }
            break;

        case TextureFormat::STENCIL8:
            // there is no pixel data type that can be used for this format
            return false;

        case TextureFormat::DEPTH16:
        case TextureFormat::DEPTH24:
        case TextureFormat::DEPTH32F:
            if (format != PixelDataFormat::DEPTH_COMPONENT) {
                return false;
            }
            break;

        case TextureFormat::DEPTH24_STENCIL8:
        case TextureFormat::DEPTH32F_STENCIL8:
            if (format != PixelDataFormat::DEPTH_STENCIL) {
                return false;
            }
            break;

        case TextureFormat::UNUSED:
        case TextureFormat::EAC_R11:
        case TextureFormat::EAC_R11_SIGNED:
        case TextureFormat::EAC_RG11:
        case TextureFormat::EAC_RG11_SIGNED:
        case TextureFormat::ETC2_RGB8:
        case TextureFormat::ETC2_SRGB8:
        case TextureFormat::ETC2_RGB8_A1:
        case TextureFormat::ETC2_SRGB8_A1:
        case TextureFormat::ETC2_EAC_RGBA8:
        case TextureFormat::ETC2_EAC_SRGBA8:
        case TextureFormat::DXT1_RGB:
        case TextureFormat::DXT1_RGBA:
        case TextureFormat::DXT3_RGBA:
        case TextureFormat::DXT5_RGBA:
        case TextureFormat::DXT1_SRGB:
        case TextureFormat::DXT1_SRGBA:
        case TextureFormat::DXT3_SRGBA:
        case TextureFormat::DXT5_SRGBA:
        case TextureFormat::RED_RGTC1:
        case TextureFormat::SIGNED_RED_RGTC1:
        case TextureFormat::RED_GREEN_RGTC2:
        case TextureFormat::SIGNED_RED_GREEN_RGTC2:
        case TextureFormat::RGB_BPTC_SIGNED_FLOAT:
        case TextureFormat::RGB_BPTC_UNSIGNED_FLOAT:
        case TextureFormat::RGBA_BPTC_UNORM:
        case TextureFormat::SRGB_ALPHA_BPTC_UNORM:
        case TextureFormat::RGBA_ASTC_4x4:
        case TextureFormat::RGBA_ASTC_5x4:
        case TextureFormat::RGBA_ASTC_5x5:
        case TextureFormat::RGBA_ASTC_6x5:
        case TextureFormat::RGBA_ASTC_6x6:
        case TextureFormat::RGBA_ASTC_8x5:
        case TextureFormat::RGBA_ASTC_8x6:
        case TextureFormat::RGBA_ASTC_8x8:
        case TextureFormat::RGBA_ASTC_10x5:
        case TextureFormat::RGBA_ASTC_10x6:
        case TextureFormat::RGBA_ASTC_10x8:
        case TextureFormat::RGBA_ASTC_10x10:
        case TextureFormat::RGBA_ASTC_12x10:
        case TextureFormat::RGBA_ASTC_12x12:
        case TextureFormat::SRGB8_ALPHA8_ASTC_4x4:
        case TextureFormat::SRGB8_ALPHA8_ASTC_5x4:
        case TextureFormat::SRGB8_ALPHA8_ASTC_5x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_6x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_6x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x8:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x8:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x10:
        case TextureFormat::SRGB8_ALPHA8_ASTC_12x10:
        case TextureFormat::SRGB8_ALPHA8_ASTC_12x12:
            return false;
    }

    switch (internalFormat) {
        case TextureFormat::R8:
        case TextureFormat::R8UI:
        case TextureFormat::RG8:
        case TextureFormat::RG8UI:
        case TextureFormat::RGB8:
        case TextureFormat::SRGB8:
        case TextureFormat::RGB8UI:
        case TextureFormat::RGBA8:
        case TextureFormat::SRGB8_A8:
        case TextureFormat::RGBA8UI:
            if (type != PixelDataType::UBYTE) {
                return false;
            }
            break;

        case TextureFormat::R8_SNORM:
        case TextureFormat::R8I:
        case TextureFormat::RG8_SNORM:
        case TextureFormat::RG8I:
        case TextureFormat::RGB8_SNORM:
        case TextureFormat::RGB8I:
        case TextureFormat::RGBA8_SNORM:
        case TextureFormat::RGBA8I:
            if (type != PixelDataType::BYTE) {
                return false;
            }
            break;

        case TextureFormat::R16F:
        case TextureFormat::RG16F:
        case TextureFormat::RGB16F:
        case TextureFormat::RGBA16F:
            if (type != PixelDataType::FLOAT && type != PixelDataType::HALF) {
                return false;
            }
            break;

        case TextureFormat::R32F:
        case TextureFormat::RG32F:
        case TextureFormat::RGB32F:
        case TextureFormat::RGBA32F:
        case TextureFormat::DEPTH32F:
            if (type != PixelDataType::FLOAT) {
                return false;
            }
            break;

        case TextureFormat::R16UI:
        case TextureFormat::RG16UI:
        case TextureFormat::RGB16UI:
        case TextureFormat::RGBA16UI:
            if (type != PixelDataType::USHORT) {
                return false;
            }
            break;

        case TextureFormat::R16I:
        case TextureFormat::RG16I:
        case TextureFormat::RGB16I:
        case TextureFormat::RGBA16I:
            if (type != PixelDataType::SHORT) {
                return false;
            }
            break;


        case TextureFormat::R32UI:
        case TextureFormat::RG32UI:
        case TextureFormat::RGB32UI:
        case TextureFormat::RGBA32UI:
            if (type != PixelDataType::UINT) {
                return false;
            }
            break;

        case TextureFormat::R32I:
        case TextureFormat::RG32I:
        case TextureFormat::RGB32I:
        case TextureFormat::RGBA32I:
            if (type != PixelDataType::INT) {
                return false;
            }
            break;

        case TextureFormat::RGB565:
            if (type != PixelDataType::UBYTE && type != PixelDataType::USHORT_565) {
                return false;
            }
            break;


        case TextureFormat::RGB9_E5:
            // TODO: we're missing UINT_5_9_9_9_REV
            if (type != PixelDataType::FLOAT && type != PixelDataType::HALF) {
                return false;
            }
            break;

        case TextureFormat::RGB5_A1:
            // TODO: we're missing USHORT_5_5_5_1
            if (type != PixelDataType::UBYTE && type != PixelDataType::UINT_2_10_10_10_REV) {
                return false;
            }
            break;

        case TextureFormat::RGBA4:
            // TODO: we're missing USHORT_4_4_4_4
            if (type != PixelDataType::UBYTE) {
                return false;
            }
            break;

        case TextureFormat::R11F_G11F_B10F:
            if (type != PixelDataType::FLOAT && type != PixelDataType::HALF
                && type != PixelDataType::UINT_10F_11F_11F_REV) {
                return false;
            }
            break;

        case TextureFormat::RGB10_A2:
            if (type != PixelDataType::UINT_2_10_10_10_REV) {
                return false;
            }
            break;

        case TextureFormat::STENCIL8:
            // there is no pixel data type that can be used for this format
            return false;

        case TextureFormat::DEPTH16:
            if (type != PixelDataType::UINT && type != PixelDataType::USHORT) {
                return false;
            }
            break;

        case TextureFormat::DEPTH24:
            if (type != PixelDataType::UINT) {
                return false;
            }
            break;

        case TextureFormat::DEPTH24_STENCIL8:
            // TODO: we're missing UINT_24_8
            return false;

        case TextureFormat::DEPTH32F_STENCIL8:
            // TODO: we're missing FLOAT_UINT_24_8_REV
            return false;

        case TextureFormat::UNUSED:
        case TextureFormat::EAC_R11:
        case TextureFormat::EAC_R11_SIGNED:
        case TextureFormat::EAC_RG11:
        case TextureFormat::EAC_RG11_SIGNED:
        case TextureFormat::ETC2_RGB8:
        case TextureFormat::ETC2_SRGB8:
        case TextureFormat::ETC2_RGB8_A1:
        case TextureFormat::ETC2_SRGB8_A1:
        case TextureFormat::ETC2_EAC_RGBA8:
        case TextureFormat::ETC2_EAC_SRGBA8:
        case TextureFormat::DXT1_RGB:
        case TextureFormat::DXT1_RGBA:
        case TextureFormat::DXT3_RGBA:
        case TextureFormat::DXT5_RGBA:
        case TextureFormat::DXT1_SRGB:
        case TextureFormat::DXT1_SRGBA:
        case TextureFormat::DXT3_SRGBA:
        case TextureFormat::DXT5_SRGBA:
        case TextureFormat::RED_RGTC1:
        case TextureFormat::SIGNED_RED_RGTC1:
        case TextureFormat::RED_GREEN_RGTC2:
        case TextureFormat::SIGNED_RED_GREEN_RGTC2:
        case TextureFormat::RGB_BPTC_SIGNED_FLOAT:
        case TextureFormat::RGB_BPTC_UNSIGNED_FLOAT:
        case TextureFormat::RGBA_BPTC_UNORM:
        case TextureFormat::SRGB_ALPHA_BPTC_UNORM:
        case TextureFormat::RGBA_ASTC_4x4:
        case TextureFormat::RGBA_ASTC_5x4:
        case TextureFormat::RGBA_ASTC_5x5:
        case TextureFormat::RGBA_ASTC_6x5:
        case TextureFormat::RGBA_ASTC_6x6:
        case TextureFormat::RGBA_ASTC_8x5:
        case TextureFormat::RGBA_ASTC_8x6:
        case TextureFormat::RGBA_ASTC_8x8:
        case TextureFormat::RGBA_ASTC_10x5:
        case TextureFormat::RGBA_ASTC_10x6:
        case TextureFormat::RGBA_ASTC_10x8:
        case TextureFormat::RGBA_ASTC_10x10:
        case TextureFormat::RGBA_ASTC_12x10:
        case TextureFormat::RGBA_ASTC_12x12:
        case TextureFormat::SRGB8_ALPHA8_ASTC_4x4:
        case TextureFormat::SRGB8_ALPHA8_ASTC_5x4:
        case TextureFormat::SRGB8_ALPHA8_ASTC_5x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_6x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_6x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_8x8:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x5:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x6:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x8:
        case TextureFormat::SRGB8_ALPHA8_ASTC_10x10:
        case TextureFormat::SRGB8_ALPHA8_ASTC_12x10:
        case TextureFormat::SRGB8_ALPHA8_ASTC_12x12:
            return false;
    }

    return true;
}

} // namespace filament
