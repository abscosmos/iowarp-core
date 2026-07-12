#pragma once

#include "defines.h"
#include "chunking.h"
#include <cstdint>
#include <vector>
#include <cuda/std/span>

namespace kvhdf5 {

struct Layout {
    std::vector<uint64_t> dims;
    std::vector<uint64_t> chunk_dims;
    uint64_t elem_size = 0;  // bytes per element

    [[nodiscard]] bool Valid() const {
        if (dims.empty() || dims.size() != chunk_dims.size() || dims.size() > MAX_DIMS) return false;
        if (elem_size == 0) return false;
        for (size_t i = 0; i < dims.size(); ++i)
            if (chunk_dims[i] == 0 || chunk_dims[i] > dims[i]) return false;
        return true;
    }
    [[nodiscard]] cstd::span<const uint64_t> Dims() const { return {dims.data(), dims.size()}; }
    [[nodiscard]] cstd::span<const uint64_t> ChunkDims() const { return {chunk_dims.data(), chunk_dims.size()}; }
    [[nodiscard]] uint64_t TotalElems() const { uint64_t n = 1; for (uint64_t d : dims) n *= d; return n; }
    [[nodiscard]] uint64_t TotalBytes() const { return TotalElems() * elem_size; }
    [[nodiscard]] uint64_t ChunkCount() const { return chunking::ChunkCount(Dims(), ChunkDims()); }
    [[nodiscard]] bool IsSingleChunk() const { return ChunkCount() == 1; }
};

}  // namespace kvhdf5
