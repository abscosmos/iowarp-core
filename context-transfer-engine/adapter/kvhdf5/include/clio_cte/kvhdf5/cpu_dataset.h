#pragma once

#include "defines.h"
#include "blob_backend.h"
#include "chunking.h"
#include "dataset_meta.h"
#include <vector>
#include <cuda/std/span>
#include <cuda/std/expected>

namespace kvhdf5 {

enum class IoError : uint8_t { BadLayout, SizeMismatch, Unsupported, NotFound, BackendFailure };

// Non-owning view: a Dataset is valid only while its originating File is alive
// and not moved (it holds borrowed DatasetMeta* and backend B* into the File).
//
// MVP: one chunk per dataset. `data`/`out` are the full row-major array.
// Write/Read reject any Layout with ChunkCount() > 1 (IoError::Unsupported) --
// real multi-chunk tiling and concurrent I/O only exist on the GPU producer
// path (GpuCteDataset), which is not built on top of this class.
//
// Not currently used outside its own unit test (dataset_io_test.cpp), and
// there only against InMemBlobBackend -- no real CTE-backed BlobBackend
// exists for this template, and no e2e test or benchmark instantiates it.
// Other files only include this header for the Layout struct.
template<BlobBackend B>
class Dataset {
    DatasetMeta* meta_;
    B* backend_;

    static cstd::span<const char> ChunkZeroName(const Layout& L,
            char (&out)[chunking::kMaxBlobNameLen + 1]) {
        uint64_t coord[MAX_DIMS] = {};
        return chunking::ChunkCoordToName({coord, L.dims.size()}, out);
    }

public:
    Dataset(DatasetMeta* meta, B* backend) : meta_(meta), backend_(backend) {}

    [[nodiscard]] const DatasetMeta& Meta() const { return *meta_; }

    [[nodiscard]] cstd::expected<void, IoError> Write(cstd::span<const byte_t> data) {
        const Layout& L = meta_->layout;
        if (!L.Valid()) return cstd::unexpected(IoError::BadLayout);
        if (data.size() != L.TotalBytes()) return cstd::unexpected(IoError::SizeMismatch);
        if (!L.IsSingleChunk()) return cstd::unexpected(IoError::Unsupported);

        char name[chunking::kMaxBlobNameLen + 1];
        auto chunk_name = ChunkZeroName(L, name);
        if (chunk_name.empty()) return cstd::unexpected(IoError::BadLayout);

        if (!backend_->WriteChunk(chunk_name, data))
            return cstd::unexpected(IoError::BackendFailure);
        return {};
    }

    [[nodiscard]] cstd::expected<cstd::span<byte_t>, IoError> Read(cstd::span<byte_t> out) {
        const Layout& L = meta_->layout;
        if (!L.Valid()) return cstd::unexpected(IoError::BadLayout);
        if (out.size() < L.TotalBytes()) return cstd::unexpected(IoError::SizeMismatch);
        if (!L.IsSingleChunk()) return cstd::unexpected(IoError::Unsupported);

        char name[chunking::kMaxBlobNameLen + 1];
        auto chunk_name = ChunkZeroName(L, name);
        if (chunk_name.empty()) return cstd::unexpected(IoError::BadLayout);

        auto r = backend_->ReadChunk(chunk_name, out);
        if (!r) {
            IoError e = r.error() == BlobError::NotFound       ? IoError::NotFound
                      : r.error() == BlobError::NotEnoughSpace ? IoError::SizeMismatch
                                                               : IoError::BackendFailure;
            return cstd::unexpected(e);
        }
        return *r;
    }
};

}  // namespace kvhdf5
