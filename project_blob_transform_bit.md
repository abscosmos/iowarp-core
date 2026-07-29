# Authoritative blob-transform bit (issue #818)

## Problem

The #783 shared-memory metadata cache lets a client copy a blob's payload straight out of
the RAM bdev segment, bypassing the runtime entirely. That fast path's whole safety argument
is *"it can only be faster or equivalent, never wrong"* — it decides whether a payload is
directly readable purely from placement (`every block is node-local and kRam`).

Placement is not the only thing that determines whether the stored bytes are the caller's
bytes. A compressed blob stores `CompressionHeader + codec output`; a future encrypted blob
would store ciphertext. Copying those out verbatim succeeds and returns the wrong thing, with
a success return code and no error anywhere.

Today this is latent rather than live, because core `GetBlob` does not decompress either — so
the RPC path is equally wrong and the two agree. The moment core learns transparent
decompression, the fast path silently diverges.

## Decision: one authoritative bit, not a compression lookup

The issue proposed carrying `compress_lib_`/`compress_preset_` into the cache record and
keying the guard off compression. **We do not key the guard off compression.** The
authoritative signal is a dedicated flag word, `BlobInfo::transform_flags_`, asked via
`BlobInfo::IsTransformed()`. Compression provenance is still carried, but only as
provenance.

Three reasons, in order of weight:

### 1. Compression is not the only transform

Encryption is the obvious next one, and it has exactly the same shape: stored bytes ≠ logical
bytes, offsets into them are meaningless, and length is not preserved. A guard spelled
`compress_lib_ != 0` would have to be found and widened by whoever adds it — i.e. the guard
fails open for every transform nobody thought of yet. `IsTransformed()` tests the whole word
against zero, so a bit set by a newer writer than the reader still answers "yes"; unknown
transforms fail *safe*.

### 2. `compress_lib_` is wrong in both directions today

It answers "which codec was *requested*", not "what is on the device".

**False positive.** `compressor_runtime.cc`, the not-beneficial branch: compression is
attempted, does not shrink the data, and the original bytes are stored — but `AsyncPutBlob`
was called with `context.compress_lib_` still set, and it was only zeroed *afterwards*. The
runtime therefore recorded a codec on a blob holding raw bytes. (This PR also reorders that
assignment, so the provenance field stops lying too.)

**False negative.** Core records `compress_lib_` under `#if CTP_ENABLE_COMPRESS`, and
`CLIO_CTE_ENABLE_COMPRESS` defaults **OFF** — while the metadata log persists the field
unconditionally. A restart can therefore restore a genuinely compressed blob into a runtime
that never populates the field, i.e. a blob that reads as "not compressed" while holding
codec output. That is precisely the silent-wrong-data case the guard exists to prevent.

A guard derived from a field that is wrong in both directions is not a guard.

### 3. The bit must not be able to vanish with a compile flag

`transform_flags_` is compiled **unconditionally**, on both `BlobInfo` and `Context` —
deliberately outside the `#if CTP_ENABLE_COMPRESS` blocks that surround the neighbouring
compression fields.

Beyond the false-negative above, a field whose *presence* depends on a compile flag changes
the size of `Context`, and therefore of `PutBlobTask`, between translation units built with
different flags. That is a live hazard in this tree: `CLIO_CTE_ENABLE_COMPRESS` defaults OFF
while several targets hardcode `CTP_ENABLE_COMPRESS=1`, and this exact skew has already been
diagnosed once as a client/runtime `PutBlobTask` size mismatch. A safety bit is the last
thing that should be able to go missing.

## Design

**Producer-declared, never inferred.** Whoever rewrites the bytes sets the bit. The compressor
chimod sets `context.transform_flags_ |= kBlobTransformed | kBlobTransformCompressed` on the
branch that actually stores compressed bytes, and leaves it clear on the branch that stores
the original. `PutBlobImpl` copies it onto the `BlobInfo`. Core never guesses.

**Sticky.** Once set, a later put does not clear it. A partial overwrite of a compressed blob
with raw bytes leaves a blob that is neither wholly raw nor wholly transformed; the only safe
reading of that is "transformed". Over-refusing costs a round-trip, under-refusing returns
codec bytes as data. Cleared only when the blob is destroyed.

**Refuse at both ends.** `BuildShmBlobRecord` declines to set `kShmBlobDirectReadable` for a
transformed blob, *and* `ShmBlobRecord::IsDirectReadable()` independently rejects a record
whose transform state is non-zero. The redundancy is the cache's existing "default must be
refuse" discipline: a future writer that forgets to clear the flag still cannot cause a
client to hand back codec bytes.

**Record layout is unchanged.** `ShmBlobRecord::transform_flags_` occupies the slot formerly
named `reserved_`, so the record's size and alignment are identical and `kLayoutVersion` does
**not** need bumping. Bumping it would have been worse than useless here: it would force
every pre-existing client to detach and lose the fast path for *all* blobs, when the actual
protection — `kShmBlobDirectReadable` being clear — is a flag those clients already honour.

## Persistence

There are **two** independent persistence paths, and the bit has to survive both. Missing
either one means a restart resurrects a transformed blob as untransformed — which is the
fail-open direction, so this is not optional.

### Metadata log (`FlushMetadata` / `RestoreMetadataFromLog`)

No magic or version header; a bare stream of entries discriminated by a leading `entry_type`
byte. Appending a field to the existing blob entry (type 1) would be misparsed by an older
reader as block data, silently reconstructing garbage placement.

So transform-carrying blob entries are written as a **new entry type 2**. The reader accepts
both: type 2 reads the field; type 1 is legacy and derives the bit conservatively from
`compress_lib_ != 0`. That derivation over-reports (it inherits the false positive above),
which is the safe direction — it costs the fast path, not correctness. An older runtime given
a new log hits the existing unknown-entry-type branch, which warns and stops the restore
rather than misparsing it.

### Transaction log (`ReplayTransactionLogs`)

This log records *placement only* — `kCreateNewBlob` (name + score) and `kExtendBlob`
(blocks). It has never carried compression state, which was harmless while nothing depended
on it and is not harmless now. Two concrete holes, both closed here:

1. **A put not yet flushed loses its mark.** A crash between the put and the next metadata
   flush replays the blob from the transaction log alone. Fixed with a new
   **`kSetBlobTransform` (type 6)** record, logged on the put that first sets the bit — only
   on an actual transition, so repeated puts to an already-marked blob cost nothing.
2. **A stale `kCreateNewBlob` clobbers a restored mark.** Replay runs *after*
   `RestoreMetadataFromLog`, and `kCreateNewBlob` does an `insert_or_assign` of a fresh
   `BlobInfo`. Because the WAL is truncated only once it exceeds a size threshold — not on
   every flush — such a record can outlive the flush that persisted the blob's transform
   state, resetting it to zero. Replay of `kCreateNewBlob` now carries over the transform
   word of any blob already present under that key.

Appending a transaction type is safe for an older runtime: its replay chain is an if/else-if
with no trailing `else`, so unrecognised types are ignored, leaving it exactly where it was
before the record existed.

## Deliberately not changed

`Tag::GetBlobSize`'s fast path still answers from `ShmBlobRecord::total_size_`, which is the
**stored** size — for a transformed blob, the post-transform byte count. It is left alone
because the RPC path returns the same number today, so refusing would cost a round-trip
without changing a single answer. The stored-vs-logical distinction is now documented on the
field itself, so whoever teaches `GetBlob` to decompress transparently has the note in front
of them: at that point `GetBlobSize` must return the logical size, and its fast path must
start consulting `IsTransformed()` too.

Client-side decompression from shared memory — direction (B) in the issue, and the actually
valuable follow-up, since it would skip the round-trip *and* move decompress CPU off the
runtime worker — is out of scope here. It needs codec linkage in the client library, a
decompressed-page cache with an invalidation story, and header-length validation against the
record. This change is the correctness guard that has to land first, so nothing starts
depending on direct reads of transformed data in the meantime.
