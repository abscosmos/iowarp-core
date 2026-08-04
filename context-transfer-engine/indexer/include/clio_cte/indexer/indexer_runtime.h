/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_CTE_INDEXER_INDEXER_RUNTIME_H_
#define CLIO_CTE_INDEXER_INDEXER_RUNTIME_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/indexer/indexer_client.h>
#include <clio_cte/indexer/indexer_tasks.h>

namespace clio::cte::indexer {

/**
 * Indexer chimod runtime (issue #905) — owns the SemanticSearch index that
 * used to be computed brute-force inside the CTE core (which re-read every
 * candidate blob's bytes on every query). An interposer directly above the
 * core (indexer -> core):
 *
 *  - PutBlob/MultiPutBlob/TruncateBlob forward down `next` FIRST; on
 *    success the affected doc is re-tokenized before the ack
 *    (read-your-writes search). Whole-blob writes — the dominant shape —
 *    tokenize the task's own payload directly; only partial writes re-read
 *    the blob from the chain, via nested inline calls on the co-located
 *    core container (never client round-trips: the dispatch/queue cycle
 *    was the measured dominant cost, see CLIO_RUN_INLINE / #862).
 *  - DelBlob/DelTag drop the affected docs; RenameTag rewrites tag names.
 *  - SemanticSearch TERMINATES here: BM25 over the per-container index
 *    slice, identical semantics to the core's old scan (df/avgdl over the
 *    regex-matched working set; Broadcast + AggregateOut merges shards).
 *  - The index is DERIVED, rebuildable state: never persisted. On restart
 *    each container rebuilds its slice from the storage below (BlobQuery +
 *    GetBlob against `next`) before Create() acks. Blobs whose bytes lived
 *    only on volatile (RAM) tiers cannot be re-read after a reboot and drop
 *    out of the index — the same content is also unreadable by GetBlob, so
 *    the index stays consistent with what storage can actually serve.
 *
 * The index is a FORWARD index (per-doc term frequencies + lengths), not a
 * term->postings inverted index: the search semantics compute BM25 corpus
 * statistics over the regex-matched slice per query, which requires
 * iterating matched docs anyway. What the index buys is eliminating the
 * per-query blob reads and tokenization.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = IndexerConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);
  clio::run::TaskResume DelBlob(
      clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task);
  clio::run::TaskResume DelTag(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  clio::run::TaskResume TruncateBlob(
      clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task);
  clio::run::TaskResume RenameTag(
      clio::run::shared_ptr<clio::cte::core::RenameTagTask> &task);
  clio::run::TaskResume SemanticSearch(
      clio::run::shared_ptr<clio::cte::core::SemanticSearchTask> &task);
  clio::run::TaskResume IndexSweep(
      clio::run::shared_ptr<IndexSweepTask> &task);

  // ---- Container virtuals (defined in autogen/indexer_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  void Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
               clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           clio::run::shared_ptr<clio::run::Task> &orig,
                                           bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /** One indexed document: the tokenization the core's old scan produced
   *  per candidate, kept current instead of recomputed per query. */
  struct IndexedDoc {
    TagId tag_id_;
    std::string tag_name_;
    std::string blob_name_;
    std::unordered_map<std::string, int> tf_;
    size_t length_ = 0;
  };

  /** Re-read one blob's current bytes from the chain and replace its index
   *  entry (erases it when the blob is empty or unreadable — matching the
   *  old scan, which skipped zero-length and unreadable blobs). Fallback
   *  path only: whole-blob puts tokenize their own payload instead. The
   *  read runs as a nested inline call on the core container. */
  clio::run::TaskResume ReindexBlob(TagId tag_id, std::string blob_name);

  /** Post-op logical blob size via a nested inline GetBlobSize on the core
   *  container (0 when absent/failed). */
  clio::run::TaskResume ProbeBlobSize(TagId tag_id,
                                      const std::string &blob_name,
                                      clio::run::u64 *total_out);

  /** Tokenize `data` and replace the blob's index entry (synchronous, no
   *  awaits; locks index_mutex_ internally). */
  void IndexDocBytes(const TagId &tag_id, const std::string &tag_name,
                     const std::string &blob_name, const char *data,
                     clio::run::u64 len);

  /** Mark a blob dirty for the asynchronous drain (O(1), coalesced by doc
   *  key — the put ack path's ONLY indexing cost). */
  void EnqueuePending(const TagId &tag_id, const std::string &blob_name);

  /** Drain the pending set to empty: pop-and-reindex until no entry remains
   *  AND no other drainer is mid-entry. Multiple drainers cooperate (each
   *  pops distinct keys); a search calls this as its read-your-writes
   *  barrier, the periodic sweep calls it for background progress. */
  clio::run::TaskResume DrainPendingIndex();

  /** Rebuild this container's index slice from the storage below (restart
   *  path): enumerate local (tag, blob) pairs via BlobQuery, re-read and
   *  tokenize each. */
  clio::run::TaskResume RebuildIndex();

  /** Resolve (and cache) a tag's full name for result rows / tag_regex. */
  clio::run::TaskResume ResolveTagName(TagId tag_id, std::string *name_out);

  /** Lazily bind the next-pool client (compose next_pool_id). */
  clio::cte::core::Client *GetNextClient();

  static std::string DocKey(const TagId &tag_id, const std::string &blob_name) {
    return std::to_string(tag_id.major_) + "." +
           std::to_string(tag_id.minor_) + "." + blob_name;
  }
  static clio::run::u64 TagKey(const TagId &tag_id) {
    return (static_cast<clio::run::u64>(tag_id.major_) << 32) |
           static_cast<clio::run::u64>(tag_id.minor_);
  }

  IndexerConfig config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;

  // Restart flag: set by Restart() before Init()/Create() (same protocol as
  // the core's is_restart_). Triggers RebuildIndex() in Create().
  bool is_restart_ = false;

  /** The index slice this container owns, keyed "major.minor.blob_name"
   *  (the core's composite key). Guarded by index_mutex_: handlers touch it
   *  only between awaits, but distinct blobs' tasks interleave on the
   *  worker pool. */
  std::unordered_map<std::string, IndexedDoc> index_;
  /** TagId -> resolved full tag name (rewritten by RenameTag). */
  std::unordered_map<clio::run::u64, std::string> tag_names_;

  /** Dirty blobs awaiting re-tokenization, keyed like index_ so overwrites
   *  COALESCE (N puts to a hot blob = one drain-time read+scan). Guarded by
   *  index_mutex_. */
  struct PendingReindex {
    TagId tag_id_;
    std::string blob_name_;
  };
  std::unordered_map<std::string, PendingReindex> pending_;
  /** Entries popped by a drainer but not yet indexed — the search barrier
   *  waits for BOTH pending_ empty and this zero. Guarded by index_mutex_. */
  clio::run::u32 draining_ = 0;

  std::mutex index_mutex_;
};

}  // namespace clio::cte::indexer

#endif  // CLIO_CTE_INDEXER_INDEXER_RUNTIME_H_
