#include "arms.h"

#include <sstream>

namespace gsbench {

namespace {

std::vector<Arm> BuildRegistry() {
    // Selectors verified against gray_scott_threeway_bench.cu TEST_CASE tags (~line 2306+).
    // Env overrides verified against run_campaign.sh's tag_of()/case block (the superset --
    // it covers all 11 arms; run_threeway_bench.sh's run_arm calls agree on the 7 they share).
    return {
        // raw arm: no CLIO server. GSBENCH_RAW_INLINE selects the writer structure.
        {"raw_inline",    "[gsbench_raw]",        {{"GSBENCH_RAW_INLINE", "1"}}, false, false},
        {"raw_threaded",  "[gsbench_raw]",        {{"GSBENCH_RAW_INLINE", "0"}}, false, false},

        // hdf5: same TEST_CASE, GSBENCH_RAW_INLINE picks inline (GPU-idle) vs threaded
        // (background writer, overlaps compute) -- PrintResult names the row accordingly.
        {"hdf5_inline",   "[gsbench_hdf5]",       {{"GSBENCH_RAW_INLINE", "1"}}, false, false},
        {"hdf5_threaded", "[gsbench_hdf5]",       {{"GSBENCH_RAW_INLINE", "0"}}, false, false},

        // hdf5_naive: typical-user baseline -- always inline structure, naive dataset config.
        {"hdf5_naive",    "[gsbench_hdf5_naive]", {{"GSBENCH_RAW_INLINE", "1"},
                                                     {"GSBENCH_HDF5_NAIVE", "1"}}, false, false},

        // hdf5_async: needs the async-VOL LD_LIBRARY_PATH/HDF5_PLUGIN_PATH/HDF5_VOL_CONNECTOR
        // env, applied specially by the runner (see runner.cpp ApplyArmEnv), plus
        // GSBENCH_HDF5_ASYNC_FWAIT=0 to avoid the H5VL_async_file_wait busy-spin livelock.
        {"hdf5_async",    "[gsbench_hdf5_async]", {{"GSBENCH_HDF5_ASYNC_FWAIT", "0"}}, true, false},

        {"hostclio",      "[gsbench_hostclio]",   {}, false, false},

        // GPUH5 arms. gpuh5 (reuse) is the canonical/default design: GPU-initiated, bounded
        // (2 reused groups) memory. gpuh5_noreuse is the old one-dataset-per-snapshot async
        // path (memory linear in snaps); gpuh5_sync is the fused submit-AND-wait variant.
        // gpuh5_sync: fused submit-AND-wait (no double buffering). Two data backends as
        // separate arms so the sync-vs-sync comparison against persistent_sync (which is
        // FORCED pinned) can isolate the data-backend cost from the cooperative-kernel cost.
        {"gpuh5_sync",           "[gsbench_gpuh5_sync]",    {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_sync_pinned",    "[gsbench_gpuh5_sync]",    {{"GSBENCH_DATA_PINNED", "1"}}, false, false},
        {"gpuh5_noreuse",        "[gsbench_gpuh5_noreuse]", {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_noreuse_pinned", "[gsbench_gpuh5_noreuse]", {{"GSBENCH_DATA_PINNED", "1"}}, false, false},

        // pooled: GSBENCH_POOL=<M> is config-supplied (the campaign's Study P sweeps M), so no
        // fixed override here -- the runner sets it per work-unit when is_pooled is true.
        {"pooled",        "[gsbench_pooled]",     {}, false, true},

        // gpuh5 (reuse): the DEFAULT GPUH5 design -- async's shape but memory constant in
        // snapshots (2 reused buffer groups, snap % 2, drain-before-refill + device tag-stamp;
        // DESIGN §7 "Option B", relaunched). Checksum must equal the other GPUH5 arms.
        // _pinned = the same arm with kPinnedHost data (the backend the persistent arm is forced
        // onto), so persistent-vs-gpuh5 can be compared on an equal data backend.
        {"gpuh5",         "[gsbench_gpuh5]",      {{"GSBENCH_DATA_PINNED", "0"}}, false, false},
        {"gpuh5_pinned",  "[gsbench_gpuh5]",      {{"GSBENCH_DATA_PINNED", "1"}}, false, false},

        // persistent: the WHOLE snapshot loop in ONE resident cooperative kernel (grid.sync()
        // between steps) -- DESIGN §7 "Option A", measured head-to-head vs the relaunched reuse
        // arm. Same bounded 2-group memory; forces kPinnedHost data (deadlock safety).
        // persistent_sync: same resident kernel but fire-AND-wait each snapshot (no overlap) --
        // the persistent analog of gpuh5_sync.
        {"persistent",      "[gsbench_persistent]",      {}, false, false},
        {"persistent_sync", "[gsbench_persistent_sync]", {}, false, false},
    };
}

}  // namespace

const std::vector<Arm>& AllArms() {
    static const std::vector<Arm> kArms = BuildRegistry();
    return kArms;
}

const Arm* FindArm(const std::string& name) {
    for (const auto& a : AllArms()) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

std::vector<std::string> DefaultArmNames() {
    return {"raw_inline", "raw_threaded", "hostclio",
            "gpuh5", "gpuh5_noreuse", "gpuh5_sync",
            "hdf5_inline", "hdf5_threaded"};
}

std::optional<std::vector<std::string>> ParseArmList(const std::string& csv, std::string& err) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        if (!FindArm(tok)) {
            err = "unknown arm: " + tok;
            return std::nullopt;
        }
        out.push_back(tok);
    }
    if (out.empty()) {
        err = "empty --arms list";
        return std::nullopt;
    }
    return out;
}

}  // namespace gsbench
