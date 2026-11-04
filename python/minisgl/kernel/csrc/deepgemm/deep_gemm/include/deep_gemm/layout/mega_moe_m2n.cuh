#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// Split MegaMoE M2N workspace: AG (token source) ranks are union ranks
// [0, num_ag_ranks), EG (expert owner) ranks are [num_ag_ranks, num_ag_ranks +
// num_eg_ranks).  The layout must be byte-identical on every union rank so
// that `SymBuffer::map` translates the same offsets.
//
// Differences from the symmetric `Workspace`:
//  - `num_experts_per_rank` divides over EG ranks only;
//  - the source-rank dimension (recv counts, src token slots, pool sizing)
//    spans AG ranks only.
struct WorkspaceM2N {
    void* base;
    uint32_t num_ag_ranks, num_eg_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts of an EG rank share a contiguous pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    static constexpr uint64_t kNumBarrierSignalBytes = Workspace::kNumBarrierSignalBytes;
    static constexpr uint32_t kNumMaxGridSyncCounters = Workspace::kNumMaxGridSyncCounters;

    CUTLASS_HOST_DEVICE
    WorkspaceM2N(void* base,
                 const uint32_t& num_ag_ranks,
                 const uint32_t& num_eg_ranks,
                 const uint32_t& num_experts,
                 const uint32_t& num_max_tokens_per_rank,
                 const uint32_t& num_topk):
        base(base),
        num_ag_ranks(num_ag_ranks), num_eg_ranks(num_eg_ranks),
        num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_eg_ranks;
        num_max_recv_tokens_per_expert = num_ag_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ag_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t);
        num_bytes += num_ag_ranks * num_experts_per_rank * sizeof(uint64_t);

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source token-topk
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) * num_ag_ranks *
                     num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    // Per-(source AG rank, local expert) receive counts on EG ranks
    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_recv_count_ptr(num_ag_ranks) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // For dispatch pulling: [local expert][source AG rank][slot]
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ag_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

// Control words of the point-to-point (P2P) protocol, the last kNumCtrlBytes of the
// symmetric buffer (after the push-dispatch staging rows, so every older offset is
// unchanged and the host finds them from the buffer size alone). P2P replaces the
// union-wide tag-2/tag-3 barriers with per-pair counters, which only grow or have a
// round's target taken off, so an early increment of the next round is never lost:
//  - credit: each EG rank adds 1 after it cleaned a lane's workspace; an AG rank's
//    next push on that lane needs kNumEGRanks more than its previous round took.
//  - combine[parity]: each EG rank adds 1 after it wrote all combine rows of a layer
//    for an AG rank, and the AG rank that dispatched those tokens adds 1 after its
//    count push; the round is complete at kNumEGRanks + 1. Indexed by the model layer's parity: an EG pass
//    can release a source that is already a layer ahead.
// The other words are rank-local state of the AG credit check and of EG early start.
struct P2PRegionsM2N {
    static constexpr uint64_t kNumCtrlBytes = 64;

    void* ctrl_base;

    CUTLASS_HOST_DEVICE
    explicit P2PRegionsM2N(void* start): ctrl_base(start) {}

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(ctrl_base, kNumCtrlBytes);
    }

    CUTLASS_DEVICE
    int* get_credit_ptr() const {
        return static_cast<int*>(ctrl_base);
    }

    CUTLASS_DEVICE
    int* get_combine_count_ptr(const uint32_t& parity) const {
        return static_cast<int*>(ctrl_base) + 1 + parity;
    }

    // EG ranks only (their own buffer): the early-start decision of the current
    // (layer, lane) -- valid bit 31, a layer tag in bits 24..30, the first pass's
    // source mask below -- and a running count of rounds split into two passes.
    CUTLASS_DEVICE
    int* get_eg_decision_ptr() const {
        return static_cast<int*>(ctrl_base) + 3;
    }

    CUTLASS_DEVICE
    int* get_eg_split_count_ptr() const {
        return static_cast<int*>(ctrl_base) + 4;
    }

    // EG ranks only: the AG ranks whose counts of the current round are in, one
    // bit each, OR-ed by each AG rank after its count push (a hint for the
    // early-start decision: the passes still wait for each source's own count
    // slots), and cleared with the decision before the credits.
    CUTLASS_DEVICE
    int* get_eg_arrival_ptr() const {
        return static_cast<int*>(ctrl_base) + 5;
    }

    // EG ranks only, early start's adaptive policy for this lane: how long the
    // late sources came after the others (ns), and a pass's time per expert it
    // serves (ns), both running averages.
    CUTLASS_DEVICE
    int* get_eg_late_ns_ptr() const {
        return static_cast<int*>(ctrl_base) + 6;
    }

    CUTLASS_DEVICE
    int* get_eg_expert_ns_ptr() const {
        return static_cast<int*>(ctrl_base) + 7;
    }

    // AG ranks only (their own buffer): the credit word only grows; these are how
    // far this rank's dispatches consumed it, and how many SMs of the current
    // dispatch launch have checked it (the last one advances the consumed value).
    // Words 8 and 9 are the monotonic combine waits' (megamoe_m2n_mega.py).
    CUTLASS_DEVICE
    int* get_credit_consumed_ptr() const {
        return static_cast<int*>(ctrl_base) + 10;
    }

    CUTLASS_DEVICE
    int* get_credit_checked_ptr() const {
        return static_cast<int*>(ctrl_base) + 11;
    }
};

} // namespace deep_gemm::layout
