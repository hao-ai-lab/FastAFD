#pragma once

#include <functional>
#include <string>
#include <pybind11/functional.h>

#include "../utils/compatibility.hpp"
#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_mega_moe_m2n.hpp"

namespace deep_gemm::mega_m2n {

// Symmetric buffer layout for the split M2N MegaMoE.  The buffer is identical
// on every union rank (AG ranks [0, m), EG ranks [m, m + n)); AG ranks consume
// the input/combine regions, EG ranks consume the pool regions.
static std::tuple<int64_t, int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe_m2n(
    const int& num_ag_ranks, const int& num_eg_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden) {
    DG_HOST_ASSERT(num_experts % num_eg_ranks == 0);

    // Workspace bytes
    const auto workspace = layout::WorkspaceM2N(
        nullptr, num_ag_ranks, num_eg_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 32);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers (used on AG ranks)
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Buffer configs
    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    int num_max_padded_sf_pool_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)
        );
    }

    // L1 input buffer (used on EG ranks)
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer (used on EG ranks)
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer (used on AG ranks)
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(num_max_padded_sf_pool_tokens % 4 == 0);

    // Push-dispatch staging (used on EG ranks): [source AG rank][its token] rows,
    // then the P2P control words, which end the buffer (layout::P2PRegionsM2N), so
    // the caller seeds the credit word at `num_bytes - P2PRegionsM2N::kNumCtrlBytes`.
    const auto staging_token_buffer = layout::Buffer(
        fp8_token_layout, num_ag_ranks, num_max_tokens_per_rank, combine_token_buffer.get_end_ptr());
    const auto staging_sf_buffer = layout::Buffer(
        fp8_sf_layout, num_ag_ranks, num_max_tokens_per_rank, staging_token_buffer.get_end_ptr());
    const auto staging_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, num_ag_ranks, num_max_tokens_per_rank, staging_sf_buffer.get_end_ptr());
    const auto p2p_ctrl = layout::P2PRegionsM2N(staging_topk_weights_buffer.get_end_ptr());
    // AG ranks: the EG ranks' combine rows ([top-k slot][token] bf16), for a combine
    // done outside the AG kernel
    const auto combine_base = reinterpret_cast<int64_t>(combine_token_buffer.base);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto combine_rows = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), combine_base),
            {num_topk, num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                               combine_rows);
    };
    return {reinterpret_cast<int64_t>(p2p_ctrl.get_end_ptr()),
            static_cast<int64_t>(num_max_pool_tokens),
            slice_input_buffers};
}

// AG half: fused (optional router gate + top-k) + per-32 FP8 quant +
// dispatch send + combine recv
static void mega_moe_m2n_ag(
    const torch::Tensor& y,
    const torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& topk_ids,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& gate_weight,
    const bool& renormalize,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_ag_ranks, const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const int& intermediate_hidden,
    const int& num_sms,
    const int& ag_phase,
    const bool& external_quant,
    const bool& direct_combine,
    const int& input_num_topk,
    const int& num_routed_experts,
    const float& shared_topk_weight,
    const std::optional<torch::Tensor>& debug_timings,
    // P2P protocol (see layout::P2PRegionsM2N); `layer_parity` is the model
    // layer's, which indexes the combine counter.
    const bool& p2p,
    const int& layer_parity,
    // Push dispatch: the quant also writes the rows into every EG rank's staging
    // (in-kernel quant with host top-k only); the EG launches must agree.
    const bool& push_dispatch,
    // Combine-only launch behind a wait kernel that saw the combine counter
    // complete and cleared it (P2P): no counter wait, no grid sync.
    const bool& waited_combine
) {
    const auto num_tokens = static_cast<int>(hidden_states.size(0));
    DG_HOST_ASSERT(y.size(0) == num_tokens);
    DG_HOST_ASSERT(not push_dispatch or (not external_quant and not gate_weight.has_value()));
    DG_HOST_ASSERT(not waited_combine or (p2p and ag_phase == 2));
    DG_HOST_ASSERT(ag_phase >= 0 and ag_phase <= 2);
    DG_HOST_ASSERT(layer_parity == 0 or layer_parity == 1);

    DG_HOST_ASSERT(not (external_quant and gate_weight.has_value()));
    const auto hidden = static_cast<int>(y.size(1));
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(hidden_states.scalar_type() == torch::kBFloat16 and hidden_states.is_contiguous());
    DG_HOST_ASSERT(hidden_states.size(1) == hidden);
    if (gate_weight.has_value()) {
        // In-kernel router gate: ids/weights are computed by the kernel
        DG_HOST_ASSERT(not topk_ids.has_value() and not topk_weights.has_value());
        DG_HOST_ASSERT(gate_weight->scalar_type() == torch::kBFloat16 and gate_weight->is_contiguous());
        DG_HOST_ASSERT(gate_weight->size(0) == num_experts and gate_weight->size(1) == hidden);
        DG_HOST_ASSERT(num_experts % 32 == 0 and num_experts / 32 <= 8);
    } else {
        DG_HOST_ASSERT(input_num_topk > 0 and input_num_topk <= num_topk);
        DG_HOST_ASSERT(num_routed_experts > 0 and num_routed_experts <= num_experts);
        DG_HOST_ASSERT(input_num_topk == num_topk or input_num_topk + 1 == num_topk);
        DG_HOST_ASSERT(input_num_topk == num_topk or num_routed_experts < num_experts);
        DG_HOST_ASSERT(topk_ids.has_value() and topk_weights.has_value());
        DG_HOST_ASSERT(topk_ids->scalar_type() == torch::kInt64 or topk_ids->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(topk_ids->is_contiguous() and topk_ids->size(0) == num_tokens and topk_ids->size(1) == input_num_topk);
        DG_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32 and topk_weights->is_contiguous());
        DG_HOST_ASSERT(topk_weights->size(0) == num_tokens and topk_weights->size(1) == input_num_topk);
    }
    DG_HOST_ASSERT(static_cast<int>(sym_buffer_ptrs.size()) == num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ag_ranks);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

    const auto [num_required_bytes, num_max_pool_tokens, slice] =
        get_symm_buffer_size_for_mega_moe_m2n(
            num_ag_ranks, num_eg_ranks, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, x_sf, buf_topk_idx, buf_topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, combine_rows_] = slice(sym_buffer);

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_mega_moe_m2n_ag(y,
                              hidden_states, topk_ids, topk_weights,
                              gate_weight, renormalize,
                              sym_buffer_ptrs, rank_idx,
                              num_ag_ranks, num_eg_ranks,
                              num_max_tokens_per_rank,
                              num_experts, num_tokens, num_topk,
                              input_num_topk, num_routed_experts,
                              hidden, intermediate_hidden,
                              static_cast<int>(num_max_pool_tokens),
                              static_cast<int>(l1_acts_sf.size(0)),
                              num_sms, ag_phase, external_quant,
                              direct_combine,
                              shared_topk_weight,
                              debug_timings,
                              p2p, layer_parity, push_dispatch, waited_combine);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

// EG half: dispatch recv + FP8xFP4 expert MLP + combine send. With
// `compile_only` it builds and loads the instance and returns its generated
// source without launching; otherwise it launches and returns "".
static std::string mega_moe_m2n_eg(
    const torch::Tensor& weight_descs,    // device uint8 [num_layers, 4, 128]
    const torch::Tensor& l1_weight_ptrs,  // device int64 [num_layers]
    const int& num_layers,
    const int& num_experts_per_rank_in,
    const int& hidden_in, const int& intermediate_hidden_in,
    const int& l1_weights_nbytes,
    const bool& use_fp8_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::optional<torch::Tensor>& sym_buffer_lane1,
    const std::optional<torch::Tensor>& sym_buffer_lane2,
    const std::optional<torch::Tensor>& sym_buffer_lane3,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& sym_buffer_ptrs_lane1,
    const std::vector<int64_t>& sym_buffer_ptrs_lane2,
    const std::vector<int64_t>& sym_buffer_ptrs_lane3,
    const int& rank_idx,
    const int& num_ag_ranks, const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const int& expected_num_tokens_per_rank,
    const std::optional<float>& activation_clamp_opt,
    // device int64 [num_layers] of pointers to float [E_local, L1 + L2 blocks]:
    // FP8 weights that keep their own 128x128 block scales give the kernel each
    // block's scale over its UE8M0 rounding, applied per K block. nullopt for
    // weights requantized to UE8M0 scales.
    const std::optional<torch::Tensor>& block_scale_residual_ptrs,
    const bool& fast_math,
    const int& num_sms,
    const int& num_prefetch_bytes,
    const std::optional<torch::Tensor>& debug_timings,
    const bool& compile_only,
    // P2P protocol; `layer_base` is the model layer the launch's first layer is
    // (the combine counters are indexed by the model layer's parity).
    const bool& p2p,
    const int& layer_base,
    // Early start (P2P): with at most early_max_late sources still missing, serve
    // the arrived ones in a first pass when the policy says so: 0 = once they are
    // still missing early_wait_cycles later; 1 = adaptive (a first pass of
    // early_pass_cycles + early_expert_cycles per expert, learned per lane, that
    // the late ones are expected to outlast by early_margin_pct). 0 = off.
    const int& early_max_late,
    const int& early_wait_cycles,
    const int& early_policy,
    const int& early_pass_cycles,
    const int& early_expert_cycles,
    const int& early_margin_pct,
    // Push dispatch (see mega_moe_m2n_ag): read the tokens from the staging rows
    const bool& push_dispatch
) {
    DG_HOST_ASSERT(static_cast<int>(sym_buffer_ptrs.size()) == num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(early_max_late >= 0 and early_wait_cycles >= 0);
    DG_HOST_ASSERT((early_policy == 0 or early_policy == 1) and early_pass_cycles >= 0 and
                   early_expert_cycles >= 0 and early_margin_pct >= 0);
    DG_HOST_ASSERT(early_max_late == 0 or (p2p and num_ag_ranks <= 24));
    DG_HOST_ASSERT(rank_idx >= num_ag_ranks and rank_idx < num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(layer_base >= 0);
    DG_HOST_ASSERT(weight_descs.is_cuda() and weight_descs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(weight_descs.numel() == static_cast<int64_t>(num_layers) * 4 * 128);
    DG_HOST_ASSERT(l1_weight_ptrs.is_cuda() and l1_weight_ptrs.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(l1_weight_ptrs.numel() == num_layers);
    if (block_scale_residual_ptrs.has_value()) {
        DG_HOST_ASSERT(use_fp8_weights);
        DG_HOST_ASSERT(block_scale_residual_ptrs->is_cuda() and
                       block_scale_residual_ptrs->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(block_scale_residual_ptrs->numel() == num_layers);
    }

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    const auto arch_major = device_runtime->get_arch_major();
    const auto num_experts_per_rank = num_experts_per_rank_in;
    const auto hidden = hidden_in;
    const auto intermediate_hidden = intermediate_hidden_in;
    DG_HOST_ASSERT(num_experts_per_rank * num_eg_ranks == num_experts);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }
    const auto [num_required_bytes, num_max_pool_tokens, slice] =
        get_symm_buffer_size_for_mega_moe_m2n(
            num_ag_ranks, num_eg_ranks, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, combine_rows_] = slice(sym_buffer);
    // Lane-1 views (dual-lane EG); fall back to lane-0 views when absent
    const auto& sym1 = sym_buffer_lane1.has_value() ? *sym_buffer_lane1 : sym_buffer;
    const auto [x1, x1_sf, topk_idx1, topk_weights1, l1_acts1, l1_acts_sf1, l2_acts1, l2_acts_sf1, combine_rows1_] = slice(sym1);
    const auto& sym2 = sym_buffer_lane2.has_value() ? *sym_buffer_lane2 : sym_buffer;
    const auto [x2, x2_sf, topk_idx2, topk_weights2, l1_acts2, l1_acts_sf2, l2_acts2, l2_acts_sf2, combine_rows2_] = slice(sym2);
    const auto& sym3 = sym_buffer_lane3.has_value() ? *sym_buffer_lane3 : sym_buffer;
    const auto [x3, x3_sf, topk_idx3, topk_weights3, l1_acts3, l1_acts_sf3, l2_acts3, l2_acts_sf3, combine_rows3_] = slice(sym3);
    const int num_lanes =
        sym_buffer_lane3.has_value() ? 4 :
        (sym_buffer_lane2.has_value() ? 3 :
        (sym_buffer_lane1.has_value() ? 2 : 1));
    const auto& ptrs1 = sym_buffer_lane1.has_value() ? sym_buffer_ptrs_lane1 : sym_buffer_ptrs;
    const auto& ptrs2 = sym_buffer_lane2.has_value() ? sym_buffer_ptrs_lane2 : sym_buffer_ptrs;
    const auto& ptrs3 = sym_buffer_lane3.has_value() ? sym_buffer_ptrs_lane3 : sym_buffer_ptrs;
    DG_HOST_ASSERT(static_cast<int>(ptrs1.size()) == num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(static_cast<int>(ptrs2.size()) == num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(static_cast<int>(ptrs3.size()) == num_ag_ranks + num_eg_ranks);

    if (arch_major == 10) {
        return sm100_mega_moe_m2n_eg(l1_acts, l1_acts_sf,
                                      l2_acts, l2_acts_sf,
                                      l1_acts1, l1_acts_sf1,
                                      l2_acts1, l2_acts_sf1,
                                      l1_acts2, l1_acts_sf2,
                                      l2_acts2, l2_acts_sf2,
                                      l1_acts3, l1_acts_sf3,
                                      l2_acts3, l2_acts_sf3,
                                      weight_descs, l1_weight_ptrs,
                                      block_scale_residual_ptrs,
                                      num_layers, l1_weights_nbytes,
                                      cumulative_local_expert_recv_stats,
                                      sym_buffer_ptrs,
                                      ptrs1,
                                      ptrs2,
                                      ptrs3,
                                      num_lanes,
                                      rank_idx,
                                      num_ag_ranks, num_eg_ranks,
                                      num_max_tokens_per_rank,
                                      num_experts_per_rank,
                                      expected_num_tokens_per_rank, num_topk,
                                      hidden, intermediate_hidden,
                                      activation_clamp, fast_math,
                                      use_fp8_weights,
                                      num_sms,
                                      num_prefetch_bytes,
                                      debug_timings,
                                      compile_only,
                                      p2p, layer_base,
                                      early_max_late, early_wait_cycles, early_policy,
                                      early_pass_cycles, early_expert_cycles, early_margin_pct,
                                      push_dispatch);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_symm_buffer_size_for_mega_moe_m2n", &get_symm_buffer_size_for_mega_moe_m2n);
    m.def("mega_moe_m2n_ag", &mega_moe_m2n_ag);
    m.def("mega_moe_m2n_eg", &mega_moe_m2n_eg);
    m.def("make_m2n_weight_descs", &make_m2n_weight_descs);
#endif
}

} // namespace deep_gemm::mega_m2n
