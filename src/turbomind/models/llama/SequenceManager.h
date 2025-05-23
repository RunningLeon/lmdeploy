// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>

#include "src/turbomind/core/allocator.h"

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/BlockTrie.h"

namespace turbomind {

struct Sequence {

    enum Status
    {
        kCached = 0,
        kLocked,
        kActive
    };

    uint64_t id;
    Status   status = kCached;

    BlockIds  blocks;
    UniqueIds block_unique_ids;

    int input_length = 0;

    mutable std::vector<int> prompt;

    mutable std::vector<int> tokens;  // update by user

    mutable int cache_len = 0;

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    mutable float rope_theta = 0.f;

    // embedding data
    mutable std::vector<std::vector<std::byte>> input_embeddings;
    mutable std::vector<std::pair<int, int>>    input_embedding_ranges;

    explicit Sequence(uint64_t _id): id(_id) {}

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

using Sequences = std::vector<const Sequence*>;

inline std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << seq.status << ", token_count=" << seq.tokens.size()
       << ", block_count=" << seq.blocks.size() << ", cache_len=" << seq.cache_len
       << ", random_state_size=" << seq.random_state.size();
    return os;
}

class SequenceManager {
public:
    // clang-format off
    struct BlockConfig {
        int head_dim_;
        int head_num_;
        int block_len_;
        int t_bits_;
        int q_bits_;
        int t_bits() const { return t_bits_; }
        int q_bits() const { return q_bits_; }
        int head_dim() const { return head_dim_; }
        int head_num() const { return head_num_; }
        int block_len() const { return block_len_; }
    };
    // clang-format on

    explicit SequenceManager(size_t             layer_num,
                             const BlockConfig& block_config,
                             double             block_count,
                             int                chunk_size,
                             bool               enable_prefix_caching,
                             int                rank,
                             core::Allocator    allocator,
                             GetFreeMemSize     get_free_size);

    SequenceManager(const SequenceManager&)     = delete;
    SequenceManager(SequenceManager&&) noexcept = default;

    [[nodiscard]] const Sequence* Create(uint64_t id);

    [[nodiscard]] const Sequence* Get(uint64_t id);

    [[nodiscard]] bool Contains(uint64_t id);

    [[nodiscard]] bool Erase(uint64_t id);

    void UpdateAndSetUnlock(const Sequence& seq);

    struct Outcome {
        int allocation;
        int swap_in;
        int swap_out;
    };

    using AdjustInputCount = std::function<int(const Sequences&, const std::vector<int>&)>;

    [[nodiscard]] Outcome Materialize(Sequences                    sequences,
                                      std::vector<int>             context_lengths,
                                      const std::vector<uint64_t>& priorities,
                                      int                          step_length,
                                      AdjustInputCount             adjust);

    void CacheIfEnabled(const Sequences& sequences, int active_size);

    [[nodiscard]] void* GetBlockPtr(int block_id)
    {
        return block_manager_->block(block_id).data;
    }

    int max_block_count() const noexcept
    {
        return block_manager_->max_block_count();
    }

private:
    void Erase(std::map<uint64_t, Sequence>::iterator& it);

    void CommitUnlockAndFree();

    void VerifyAndLockCached(const Sequences& sequences);

    std::vector<int> CountRequiredBlocks(const Sequences&        sequences,  //
                                         const std::vector<int>& context_lengths,
                                         int                     step_length);

    static void SortByPriority(Sequences&                   sequences,  //
                               std::vector<int>&            context_lengths,
                               const std::vector<uint64_t>& priorities);

    static void AssignAndActivate(const Sequences&        sequences,  //
                                  const std::vector<int>& counts,
                                  const BlockIds&         blocks,
                                  const UniqueIds&        unique_ids);

private:
    int block_seq_len_;
    int rank_;

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::shared_ptr<BlockManager> block_manager_;
    std::shared_ptr<BlockTrie>    block_trie_;

    BlockIds unlocked_;
    BlockIds freed_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
