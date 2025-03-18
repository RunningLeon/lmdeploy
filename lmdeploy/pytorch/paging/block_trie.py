# Copyright (c) OpenMMLab. All rights reserved.
import heapq
from typing import Dict, List, Set, Tuple

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence

from ..config import CacheConfig
from .block_manager import BaseBlockManager


class Node:
    """node of block trie."""

    def __init__(self,
                 hash_key: int,
                 block: int,
                 tokens: np.ndarray,
                 num_matched: int = 0,
                 is_full: bool = True,
                 mm_hashes: Tuple[str] = None):
        self.hash_key = hash_key
        self.block = block
        self.tokens = tokens
        self.num_matched = num_matched
        self.children: Dict[int, 'Node'] = dict()
        self._parent: 'Node' = None
        self.is_full = is_full
        self.mm_hashes = mm_hashes

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val: 'Node'):
        old_parent = self._parent
        if old_parent is not None:
            old_parent.children.pop(self.hash_key)
        if val is not None:
            val.children[self.hash_key] = self
        self._parent = val

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True


class BlockTrie:
    """block trie for prefix caching."""

    def __init__(self, cache_config: CacheConfig, block_manager: BaseBlockManager):
        self.block_manager = block_manager
        self.cache_config = cache_config
        self.allocator = self.block_manager.allocator
        self.block_size = cache_config.block_size
        self.enable = self.cache_config.enable_prefix_caching

        # caches with different adapter should not be shared.
        self._roots: Dict[str, Node] = dict()
        self.leaves: Set[Node] = set()

    def get_root(self, adapter_name: str):
        """get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None)
        return self._roots[adapter_name]

    def match(self, seq: SchedulerSequence) -> Dict[int, int]:
        """match sequence and cache."""
        copy_map = {}
        if self.enable:
            if seq.history_multimodals.empty():
                self._match_text(seq)
            else:
                copy_map = self._match_multimodals(seq)
        return copy_map

    def allocate(self, seq: SchedulerSequence) -> Dict[int, int]:
        """allocate."""
        copy_map = {}
        if self.enable:
            if seq.history_multimodals.empty():
                self._allocate_text(seq)
            else:
                copy_map = self._allocate_multimodals(seq)
        return copy_map

    def _match_text(self, seq: SchedulerSequence):
        """match sequence and cache."""
        if not self.enable:
            return

        block_size = self.block_size
        matched_blocks = []

        logical_blocks = seq.logical_blocks
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)
        if curr is None:
            curr = self.get_root(seq.adapter_name)
        num_matched = curr.num_matched

        def __match_success(node: Node):
            nonlocal curr, num_matched
            matched_blocks.append(node.block)
            curr = node
            num_matched += block_size

        while num_matched + block_size < seq.num_all_ids:
            curr_tokens = seq.history_cache[num_matched:num_matched + block_size]

            key = hash(tuple(curr_tokens))
            if key not in curr.children:
                break

            child = curr.children[key]
            if not np.array_equal(curr_tokens, child.tokens):
                break

            __match_success(child)

        if len(matched_blocks) > 0:
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(num_matched)

        seq.logical_blocks.last_shared_node = curr

    def _allocate_text(self, seq: SchedulerSequence):
        """allocate."""
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = getattr(logical_blocks, 'last_shared_node', None)
        if node is None:
            node = self.get_root(seq.adapter_name)
            logical_blocks.last_shared_node = node

        num_matched = node.num_matched
        num_all_ids = seq.num_all_ids

        if num_matched + block_size > num_all_ids:
            return

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.remove(node)

        block_id = num_matched // block_size
        blocks = []
        free_blocks = []
        while num_matched + block_size <= num_all_ids:
            curr_tokens = seq.history_cache[num_matched:num_matched + block_size]

            block = logical_blocks[block_id]

            hash_key = hash(tuple(curr_tokens))
            parent = node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if not np.array_equal(curr_tokens, child.tokens):
                    break
                node = child
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
            else:
                node = Node(hash_key=hash_key, block=block, tokens=curr_tokens, num_matched=num_matched + block_size)
                node.parent = parent
            blocks.append(node.block)
            num_matched += block_size
            block_id += 1

        logical_blocks.last_shared_node = node
        if node.parent is not None and len(node.children) == 0:
            # ignore root
            self.leaves.add(node)
        if len(blocks) > 0:
            self.allocator.add_ref_count(np.array(blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

    def _match_multimodals(self, seq: SchedulerSequence) -> Dict[int, int]:
        """match sequence and cache."""
        copy_map = {}
        if not self.enable:
            return copy_map

        block_size = self.block_size
        matched_blocks = []
        logical_blocks = seq.logical_blocks
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)

        last_max_num_matched = 0
        if curr is None:
            curr = self.get_root(seq.adapter_name)
        elif not curr.is_full:
            if curr.parent is None:
                curr.parent = self.get_root(seq.adapter_name)
            # if there is a full block or the rest blocks contain vision tokens
            if (curr.parent.num_matched + block_size) > seq.num_all_ids:
                mm_hash_values, _ = seq.history_multimodals.get_hash_values(curr.num_matched, seq.num_all_ids)
                if not mm_hash_values:
                    return copy_map

            last_max_num_matched = curr.num_matched
            curr = curr.parent
        else:
            last_max_num_matched = curr.num_matched

        num_matched = curr.num_matched
        print(f'>>> match seq {seq.seq_id} {num_matched} {seq.num_all_ids}')

        def __match_success(node: Node):
            nonlocal curr, num_matched, copy_map
            print(
                f'matched success: seq_id {seq.seq_id} range=({num_matched}, {node.num_matched}), block={node.block} is_full={node.is_full}'
            )

            if not node.is_full:
                # when match an unfull block, need to copy to reuse the kv cache in that block
                block = self.allocator.allocate(1, device='gpu').item()
                print(
                    f'matched create new copy block {node.block} -> {block} now free blocks={self.block_manager.get_num_free_gpu_blocks()}'
                )
                copy_map[node.block] = block
            else:
                block = node.block
            matched_blocks.append(block)
            curr = node
            num_matched += block_size

        matched_step = num_matched

        while num_matched < seq.num_all_ids:
            mm_hash_values, mm_multi_ends = seq.history_multimodals.get_hash_values(num_matched,
                                                                                    num_matched + block_size)
            if not mm_hash_values:
                mm_multi_ends = [(None, num_matched + block_size)]
            else:
                # if full vision tokens without last vision token in the range
                # or the last token is vision token and is the end of the vision range
                if len(mm_multi_ends) == 0 or mm_multi_ends[-1][1] != (num_matched + block_size):
                    mm_multi_ends.append((mm_hash_values, num_matched + block_size))
                mm_multi_ends = [(key, end) for key, end in mm_multi_ends
                                 if last_max_num_matched < end <= seq.num_all_ids]
                mm_multi_ends.reverse()

            matched_node = None
            for cur_mm_hash_values, cur_matched_end in mm_multi_ends:
                curr_tokens = seq.history_cache[num_matched:cur_matched_end]
                is_full = len(curr_tokens) == block_size
                if (not is_full) and self.block_manager.get_num_free_gpu_blocks() < 1:
                    print(f'>>> no free gpu blocks seq {seq.seq_id}')
                    continue
                hash_data = tuple(curr_tokens)
                if cur_mm_hash_values:
                    hash_data = (hash_data, cur_mm_hash_values)
                elif not is_full:
                    continue
                hash_key = hash(hash_data)
                if hash_key in curr.children and np.array_equal(curr_tokens, curr.children[hash_key].tokens):
                    matched_node = curr.children[hash_key]
                    break

            if matched_node is not None:
                matched_step = matched_node.num_matched
                __match_success(matched_node)

            if matched_node is None or not matched_node.is_full:
                break

        if len(matched_blocks) > 0:
            # for vlm if matched step is in the middle of a vision segment, then match failed
            if seq.history_multimodals is not None and seq.history_multimodals.get_step(matched_step) != matched_step:
                return {}

            add_ref_blocks = matched_blocks
            if len(copy_map):
                add_ref_blocks = [b for b in add_ref_blocks if b not in copy_map.values()]
            add_ref_blocks = np.array(add_ref_blocks)
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            if len(add_ref_blocks) > 0:
                self.allocator.add_ref_count(add_ref_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(matched_step)

        seq.logical_blocks.last_shared_node = curr
        return copy_map

    def _allocate_multimodals(self, seq: SchedulerSequence) -> Dict[int, int]:
        """allocate."""
        copy_map = {}
        if not self.enable:
            return copy_map

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = getattr(logical_blocks, 'last_shared_node', None)

        last_max_num_matched = 0
        if node is None:
            node = self.get_root(seq.adapter_name)
            logical_blocks.last_shared_node = node
        elif not node.is_full:
            if node.parent is None:
                node.parent = self.get_root(seq.adapter_name)
            # if there is a full block or the rest blocks contain vision tokens
            if (node.parent.num_matched + block_size) > seq.num_all_ids:
                mm_hash_values, _ = seq.history_multimodals.get_hash_values(node.num_matched, seq.num_all_ids)
                if not mm_hash_values:
                    return copy_map

            # back to parent node for un-full node
            last_max_num_matched = node.num_matched
            node = node.parent
        else:
            last_max_num_matched = node.num_matched

        num_matched = node.num_matched
        num_all_ids = seq.num_all_ids

        print(f'>>> allocate seq {seq.seq_id} {num_matched} {seq.num_all_ids}')

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.remove(node)

        block_id = num_matched // block_size

        blocks = []
        free_blocks = []

        def __add_unfull_nodes(parent: Node, multi_segments: List[Tuple[Tuple[str], int]]):
            # add multiple nodes of un-full blocks
            nonlocal free_blocks, blocks

            unfull_nodes = []
            for cur_mm_hash_values, cur_matched_end in multi_segments:
                assert cur_mm_hash_values, 'only support multimodal unfull'
                curr_tokens = seq.history_cache[num_matched:cur_matched_end]
                is_full = len(curr_tokens) == block_size
                assert not is_full
                hash_data = (tuple(curr_tokens), cur_mm_hash_values)
                hash_key = hash(hash_data)
                block = logical_blocks[block_id]
                if hash_key in parent.children:
                    child = parent.children[hash_key]
                    if child.mm_hashes == cur_mm_hash_values and np.array_equal(curr_tokens,
                                                                                child.tokens) and block != child.block:
                        copy_map[child.block] = block
                        unfull_nodes.append(child)
                        print(
                            f'allocate seq {seq.seq_id } num_matched={num_matched} reuse a unfull node block={block} ')
                else:
                    child = Node(hash_key=hash_key,
                                 block=block,
                                 tokens=curr_tokens,
                                 num_matched=cur_matched_end,
                                 is_full=is_full,
                                 mm_hashes=cur_mm_hash_values)
                    print(f'allocate seq {seq.seq_id } num_matched={num_matched} add a unfull node block={block} ')
                    child.parent = parent
                    blocks.append(child.block)
                    unfull_nodes.append(child)

            return unfull_nodes

        def __add_full_node(node, mm_hash_values):
            # add a node of a full-filled block
            nonlocal free_blocks, blocks

            curr_tokens = seq.history_cache[num_matched:num_matched + block_size]
            is_full = len(curr_tokens) == block_size
            assert is_full
            hash_data = tuple(curr_tokens)
            if mm_hash_values:
                hash_data = (hash_data, mm_hash_values)
            hash_key = hash(hash_data)
            block = logical_blocks[block_id]
            parent = node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if child.mm_hashes != mm_hash_values or not np.array_equal(curr_tokens, child.tokens):
                    # hash collision
                    return node, False
                node = child
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
                print(f'allocate seq {seq.seq_id } num_matched={num_matched}  reuse [full] node block={node.block}')
            else:
                node = Node(hash_key=hash_key,
                            block=block,
                            tokens=curr_tokens,
                            num_matched=num_matched + block_size,
                            is_full=is_full,
                            mm_hashes=mm_hash_values)
                print(f'allocate seq {seq.seq_id } num_matched={num_matched}  add [full] node block={block}')
                node.parent = parent
            blocks.append(node.block)
            return node, True

        last_node = node
        unfull_nodes = []
        while num_matched < num_all_ids:
            mm_hash_values, mm_multi_ends = seq.history_multimodals.get_hash_values(num_matched,
                                                                                    num_matched + block_size)
            full_mm_hash_values = None
            if mm_hash_values:
                if len(mm_multi_ends) == 0:
                    # it's a full block with all tokens are vision tokens and no vision end token in this block
                    full_mm_hash_values = mm_hash_values
                elif mm_multi_ends[-1][1] == (num_matched + block_size):
                    # the last token is vision token and is vision end token
                    full_mm_hash_values, _ = mm_multi_ends.pop(-1)
                mm_multi_ends = [(key, end) for key, end in mm_multi_ends if last_max_num_matched < end <= num_all_ids]
                if len(mm_multi_ends) > 0:
                    cur_unfull_nodes = __add_unfull_nodes(node, mm_multi_ends)
                    if len(cur_unfull_nodes) > 0:
                        last_node = cur_unfull_nodes[-1]
                        unfull_nodes += cur_unfull_nodes

            if num_matched + block_size <= num_all_ids:
                node, success = __add_full_node(node, full_mm_hash_values)
                last_node = node
                if not success:
                    break

            num_matched += block_size
            block_id += 1

        if last_node.num_matched > logical_blocks.last_shared_node.num_matched:
            logical_blocks.last_shared_node = last_node

        # add leaf nodes
        for cur_node in (unfull_nodes + [last_node]):
            if cur_node.parent is not None and len(cur_node.children) == 0:
                # ignore root
                self.leaves.add(cur_node)

        if len(blocks) > 0:
            self.allocator.add_ref_count(np.array(blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

        return copy_map

    def evict(self, max_num_blocks: int):
        """evict."""
        if not self.enable:
            return 0

        def __remove_leaf(leaves, evicted_blocks):
            _, leaf = heapq.heappop(leaves)
            evicted_blocks.append(leaf.block)
            parent = leaf.parent
            leaf.parent = None
            self.leaves.remove(leaf)
            return parent, leaf

        def __add_leaf(leaves, parent):
            self.leaves.add(parent)
            if self.allocator.get_ref_count(parent.block) == 1:
                access_time = self.allocator.get_access_time(parent.block)
                heapq.heappush(leaves, (access_time, parent))

        evicted_blocks = []
        leaves = list(self.leaves)

        # filter ref-cnt == 1 (trie own one block ref)
        leave_blocks = np.array(list(leaf.block for leaf in leaves))
        ref_cnt = self.allocator.get_ref_count(leave_blocks)
        indices = (ref_cnt == 1).nonzero()[0]
        if len(indices) == 0:
            return 0

        # make heap
        leaves = list(leaves[i] for i in indices)
        access_times = self.allocator.get_access_time(leave_blocks)
        access_times = list(access_times[i] for i in indices)
        leaves = list(zip(access_times, leaves))
        heapq.heapify(leaves)

        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            parent, removed_leaf = __remove_leaf(leaves, evicted_blocks)
            if parent.parent is None:
                # ignore root
                continue
            # remove nodes of with same mm_hashes
            if removed_leaf.mm_hashes:
                while removed_leaf.mm_hashes == parent.mm_hashes and len(
                        parent.children) == 0 and self.allocator.get_ref_count(parent.block) == 1:
                    tmp_parent = parent.parent
                    evicted_blocks.append(parent.block)
                    parent.parent = None
                    parent = tmp_parent

            if len(parent.children) == 0:
                __add_leaf(leaves, parent)

        self.allocator.free(np.array(evicted_blocks))

        return len(evicted_blocks)
