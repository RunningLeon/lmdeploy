# Copyright (c) OpenMMLab. All rights reserved.
import heapq
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from collections import defaultdict

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger, logging_timer

from ..config import CacheConfig
from .block_manager import BaseBlockManager


logger = get_logger('lmdeploy')

class Children:
    """Children class to make unfull nodes with same hashkey can reference the same parent node"""
    def __init__(self):
        self._children = defaultdict(set)
    
    def __getitem__(self, key: Any, *args, **kwargs) -> Optional[Union['Node', Set['Node']]]:
        if not key in self._children:
            raise AttributeError(f'{self} does not have key: {key}')
        res = self._children[key]
        num = len(res)
        if num == 1:
            res = list(res)[0]
        return res
    
    def __setitem__(self, key: Any, value: 'Node', *args, **kwargs):
        assert key == value.hash_key
        logger.debug(f'Add new {value} with key = {key} in {self._children}')
        self._children[key].add(value)

    def __contains__(self, key: Any)->bool:
        """Check if exists any node with key"""
        return key in self._children and len(self._children[key]) > 0
    
    def __len__(self):
        """Get number of all children nodes"""
        return sum([len(sub) for sub in self._children.values()])
    
    def pop(self, node: 'Node'):
        """Remove a node if exists"""
        if node is not None and node.hash_key in self._children:
            subset = self._children[node.hash_key]
            if node in subset:
                subset.remove(node)
            else:
                raise ValueError(f'Does not have {node} in {self._children}')
        else:
            raise ValueError(f'Does not have {node} in {self._children}')


class Leaves:
    """Leaves"""
    def __init__(self):
        self._leaves: Dict[int, Set['Node']] = defaultdict(set)
    
    def add(self, node: 'Node'):
        assert node.parent is not None 
        assert len(node.children) == 0
        self._leaves[node.block].add(node)

    def remove(self, node: 'Node'):
        """if a leaf if exists"""
        self._leaves[node.block].discard(node)

    def candidates(self, allocator) -> List['Node']:
        blocks = [block for block, subset in self._leaves.items() if len(subset) > 0]
        ref_cnt = allocator.get_ref_count(np.array(blocks))
        free_blocks = [(b, ref) for b, ref in zip(blocks, ref_cnt) if ref == len(self._leaves[b])]
        candidates = []
        deduce_blocks = []
        for block, ref in free_blocks:
            leaves = list(self._leaves[block])
            leaves.sort(key=lambda x: x.num_matched)
            last_node = leaves[0]
            # multiple unfull nodes in same block, try deduce ref of last node
            candidates.append(last_node)
            if ref > 1:
                for node in leaves[1:]:
                    # only deduce ref
                    deduce_blocks.append(block)
                    self._leaves[block].remove(node)
                    logger.debug(f'Remove duplicate node={node}')
                    node.parent = None
        
        if len(deduce_blocks) > 0:
            allocator.add_ref_count(np.array(deduce_blocks), -1)
        return candidates
    
    def __contains__(self, node: 'Node'):
        return node.block in self._leaves and node in self._leaves[node.block]
    
    def __len__(self):
        return sum([len(subset) for subset in self._leaves.values()])

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
        self.children: Children = Children()
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
            old_parent.children.pop(self)
        if val is not None:
            val.children[self.hash_key] = self
        self._parent = val

    def clone(self):
        """clone node."""
        ret = Node(self.hash_key, self.block, self.tokens, num_matched=self.num_matched, is_full=self.is_full, mm_hashes=self.mm_hashes)
        ret._parent = self._parent
        # need add the new child to the parent.children
        ret._parent.children[self.hash_key] = ret
        return ret

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True
    
    def __repr__(self):
        repr = f'Node(num_matched={self.num_matched}, is_full={self.is_full}, '
        repr += f'block={self.block}, tokens={self.tokens.tolist()}, '
        repr += f'hash_key={self.hash_key}, mm_hashes={self.mm_hashes}, '
        repr += f'children_num={len(self.children)}, is_root={self.parent is None})'
        return repr
    
    def __str__(self):
        return self.__repr__()


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
        self.leaves: Leaves = Leaves()

    def get_root(self, adapter_name: str):
        """get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, np.array([]))
        return self._roots[adapter_name]

    @property
    def roots(self):
        """all roots"""
        return self._roots.values()
    
    @logging_timer('BlockTrie_Match', logger)
    def match(self, seq: SchedulerSequence) -> Dict[int, int]:
        """match sequence and cache."""
        copy_map = {}
        if self.enable:
            if seq.history_multimodals.empty():
                self._match_text(seq)
            else:
                copy_map = self._match_multimodals(seq)
        return copy_map

    @logging_timer('BlockTrie_Allocate', logger)
    def allocate(self, seq: SchedulerSequence):
        """allocate."""
        if self.enable:
            if seq.history_multimodals.empty():
                self._allocate_text(seq)
            else:
                self._allocate_multimodals(seq)

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

    def _hash_tokens(self, tokens: np.ndarray, mm_hash_values=None):
        """hash func """
        hash_data = tuple(tokens)
        if mm_hash_values:
            hash_data = (hash_data, mm_hash_values)
        hash_key = hash(hash_data)
        return hash_key

    def _try_match(self, seq: SchedulerSequence, node: Node, start:int, end: int, mm_hash_values=None) -> Node:
        "try if seq[start:end] cant match a node"
        if node.mm_hashes != mm_hash_values:
            return None
        curr_tokens = seq.history_cache[start:end]
        is_full = len(curr_tokens) == self.block_size

        if not mm_hash_values and not is_full:
            return None
        hash_key = self._hash_tokens(curr_tokens, mm_hash_values)
        if hash_key in node.children and np.array_equal(curr_tokens, node.children[hash_key].tokens):
            matched_node = node.children[hash_key]
        return matched_node
    
    def _match_multimodals(self, seq: SchedulerSequence) -> Dict[int, int]:
        """match sequence and cache."""
        copy_map = {}
        if not self.enable:
            return copy_map

        block_size = self.block_size
        matched_blocks = []
        logical_blocks = seq.logical_blocks
        num_all_ids = seq.num_all_ids
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)

        if curr is None:
            curr = self.get_root(seq.adapter_name)

        prev_node = curr

        if not curr.is_full:
            assert curr.parent is not None, 'unfull block should not be root node'
            # if the unfull block doese not contain vision token
            if (curr.parent.num_matched + block_size) > num_all_ids:
                if not seq.history_multimodals.has_data(curr.num_matched, num_all_ids):
                    return copy_map
            # back to parent node for rebuilding
            curr = curr.parent
        elif (curr.num_matched + block_size) > num_all_ids:
            # if the unfull block doese not contain vision token
            mm_hash_values, _ = seq.history_multimodals.get_hash_values(curr.num_matched, num_all_ids)
            if not seq.history_multimodals.has_data(curr.num_matched, num_all_ids):
                return copy_map

        num_matched = curr.num_matched
        last_node = curr
        def __match_success(node: Node):
            """match full filled blocks or unfull blocks with multimodals"""
            nonlocal curr, num_matched, prev_node, last_node

            logger.debug(f'Matched token range=({num_matched}, {node.num_matched}), {node}')

            if not node.is_full:
                # when match an unfull block, need to copy to reuse the kv cache in that block
                block = self.allocator.allocate(1, device='gpu').item()
                logger.debug(
                    f'Create new copy block {node.block} -> {block} now free blocks={self.block_manager.get_num_free_gpu_blocks()}')
                copy_map[node.block] = block
                # directly create new unfull child, no need to match, multiple unfull nodes
                # with same hash_key can be added to parent.children
                new_node = node.clone()
                new_node.block = block
                node = new_node
            else:
                block = node.block
            matched_blocks.append(block)
            prev_node = curr
            curr = node
            num_matched += block_size
        
        matched_step = num_matched
        
        logger.debug(f'Matching seq-{seq.seq_id} all_token={num_all_ids} last node={curr}')

        while num_matched < num_all_ids:
            mm_hash_values, mm_multi_ends = seq.history_multimodals.get_hash_values(num_matched,
                                                                                    num_matched + block_size)
            if not mm_hash_values:
                # pure text
                matched_node = self._try_match(seq, curr, num_matched, num_matched + block_size)
            else:
                # filter last time processed ends
                mm_multi_ends = [key_end for key_end in mm_multi_ends if key_end[1] > prev_node.num_matched]
                # if full vision tokens without last vision token in the range
                # or the last token is vision token and is the end of the vision range
                if len(mm_multi_ends) == 0 or mm_multi_ends[-1][1] != (num_matched + block_size):
                    mm_multi_ends.append((mm_hash_values, num_matched + block_size))
                mm_multi_ends = [(key, end) for key, end in mm_multi_ends
                                 if prev_node.num_matched < end]
                mm_multi_ends.reverse()
                matched_node = None

                for cur_mm_hash_values, cur_matched_end in mm_multi_ends:
                    matched_node = self._try_match(seq, curr, num_matched, cur_matched_end, mm_hash_values=cur_mm_hash_values)
                    if matched_node is not None:
                        if (not matched_node.is_full) and self.block_manager.get_num_free_gpu_blocks() < 1:
                            logger.debug(f'No free gpu blocks for seq {seq.seq_id} total={self.block_manager.num_gpu_blocks}')
                            matched_node = None
                        break

            if matched_node is None:
                break
            # matched success
            matched_step = matched_node.num_matched
            __match_success(matched_node)
            if not matched_node.is_full:
                break

        if len(matched_blocks) > 0:
            # for vlm if matched step is in the middle of a vision segment, then match failed
            if seq.history_multimodals is not None and seq.history_multimodals.get_step(matched_step) != matched_step:
                logger.debug(f'matched half image for seq={seq} last node={curr}')
                if len(copy_map) > 0:
                    self.block_manager.allocator.free(np.array([b for b in copy_map.values()]))
                return {}
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(matched_step)

        seq.logical_blocks.last_shared_node = curr
        return copy_map

    def _allocate_multimodals(self, seq: SchedulerSequence):
        """allocate."""
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        num_all_ids = seq.num_all_ids
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)

        if curr is None:
            curr = self.get_root(seq.adapter_name)
        
        # prev_node is used to 
        prev_node = curr

        if not curr.is_full:
            # need add to leaves at first because will back to parent later
            self.leaves.add(curr)
            assert curr.parent is not None, 'unfull block should not be root node'
            # if the unfull block doese not contain vision token
            if (curr.parent.num_matched + block_size) > num_all_ids:
                if not seq.history_multimodals.has_data(curr.num_matched, num_all_ids):
                    return
            # back to parent node for rebuilding
            curr = curr.parent
        elif (curr.num_matched + block_size) > num_all_ids:
            # if the unfull block doese not contain vision token
            if not seq.history_multimodals.has_data(curr.num_matched, num_all_ids):
                return

        num_matched = curr.num_matched

        logger.debug(f'Allocate seq-{seq.seq_id} {num_matched}/{num_all_ids} curr={curr}')

        # remove if is leaf node
        self.leaves.remove(curr)

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
                hash_key = self._hash_tokens(curr_tokens, cur_mm_hash_values)
                block = logical_blocks[block_id]
                # directly create new unfull child, no need to match, multiple unfull nodes
                # with same hash_key can be added to parent.children
                child = Node(hash_key=hash_key,
                                block=block,
                                tokens=curr_tokens,
                                num_matched=cur_matched_end,
                                is_full=False,
                                mm_hashes=cur_mm_hash_values)
                child.parent = parent
                logger.debug(f'allocate num_matched={num_matched} add [unfull] node={child}')
                blocks.append(child.block)
                unfull_nodes.append(child)

            return unfull_nodes

        def __add_full_node(node, mm_hash_values=None):
            # add a node of a full-filled block
            nonlocal free_blocks, blocks

            curr_tokens = seq.history_cache[num_matched:num_matched + block_size]
            is_full = len(curr_tokens) == block_size
            assert is_full
            hash_key = self._hash_tokens(curr_tokens, mm_hash_values)
            block = logical_blocks[block_id]
            parent = node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if child.mm_hashes != mm_hash_values or not np.array_equal(curr_tokens, child.tokens):
                    logger.debug(f'Hash collision for seq={seq} with node={child}')
                    # hash collision
                    return node, False
                node = child
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
                logger.debug(f'allocate num_matched={num_matched}  reuse [full] node={node}')
            else:
                node = Node(hash_key=hash_key,
                            block=block,
                            tokens=curr_tokens,
                            num_matched=num_matched + block_size,
                            is_full=is_full,
                            mm_hashes=mm_hash_values)
                node.parent = parent
                logger.debug(f'allocate num_matched={num_matched}  add [full] node={node}')
                
            blocks.append(node.block)
            return node, True

        last_node = curr
        unfull_nodes = []

        while num_matched < num_all_ids:
            mm_hash_values, mm_multi_ends = seq.history_multimodals.get_hash_values(num_matched,
                                                                                    num_matched + block_size)
            full_mm_hash_values = None
            if mm_hash_values:
                mm_multi_ends = [key_end for key_end in mm_multi_ends if key_end[1] > prev_node.num_matched] 
                if len(mm_multi_ends) == 0:
                    # it's a full block with all tokens are vision tokens and no vision end token in this block
                    full_mm_hash_values = mm_hash_values
                elif mm_multi_ends[-1][1] == (num_matched + block_size):
                    # the last token is vision token and is vision end token
                    full_mm_hash_values, _ = mm_multi_ends.pop(-1)
                
                if len(mm_multi_ends) > 0:
                    cur_unfull_nodes = __add_unfull_nodes(curr, mm_multi_ends)
                    if len(cur_unfull_nodes) > 0:
                        last_node = cur_unfull_nodes[-1]
                        unfull_nodes += cur_unfull_nodes

            if num_matched + block_size <= num_all_ids:
                prev_node = curr
                curr, success = __add_full_node(curr, full_mm_hash_values)
                last_node = curr
                if not success:
                    break

            num_matched += block_size
            block_id += 1

        if last_node.num_matched > logical_blocks.last_shared_node.num_matched:
            logical_blocks.last_shared_node = last_node

        # add leaf nodes
        for cur_node in (unfull_nodes + [last_node]):
            if len(cur_node.children) == 0 and cur_node not in self.roots:
                # ignore root
                self.leaves.add(cur_node)

        if len(blocks) > 0:
            blocks = np.array(blocks)
            self.allocator.update_access_time(blocks)
            self.allocator.add_ref_count(blocks, 1)

        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

    @logging_timer('BlockTrie_Evict', logger)
    def evict(self, max_num_blocks: int):
        """evict."""
        if not self.enable or len(self.leaves) == 0:
            return 0
        logger.debug(f'Need to evict max_num_blocks={max_num_blocks}')

        def __remove_leaf(leaves, evicted_blocks):
            _, leaf = heapq.heappop(leaves)
            evicted_blocks.append(leaf.block)
            parent = leaf.parent
            leaf.parent = None
            self.leaves.remove(leaf)
            logger.debug(f'Evict {leaf}')
            return parent, leaf

        def __add_leaf(leaves, parent):
            self.leaves.add(parent)
            if self.allocator.get_ref_count(parent.block) == 1:
                access_time = self.allocator.get_access_time(parent.block)
                logger.debug(f'Evict heappush {parent}')
                heapq.heappush(leaves, (access_time, parent))

        leaves = self.leaves.candidates(self.allocator)
        if len(leaves) == 0:
            return 0
        
        # make heap
        leave_blocks = np.array([leaf.block for leaf in leaves])
        access_times = self.allocator.get_access_time(leave_blocks)
        leaves = list(zip(access_times, leaves))
        heapq.heapify(leaves)

        evicted_blocks = []
        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            parent, removed_leaf = __remove_leaf(leaves, evicted_blocks)
            if parent in self.roots:
                continue
            # remove nodes of with same mm_hashes
            if removed_leaf.mm_hashes:
                while removed_leaf.mm_hashes == parent.mm_hashes and len(
                        parent.children) == 0 and self.allocator.get_ref_count(parent.block) == 1:
                    tmp_parent = parent.parent
                    evicted_blocks.append(parent.block)
                    parent.parent = None
                    logger.debug(f'Evict multimodal node={parent}')
                    parent = tmp_parent
                    logger.debug(f'Neext multimodal node={parent}')

            if len(parent.children) == 0 and not parent in self.roots:
                __add_leaf(leaves, parent)

        self.allocator.free(np.array(evicted_blocks))
        logger.debug(f'Evict final blocks={evicted_blocks} refs={self.allocator.get_ref_count(np.array(evicted_blocks))}')
        return len(evicted_blocks)
