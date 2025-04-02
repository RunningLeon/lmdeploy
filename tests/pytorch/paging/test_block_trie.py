import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.paging.block_manager import build_block_manager
from lmdeploy.pytorch.paging.block_trie import build_blocktrie, NodeType, BlockTrie, BlockTrieVLM


class TestBlockTrie:

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 16

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(max_batches=256,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          enable_prefix_caching=True)

    @pytest.fixture
    def block_mgr(self, cache_config):
        yield build_block_manager(cache_config)

    @pytest.fixture
    def block_trie(self, cache_config, block_mgr):
        yield BlockTrie(cache_config, block_mgr)

    def test_allocate(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)

        # first allocate
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 3
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 2
        assert np.array_equal(node.tokens, [2] * block_size)
        assert np.array_equal(node.parent.tokens, [1] * block_size)
        assert node in block_trie.leaves
        assert node.parent not in block_trie.leaves

        # append
        seq.update_token_ids([4] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 3
        expect_tokens = [3] * (block_size // 2) + [4] * (block_size // 2)
        assert np.array_equal(node.tokens, expect_tokens)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1

    def test_match(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)

        # initialize cache
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)

        # test1
        token_ids = ([1] * block_size + [3] * block_size)
        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 1
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size
        assert np.array_equal(node.tokens, [1] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert len(block_trie.leaves) == 2

        # test2
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [4] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_trie.match(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 2
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [4, 3])

    def test_evict(self, block_trie, block_size, num_gpu_blocks):
        block_mgr = block_trie.block_manager
        sess = SchedulerSession(0, block_size)
        token_ids = ([1] * block_size * (num_gpu_blocks - 1))
        token_ids += [2] * (block_size // 2)
        seq = sess.add_sequence(token_ids)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert block_mgr.get_num_free_gpu_blocks() == 0

        # test free
        block_mgr.free(seq)
        seq.set_step(0)
        assert block_mgr.get_num_free_gpu_blocks() == 1

        # test evict
        leaf = next(iter(block_trie.leaves))
        block_trie.evict(4)
        new_leaf = next(iter(block_trie.leaves))
        assert leaf != new_leaf
        assert block_mgr.get_num_free_gpu_blocks() == 5


class TestBlockTrieVLM:
    @pytest.fixture
    def image_token_id(self):
        yield 0
        
    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 16

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(max_batches=256,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          enable_prefix_caching=True)

    @pytest.fixture
    def block_mgr(self, cache_config):
        yield build_block_manager(cache_config)

    @pytest.fixture
    def block_trie(self, cache_config, block_mgr):
        yield BlockTrieVLM(cache_config, block_mgr)

    @pytest.mark.test
    def test_build_blocktrie(self, cache_config, block_mgr):
        task_type = 'vlm'
        obj = build_blocktrie(cache_config, block_mgr, task_type=task_type)
        assert type(obj) is BlockTrieVLM
        task_type = 'llm'
        obj = build_blocktrie(cache_config, block_mgr, task_type=task_type)
        assert type(obj) is BlockTrie
        
    @pytest.mark.test
    def test_allocate(self, block_trie, block_mgr, block_size, image_token_id):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        half_block_size = block_size // 2
        # test case 1 single block
        token_ids = [1] * half_block_size + [image_token_id] * 2 * block_size
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2*block_size+half_block_size, meta=dict(hash_value='image_0')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 3
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 2])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.node_type == NodeType.UNFULL
        assert node.mm_hashes == tuple(['image_0'])
        assert node.num_matched == 2 * block_size + half_block_size
        assert np.array_equal(node.tokens, [image_token_id] * half_block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 1
        assert node.parent not in block_trie.leaves
        assert node.parent.node_type == NodeType.FULL
        assert node.parent.mm_hashes == tuple(['image_0'])
        assert node.parent.num_matched == 2 * block_size
        assert np.array_equal(node.parent.tokens, [image_token_id] * block_size)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)
        
        # append text token to make last block full
        seq.update_token_ids([2] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.node_type == NodeType.FULL
        assert node.mm_hashes == tuple(['image_0'])
        assert node.num_matched == 3 * block_size
        assert np.array_equal(node.tokens, [image_token_id] * half_block_size + [2] * half_block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 2
        assert allocator.get_ref_count(node.block) == 3
        assert node.parent is not None
        assert node.parent.node_type == NodeType.FULL
        assert node.parent.mm_hashes == tuple(['image_0'])
        assert node.parent.num_matched == 2 * block_size
        assert np.array_equal(node.parent.tokens, [image_token_id] * block_size)
        block_mgr.free(seq)
        assert len(block_trie.leaves) == 2
        assert allocator.get_ref_count(node.block) == 2
        block_trie.evict(3)
        assert len(block_trie.leaves) == 0
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks
        
        # test multi images
        quarter_block_size = block_size // 4
        token_ids = [1] * half_block_size # text
        token_ids += [image_token_id] * 2 * block_size # img0
        token_ids += [2] * quarter_block_size # text
        token_ids += [image_token_id] * block_size # img1
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=half_block_size, end=2*block_size + half_block_size, meta=dict(hash_value='image_0')),
            MultiModalTensor(
                data=None, start=3*block_size - quarter_block_size, end=4*block_size-quarter_block_size, meta=dict(hash_value='image_1')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 2])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_1'])
        assert node.num_matched == 4 * block_size - quarter_block_size
        expect_tokens = [image_token_id] * quarter_block_size * 3
        assert np.array_equal(node.tokens, expect_tokens)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 2
        assert node.parent not in block_trie.leaves
        assert node.parent.mm_hashes == tuple(['image_0', 'image_1'])
        assert node.parent.num_matched == 3 * block_size
        expect_tokens = [image_token_id] * half_block_size + [2] * quarter_block_size + [image_token_id] * quarter_block_size
        assert np.array_equal(node.parent.tokens, expect_tokens)
        assert node.parent.parent is not None
        assert node.parent.parent.mm_hashes == tuple(['image_0'])

        # append text
        # append text token to make last block full
        seq.update_token_ids([3] * block_size * 2)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 6
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 3, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes is None
        assert node.num_matched == 5 * block_size
        assert node.parent.mm_hashes == tuple(['image_1'])
        assert node.parent.num_matched == 4 * block_size
        assert len(block_trie.leaves) == 3
        blocks = seq.logical_blocks.get_real_blocks()
        block_mgr.free(seq)
        ref_cnt = allocator.get_ref_count(blocks)
        assert np.array_equal(ref_cnt, [1, 1, 2, 2, 1, 0])
        block_trie.evict(5)
        assert len(block_trie.leaves) == 0

    @pytest.mark.test
    def test_match(self, block_trie, block_mgr, block_size, image_token_id):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        half_block_size = block_size // 2
        quarter_block_size = block_size // 4

        # initialize cache with single image

        token_ids = [1] * half_block_size  # text
        token_ids += [image_token_id] * 2 * block_size  # img0
        token_ids += [2] * quarter_block_size  # text
        img0 = MultiModalTensor(data=None, start=half_block_size, end=2 * block_size + half_block_size, meta=dict(hash_value='image_0'))
        seq0 = sess.add_sequence(token_ids, multimodals=dict(image=[img0]))

        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # test with same image, but diff suffix text
        token_ids = [1] * half_block_size  # text
        token_ids += [image_token_id] * 2 * block_size  # img0
        token_ids += [3] * block_size  # text
        seq_prob = sess.add_sequence(token_ids, multimodals=dict(image=[img0]))

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert last_node.node_type == NodeType.COPY
        assert last_node.num_matched == 2 * block_size + half_block_size
        assert last_node.mm_hashes == tuple(['image_0'])
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3])
        assert allocator.get_ref_count(last_node.block) == 2
        block_mgr.allocate(seq_prob)
        src_block = last_node.block
        copy_map = block_trie.update_copy_map([seq_prob], dict())
        assert copy_map == {src_block:last_node.block}
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 1, 1])
        block_mgr.free(seq_prob)

        # append another image 1
        img1 = MultiModalTensor(data=None, start=0, end=block_size, meta=dict(hash_value='image_1'))
        seq0.update_token_ids(token_ids=[image_token_id] * block_size, multimodals=dict(image=[img1]))
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # test with two images
        token_ids = [1] * half_block_size  # text
        token_ids += [image_token_id] * 2 * block_size  # img0
        token_ids += [2] * quarter_block_size  # text
        token_ids += [image_token_id] * block_size  # img1
        token_ids += [4] * block_size # text
        img1 = MultiModalTensor(data=None, start=3 * block_size - quarter_block_size, end=4 * block_size - quarter_block_size, meta=dict(hash_value='image_1'))
        seq_prob = sess.add_sequence(token_ids, multimodals=dict(image=[img0, img1]))

        block_trie.match(seq_prob)
        last_node = getattr(seq_prob.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert last_node.node_type == NodeType.COPY
        assert last_node.num_matched == 4 * block_size - quarter_block_size
        assert last_node.mm_hashes == tuple(['image_1'])
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 4])
        assert allocator.get_ref_count(last_node.block) == 2
        block_mgr.allocate(seq_prob)
        src_block = last_node.block
        copy_map = block_trie.update_copy_map([seq_prob], dict())
        assert copy_map == {src_block:last_node.block}
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 4, 1, 1])
        block_trie.allocate(seq_prob)
        ref_cnt = allocator.get_ref_count(seq_prob.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 3, 4, 2, 1])
        block_mgr.free(seq_prob)
        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 2])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 5)
        block_trie.evict(1)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)
        block_trie.evict(4)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)

    @pytest.mark.test
    def test_evict(self, block_trie, block_mgr, block_size, image_token_id):
        
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        half_block_size = block_size // 2
        quarter_block_size = block_size // 4

        # initialize cache with single image

        token_ids = [1] * half_block_size  # text
        token_ids += [image_token_id] * 2 * block_size  # img0
        token_ids += [2] * quarter_block_size  # text
        img0 = MultiModalTensor(data=None, start=half_block_size, end=2 * block_size + half_block_size, meta=dict(hash_value='image_0'))
        seq0 = sess.add_sequence(token_ids, multimodals=dict(image=[img0]))

        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        # append another image 1
        img1 = MultiModalTensor(data=None, start=0, end=block_size, meta=dict(hash_value='image_1'))
        token_ids = [image_token_id] * block_size + [3] * block_size
        seq0.update_token_ids(token_ids, multimodals=dict(image=[img1]))
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)

        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 3, 1])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 5)
        block_trie.evict(5)
        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 2, 3, 3, 1])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 5)
        block_mgr.free(seq0)
        block_trie.evict(5)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks