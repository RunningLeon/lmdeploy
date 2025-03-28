import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.paging.block_manager import build_block_manager
from lmdeploy.pytorch.paging.block_trie import BlockTrie


class TestBlockTire:

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

    # @pytest.mark.multimodals
    @pytest.mark.test
    def test_match_multimodals(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        img_data0 = MultiModalTensor(data=None,
                                     start=block_size - (block_size // 2),
                                     end=block_size + (block_size // 2),
                                     meta=dict(hash_value='image_0'))
        img_data1 = MultiModalTensor(data=None,
                                     start=block_size - (block_size // 2),
                                     end=block_size + (block_size // 2),
                                     meta=dict(hash_value='image_1'))
        # initialize cache
        token_ids0 = ([1] * block_size + [2] * block_size)
        token_ids0 += [3] * (block_size // 2)
        multimodals0 = dict(image=[img_data0])
        seq0 = sess.add_sequence(token_ids0, multimodals=multimodals0)
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # test1 pure test
        seq1 = sess.add_sequence(token_ids0)
        block_trie.match(seq1)
        last_node = getattr(seq1.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert last_node.parent is None
        assert len(seq1.logical_blocks) == 0
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # test2 same text and same image
        token_ids2 = token_ids0 + [4] * block_size
        seq2 = sess.add_sequence(token_ids2, multimodals=multimodals0)
        block_trie.match(seq2)
        last_node = getattr(seq2.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert last_node.is_full
        assert last_node.mm_hashes == ('image_0', )
        assert last_node.num_matched == block_size * 2
        assert np.array_equal(last_node.parent.tokens, [1] * block_size)
        assert np.array_equal(last_node.tokens, [2] * block_size)
        ref_cnt = allocator.get_ref_count(seq2.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 4])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)
        block_mgr.free(seq2)
        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 3, 1])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # test2.5 same text prefix and same image
        token_ids = [1] * block_size + [2] * (block_size // 2) + [4] * block_size
        seq2 = sess.add_sequence(token_ids, multimodals=multimodals0)
        block_trie.match(seq2)
        last_node = getattr(seq2.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert not last_node.is_full
        assert last_node.mm_hashes == ('image_0', )
        assert last_node.num_matched == block_size + (block_size // 2)
        assert np.array_equal(last_node.parent.tokens, [1] * block_size)
        assert np.array_equal(last_node.tokens, [2] * (block_size // 2))
        ref_cnt = allocator.get_ref_count(seq2.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 2])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)
        block_mgr.free(seq2)
        allocator.free(np.array([last_node.block]))
        ref_cnt = allocator.get_ref_count(seq0.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [2, 3, 1])
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # test3, same text, different image
        multimodals1 = dict(image=[img_data1])
        seq3 = sess.add_sequence(token_ids0, multimodals=multimodals1)
        block_trie.match(seq3)
        last_node = getattr(seq3.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert last_node.parent is None
        assert len(seq3.logical_blocks) == 0
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # test4 after allocate
        block_mgr.allocate(seq3)
        block_trie.allocate(seq3)
        # same prefix text, same image
        token_ids4 = [1] * block_size + [2] * (block_size // 2) + [4] * block_size
        seq4 = sess.add_sequence(token_ids4, multimodals=multimodals1)
        block_trie.match(seq4)
        last_node = getattr(seq4.logical_blocks, 'last_shared_node', None)
        assert last_node is not None
        assert not last_node.is_full
        assert last_node.parent is not None
        assert last_node.num_matched == block_size + (block_size // 2)
        assert len(seq4.logical_blocks) == 2
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 7)
        ref_cnt = allocator.get_ref_count(seq4.logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [3, 2])
        allocator.free(np.array([last_node.block] * 2))
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 6)

        # # test with multi images
        # token_ids5 = ([1] * block_size + [2] * block_size)
        # token_ids5 += [3] * (block_size // 2) + [4] * block_size
        # multimodals = dict(image=[img_data0, img_data1])
        # seq5 = sess.add_sequence(token_ids5, multimodals=multimodals)
        # block_trie.match(seq5)
        # last_node = getattr(seq5.logical_blocks, 'last_shared_node', None)
        # assert last_node is not None
        # assert last_node.parent is not None
        # assert last_node.num_matched == block_size + 2
        # assert len(seq5.logical_blocks) == 2
        # assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 7)
        # ref_cnt = allocator.get_ref_count(seq5.logical_blocks.get_real_blocks())
        # assert np.array_equal(ref_cnt, [4, 2])
        # allocator.free(np.array([last_node.block] * 2))
        # assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 6)

    # @pytest.mark.multimodals
    # @pytest.mark.test
    def test_allocate_multimodals(self, block_trie, block_mgr, block_size):
        allocator = block_trie.allocator
        sess = SchedulerSession(0, block_size)
        token_ids = ([1] * block_size + [2] * block_size)
        token_ids += [3] * (block_size // 2)
        multimodals = dict(image=[
            MultiModalTensor(data=None, start=0, end=block_size // 4, meta=dict(hash_value='image_0')),
            MultiModalTensor(data=None, start=block_size // 4 + 2, end=block_size //
                             2, meta=dict(hash_value='image_1')),
            MultiModalTensor(data=None, start=block_size // 2 + 2, end=block_size * 2, meta=dict(hash_value='image_2')),
        ])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)

        # first allocate
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 3
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [4, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_2'])
        assert node.num_matched == block_size * 2
        assert np.array_equal(node.tokens, [2] * block_size)
        assert np.array_equal(node.parent.tokens, [1] * block_size)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 3
        assert node.parent not in block_trie.leaves
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # append text token
        seq.update_token_ids([4] * block_size)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 4
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [4, 2, 2, 1])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.num_matched == block_size * 3
        assert node.mm_hashes is None
        assert node.is_full
        expect_tokens = [3] * (block_size // 2) + [4] * (block_size // 2)
        assert np.array_equal(node.tokens, expect_tokens)
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 3
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 4)

        # append image token
        multimodals = dict(
            image=[MultiModalTensor(data=None, start=0, end=block_size, meta=dict(hash_value='image_3'))])
        seq.update_token_ids([5] * (block_size), multimodals=multimodals)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        logical_blocks = seq.logical_blocks
        assert len(logical_blocks) == 5
        ref_cnt = allocator.get_ref_count(logical_blocks.get_real_blocks())
        assert np.array_equal(ref_cnt, [4, 2, 2, 2, 2])
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        assert node is not None
        assert node.mm_hashes == tuple(['image_3'])
        assert not node.is_full
        assert node.num_matched == block_size * 4 + block_size // 2
        assert np.array_equal(node.parent.tokens, [4] * (block_size // 2) + [5] * (block_size // 2))
        assert np.array_equal(node.tokens, [5] * (block_size // 2))
        assert node in block_trie.leaves
        assert len(block_trie.leaves) == 3
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 5)

    @pytest.mark.evict
    def test_evict_multimodals(self, block_trie, block_size, num_gpu_blocks):
        block_mgr = block_trie.block_manager
        sess = SchedulerSession(0, block_size)
        token_ids0 = [1] * (block_size // 2)  # text0
        token_ids0 += [2] * block_size  # img0
        token_ids0 += [3] * (block_size // 2)  # text1
        token_ids0 += [4] * (block_size // 2)  # img1
        token_ids0 += [5] * (block_size // 4)  # text2

        img0 = MultiModalTensor(data=None,
                                start=block_size // 2,
                                end=block_size + (block_size // 2),
                                meta=dict(hash_value='image_0'))
        img1 = MultiModalTensor(data=None,
                                start=block_size * 2,
                                end=block_size * 2 + (block_size // 2),
                                meta=dict(hash_value='image_1'))
        img2 = MultiModalTensor(data=None,
                                start=block_size // 2,
                                end=block_size + (block_size // 2),
                                meta=dict(hash_value='image_2'))

        multimodals0 = dict(image=[img0, img1])
        # three blocks with two image
        seq0 = sess.add_sequence(token_ids0, multimodals=multimodals0)
        block_mgr.allocate(seq0)
        block_trie.allocate(seq0)
        assert len(block_trie.leaves) == 2
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # add other seq1 with two image
        token_ids1 = [1] * (block_size // 2)  # text0
        token_ids1 += [2] * block_size  # img2
        token_ids1 += [3] * (block_size // 2)  # text1
        token_ids1 += [4] * (block_size // 2)  # img1
        token_ids1 += [6] * (block_size // 4)  # text2

        multimodals1 = dict(image=[img2, img1])
        seq1 = sess.add_sequence(token_ids1, multimodals=multimodals1)
        block_trie.match(seq1)
        block_mgr.allocate(seq1)
        block_trie.allocate(seq1)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 6)
        assert len(block_trie.leaves) == 4

        # test1 add seq one same image0 as seq0
        token_ids = [1] * (block_size // 2)  # text0
        token_ids += [2] * block_size  # img0
        token_ids += [7] * block_size  # text, different
        multimodals = dict(image=[img0])
        seq = sess.add_sequence(token_ids, multimodals=multimodals)
        block_trie.match(seq)
        block_mgr.allocate(seq)
        block_trie.allocate(seq)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 8)
        assert len(block_trie.leaves) == 6
        block_mgr.free(seq)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 7)
        block_trie.evict(3)
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 6)
        assert len(block_trie.leaves) == 4
        # free seq1
        block_mgr.free(seq1)
        print(seq1.logical_blocks.get_real_blocks())
        assert len(block_trie.leaves) == 4
        # TODO need to evict twice, how to optimize?
        block_trie.evict(3)
        assert len(block_trie.leaves) == 4
        block_trie.evict(3)
        assert len(block_trie.leaves) == 2
        assert block_mgr.get_num_free_gpu_blocks() == (block_mgr.num_gpu_blocks - 3)

        # free all seqs
        block_mgr.free(seq0)
        # TODO need to evict twice, how to optimize?
        block_trie.evict(num_gpu_blocks)
        block_trie.evict(num_gpu_blocks)
        assert len(block_trie.leaves) == 0
        assert block_mgr.get_num_free_gpu_blocks() == block_mgr.num_gpu_blocks

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
        print(block_trie.leaves)
        assert len(block_trie.leaves) == 1

        block_trie.evict(4)
        assert block_mgr.get_num_free_gpu_blocks() == 5
