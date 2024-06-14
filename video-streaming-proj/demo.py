# flake8: noqa:
import sys

import numpy as np
import torch

from lmdeploy import PytorchEngineConfig
from lmdeploy.pytorch.engine import Engine
from lmdeploy.pytorch.models.module_map import MODULE_MAP

model_path = '/home/maningsheng/video_streaming/video_inference'
sys.path.append(model_path)
sys.path.append('/home/maningsheng/lmdeploy/video-streaming-proj')

llava_phi_path = '/home/maningsheng/video_streaming/llava-phi-2-pool-16f-64r-4p-causal'

sys.path.append(model_path)

from config_llava_phi import LlavaPhiModelConfigBuilder  # noqa F401

# model_path = '/nvme/shared_data/InternLM/internlm2-chat-1_8b'
# long_model.model.compressor.compressor.save_pretrained('/home/maningsheng/llava_phi')
# update llava_phi/pytorch_model.bin

# update model_map
MODULE_MAP.update({
    'modeling_phi.MLP':
    'rewrite_llava_phi.PatchedMLP',
    'modeling_phi.MHA':
    'rewrite_llava_phi.PatchedMHA',
    'modeling_phi.PhiModel':
    'rewrite_llava_phi.PatchedPhiModel',
    'llava_phi.LlavaPhiModel':
    'rewrite_llava_phi.PatchedPhiModel',
    'llava_phi.LlavaPhiForCausalLM':
    'rewrite_llava_phi.PatchedLlavaPhiForCausalLM',
})

if __name__ == '__main__':
    engine = Engine(llava_phi_path,
                    engine_config=PytorchEngineConfig(
                        tp=2, cache_max_entry_count=0.1))
    tokenizer = engine.tokenizer
    prompts = 'hello. who are you'
    input_ids = tokenizer.encode(prompts)
    print(input_ids)
    input_ids = np.array(input_ids).reshape(1, -1)

    inputs_embeds, attention_mask, indicators = torch.load(
        '/home/maningsheng/video_streaming/input_clips_phi.pt')
    clip_prompts = torch.load(
        '/home/maningsheng/video_streaming/clip_prompts.pt')
    block_size = 1
    n_seq = inputs_embeds.shape[0]
    full_memory = []
    full_time = []
    select_layer = 15
    N = 64
    P = 4
    TP = 16 * 4
    TN = 16 * 64
    for i in range(0, n_seq, block_size):
        current_embeds = inputs_embeds[i:i + block_size]  # b n c
        current_mask = attention_mask[i:i + block_size]  # b n
        current_indicators = indicators[i:i + block_size]  # b n
        clip_text_prompts = clip_prompts[i]
        input_ids = tokenizer.encode(clip_text_prompts)
        T = len(input_ids)
        current_embeds = current_embeds.squeeze(0)
        n_feat = TN + TP + 1
        secs = [T, n_feat, current_embeds.shape[0] - T - n_feat]
        _, emb, _ = current_embeds.split(secs, dim=0)
        history_time_idx = (current_indicators.ravel() == 200).nonzero(
            as_tuple=False).item()
        if i == 0:
            input_embeddings = [emb]
            input_embedding_ranges = [[T, T + n_feat]]
            input_ids += [0] * n_feat
            current_states = engine.decode(
                [input_ids],
                input_embeddings=[input_embeddings],
                input_embedding_ranges=[input_embedding_ranges],
            )

            history_mem = current_states[:, T + TN:T + TN +
                                         TP, :].detach().clone()
            history_time = current_states[:, -1, :].detach().clone()
            full_memory.append(history_mem)
            full_time.append(history_time)
        else:
            emb = torch.cat([full_memory[-1].detach().clone().squeeze(), emb],
                            dim=0)
            input_embeddings = [emb]
            input_embedding_ranges = [[T, T + TP + n_feat]]
            input_ids += [0] * (TP + n_feat)
            current_states = engine.decode(
                [input_ids],
                input_embeddings=[input_embeddings],
                input_embedding_ranges=[input_embedding_ranges],
            )

            history_states = current_states[:, T + TP + TN:T + TP + TN +
                                            TP, :].detach().clone()
            history_time = current_states[:, -1, :].detach().clone()
            history_mem = history_states.detach().clone()
            full_memory.append(history_mem)
            full_time.append(history_time)

    full_memory = torch.cat(full_memory, dim=0)  # n k c
    full_time = torch.cat(full_time, dim=0)  # n c

    print(full_memory.shape)
    print(full_time.shape)

    questions = [
        'How many pilots are shown in the video?',
        'What airlines are shown in the video?',
        'How is the decoration of the airport?'
    ]

    # forward question
    system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'

    question_feats = []
    for q in questions:
        input_ids = tokenizer.encode(q)
        input_ids = [0] * TP + input_ids
        input_embeddings = [full_memory[-1].detach().clone()]

        input_embedding_ranges = [[0, TP]]
        current_states = engine.decode(
            [input_ids],
            input_embeddings=[input_embeddings],
            input_embedding_ranges=[input_embedding_ranges],
        )
        q_feat = current_states[:, -1, :]
        question_feats.append(q_feat)

    qs_tokens = torch.cat(question_feats, dim=0)
    print(qs_tokens.shape)
    torch.save((full_memory, full_time, qs_tokens), 'lmdeploy_phi_outputs.pt')
