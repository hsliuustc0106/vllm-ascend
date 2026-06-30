from typing import Tuple, List, Dict, Optional
from vllm.utils import cdiv
from vllm_ascend.attention.attention_v1 import AscendMetadata, AscendAttentionState
import torch
import numpy as np

from vllm.debug_config import global_debug_config, DebugConfig
from vllm.v1.cache import AbstractCache, LRUCache, LFUCache, HotScoreCache
from vllm_ascend.attention.sparse_select import SelectMethod, ScoreMaskBuilder, gen_local_choices, gen_all_choices, gen_topk_blocks_decode

class NPUCacheManager:
    """
    管理 NPU Cache Block 与 CPU KV Cache Block 的映射关系
    """
    def __init__(
        self,
        num_gpu_cache_blocks: int,
        num_cpu_blocks: int,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_batch_size: int,
        max_num_batch_tokens: int,
        max_seq_len: int,
        block_repr_table: torch.Tensor,  # (num_layers, max_num_reqs, max_num_blocks_per_req, num_kv_heads, 1, head_dim)
        sparse_topk: int = 10240000,
        copy_method: str = "merged",
        cache_policy: str = "lru",
    ):
        self.num_gpu_cache_blocks = num_gpu_cache_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_cpu_blocks = num_cpu_blocks

        self.device = block_repr_table.device
        self.copy_method = copy_method

        self.cache_policy = cache_policy
        if self.cache_policy == "lru":
            self.cache = LRUCache(num_gpu_cache_blocks)
        elif self.cache_policy == "lfu":
            self.cache = LFUCache(num_gpu_cache_blocks)
        elif self.cache_policy == "hot-score":
            self.cache = HotScoreCache(num_gpu_cache_blocks)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_blocks_per_seq = cdiv(self.max_seq_len, self.block_size)
        self.max_num_batch_tokens = max_num_batch_tokens

        # !!! 每个 layer 的 CPU tensor 独立, 防止 CPU->NPU 拷贝操作在 device 执行的时候, CPU tensor 的内容已经被下一 layer 修改
        self.selected_logical_block_ids_cpu_tensor_list = [
            torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.selected_logical_block_ids_np_list = [t.numpy() for t in self.selected_logical_block_ids_cpu_tensor_list]

        self.num_selected_existing_full_blocks_cpu_tensor = torch.zeros(self.max_batch_size, dtype=torch.int32, device="cpu", pin_memory=True)
        self.num_selected_existing_full_blocks_np: np.ndarray = self.num_selected_existing_full_blocks_cpu_tensor.numpy()

        # TODO: slot_mapping int32 会不会存在溢出问题？
        self.new_slot_mapping = torch.zeros(self.max_num_batch_tokens, dtype=torch.int32, device=self.device)
        self.new_slot_mapping_cpu_list = [
            torch.zeros(self.max_num_batch_tokens, dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.new_slot_mapping_np_list = [t.numpy() for t in self.new_slot_mapping_cpu_list]

        # 每层单独分配一个 CPU tensor, 防止 CPU->GPU 提交还未开始时下一 layer 覆盖上一 layer 的 block table
        self.new_block_tables = torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=torch.int32, device=self.device)
        self.new_block_tables_cpu_list = [
            torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]

        self.new_block_tables_np_list = [t.numpy() for t in self.new_block_tables_cpu_list]

        # CPU Tensor
        self.new_seq_lens_list = [torch.zeros(self.max_batch_size, dtype=torch.int32, device="cpu") for _ in range(self.num_layers)]
        self.new_seq_lens_np_list = [t.numpy() for t in self.new_seq_lens_list]

        self.curr_num_actual_tokens = 0
        self.curr_batch_size = 0

        self.swap_in_cpu_tensor_list = [
            torch.zeros((self.num_gpu_cache_blocks, 2), dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.swap_in_np_list = [t.numpy() for t in self.swap_in_cpu_tensor_list]

        self.num_swap_in_blocks = 0

        self.swap_out_cpu_tensor_list = [
            torch.zeros((self.num_gpu_cache_blocks, 2), dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.swap_out_np_list = [t.numpy() for t in self.swap_out_cpu_tensor_list]

        self.num_swap_out_blocks = 0

        # KV Cache 写入 NPU
        self.kv_swap_in_stream = torch.npu.Stream(device=self.device)
        self.kv_swap_in_event = torch.npu.Event()

        # KV Cache 写回 CPU
        self.kv_swap_out_stream = torch.npu.Stream(device=self.device)
        self.kv_swap_out_event = torch.npu.Event()

        self.main_stream = torch.npu.current_stream()

        # reshape 将新产生的 KV 写入 KV Cache
        self.kv_cache_update_event = torch.npu.Event()

        self.sparse_topk = sparse_topk
        self.num_top_k_blocks = 0

        # (num_layers, max_num_reqs, max_num_blocks_per_req, num_kv_heads, 1, head_dim)
        self.block_repr_table = block_repr_table

        self.scores = torch.zeros((self.max_batch_size, self.max_blocks_per_seq), dtype=block_repr_table.dtype, device=self.device)
        self.scores_narrowed = self.scores
        self.score_mask_builder = ScoreMaskBuilder(self.max_batch_size, self.max_blocks_per_seq, device=self.device, dtype=self.scores.dtype)

        # cache attention mask, generate only in layer 0
        self.cached_new_attention_mask: Optional[torch.Tensor] = None

        self.max_num_new_full_blocks = self.max_num_batch_tokens // self.block_size + 2

        # new full blocks logical index
        # for example, new full logical blocks: seq0:1,2,3; seq1:2,3
        # new_full_logical_blocks[0]: [0, 0, 0, 1, 1]
        # new_full_logical_blocks[1]: [1, 2, 3, 2, 3]
        self.new_full_logical_blocks = torch.zeros((2, self.max_num_new_full_blocks), dtype=torch.int32, device=self.device)
        self.new_full_logical_blocks_cpu = torch.zeros((2, self.max_num_new_full_blocks), dtype=torch.int32, device="cpu")
        self.new_full_logical_blocks_np = self.new_full_logical_blocks_cpu.numpy()

    def _batch_prepare(self, attn_metadata: AscendMetadata, top_k_tokens_override: Optional[int] = None):
        """
        batch 中的第一层, 计算 batch 中的 seq_len, query_len, kv_len 等信息
        top_k_tokens_override: 如果不为 None, 则使用该值覆盖 self.sparse_topk
        Args:
            attn_metadata: AscendMetadata
        Returns:
        """
        self.origin_attn_metadata = attn_metadata

        self.attn_state = self.origin_attn_metadata.attn_state

        self.old_seq_lens_np = self.origin_attn_metadata.seq_lens_np

        self.curr_num_actual_tokens = self.origin_attn_metadata.num_actual_tokens

        self.curr_batch_size = self.origin_attn_metadata.num_seqs

        self.query_lens_np = self.origin_attn_metadata.query_lens_np

        self.old_block_tables_np = self.origin_attn_metadata.block_tables_np

        self.in_block_offset_np = self.origin_attn_metadata.in_block_offset_np

        self.kv_lens_list = [0] * self.curr_batch_size
        self.num_existing_full_blocks = [0] * self.curr_batch_size
        self.num_full_blocks = [0] * self.curr_batch_size
        self.num_new_full_blocks = [0] * self.curr_batch_size
        self.num_partial_blocks = [0] * self.curr_batch_size

        top_k_tokens = self.sparse_topk
        if top_k_tokens_override is not None:
            top_k_tokens = top_k_tokens_override
        max_num_existing_full_blocks = 0
        self.num_top_k_blocks = top_k_tokens // self.block_size

        total_num_new_full_blocks = 0
        max_num_existing_full_blocks = 0

        for seq_idx in range(self.curr_batch_size):
            # 这里的隐含条件是 seq = [ kv | q ]
            # q 的长度
            seq_len = int(self.old_seq_lens_np[seq_idx])
            query_len = int(self.query_lens_np[seq_idx])
            # kv 长度
            self.kv_lens_list[seq_idx] = seq_len - query_len

            # 已经 cache 的完整 block 数量
            self.num_existing_full_blocks[seq_idx] = self.kv_lens_list[seq_idx] // self.block_size
            max_num_existing_full_blocks = max(max_num_existing_full_blocks, self.num_existing_full_blocks[seq_idx])
            self.num_selected_existing_full_blocks_np[seq_idx] = min(self.num_existing_full_blocks[seq_idx], self.num_top_k_blocks)

            self.num_full_blocks[seq_idx] = seq_len // self.block_size
            # num_new_full_blocks 新填满的 block 数量(可能是之前未填满的 block, 也可能是新分配的完整 block)
            self.num_new_full_blocks[seq_idx] = self.num_full_blocks[seq_idx] - self.num_existing_full_blocks[seq_idx]
            # 未填满的 block(有可能是新产生的, 也有可能是上一轮为填满, 这一轮仍未填满)
            self.num_partial_blocks[seq_idx] = cdiv(seq_len, self.block_size) - self.num_full_blocks[seq_idx]

            # 计算新产生的 full block 的 logical index
            if self.num_new_full_blocks[seq_idx] > 0:
                # for example, new full logical blocks: seq0:1,2,3; seq1:2,3
                # new_full_logical_blocks[0]: [0, 0, 0, 1, 1]
                # new_full_logical_blocks[1]: [1, 2, 3, 2, 3]
                range_start_idx = self.num_existing_full_blocks[seq_idx]
                range_num = self.num_new_full_blocks[seq_idx]
                self.new_full_logical_blocks_np[0, total_num_new_full_blocks: total_num_new_full_blocks+range_num] = seq_idx
                self.new_full_logical_blocks_np[1, total_num_new_full_blocks: total_num_new_full_blocks+range_num] = np.arange(range_start_idx, range_start_idx+range_num)

                total_num_new_full_blocks += range_num

        if total_num_new_full_blocks > 0:
            self.new_full_logical_blocks.copy_(self.new_full_logical_blocks_cpu, non_blocking=True)

        if max_num_existing_full_blocks > 0:
            self.scores_narrowed = self.scores[:self.curr_batch_size]
            self.score_mask_builder.make_score_mask(num_blocks_per_req=torch.tensor(self.num_existing_full_blocks, dtype=torch.int32, device=self.device))

        self.select_method = SelectMethod.SelectTopK
        if max_num_existing_full_blocks <= self.num_top_k_blocks:
            self.select_method = SelectMethod.SelectALL

    def layer_prepare(self, layer_idx: int, attn_metadata: AscendMetadata):
        """
        根据 layer_idx 选择 cpu tensor. 如果是第一层, 还需要计算 batch 中的 seq_len, query_len, kv_len 等信息
        Args:
            layer_idx: int
            attn_metadata: AscendMetadata
        Returns:
        """
        if layer_idx == 0:
            self._batch_prepare(attn_metadata=attn_metadata)

        self.selected_logical_block_ids_cpu_tensor = self.selected_logical_block_ids_cpu_tensor_list[layer_idx]
        self.selected_logical_block_ids_np = self.selected_logical_block_ids_np_list[layer_idx]

        self.new_slot_mapping_cpu = self.new_slot_mapping_cpu_list[layer_idx]
        self.new_slot_mapping_np = self.new_slot_mapping_np_list[layer_idx]

        self.new_block_tables_cpu = self.new_block_tables_cpu_list[layer_idx]
        self.new_block_tables_np = self.new_block_tables_np_list[layer_idx]

        self.new_seq_lens = self.new_seq_lens_list[layer_idx]
        self.new_seq_lens_np = self.new_seq_lens_np_list[layer_idx]

        self.swap_in_np = self.swap_in_np_list[layer_idx]
        self.swap_in_cpu_tensor = self.swap_in_cpu_tensor_list[layer_idx]

        self.swap_out_np = self.swap_out_np_list[layer_idx]
        self.swap_out_cpu_tensor = self.swap_out_cpu_tensor_list[layer_idx]

        self.new_block_tables_narrowed = self.new_block_tables.narrow(0, 0, self.curr_batch_size)
        self.new_block_tables_cpu_narrowed = self.new_block_tables_cpu.narrow(0, 0, self.curr_batch_size)

        self.new_slot_mapping_narrowed = self.new_slot_mapping.narrow(0, 0, self.curr_num_actual_tokens)
        self.new_slot_mapping_cpu_narrowed = self.new_slot_mapping_cpu.narrow(0, 0, self.curr_num_actual_tokens)

        self.new_seq_lens_narrowed = self.new_seq_lens.narrow(0, 0, self.curr_batch_size)

        self.block_repr_table_this_layer = self.block_repr_table[layer_idx]

    def select(self, layer_idx: int, query: torch.Tensor, select_method_override: Optional[SelectMethod] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据 block_repr 选择 top-k 个 KV 块
        Args:
            attn_metadata: AscendMetadata 中包含的 block_tables 和 slot_mapping 映射的是在 CPU block 中的位置
            layer_idx: int
            query: [num_tokens, num_q_heads, head_size]
        Returns:
            selected_logical_blocks_np: batch 中每个 seq 选择的 KV 块在 seq 中的位置
            top_k_block_scores_np: batch 中每个 seq 选择的 KV 块的 attn score
            num_selected_blocks_cpu_np: batch 中每个 seq 实际选择的 KV 块数量
        """
        select_method = self.select_method  # self.select_method 在 _batch_prepare 中设置
        if select_method_override is not None:
            select_method = select_method_override
        if select_method == SelectMethod.SelectALL:
            # select all
            top_k_logical_block_ids_np, top_k_block_scores_np = gen_all_choices(batch_size=self.curr_batch_size,
                                                                                num_existing_full_blocks_per_seq=self.num_existing_full_blocks,
                                                                                selected_logical_block_ids_np=self.selected_logical_block_ids_np)
            return top_k_logical_block_ids_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np
        
        if select_method == SelectMethod.SelectLOCAL:
            # select init + local tokens
            top_k_logical_block_ids_np, top_k_block_scores_np = gen_local_choices(batch_size=self.curr_batch_size,
                                                                                  num_existing_full_blocks_per_seq=self.num_existing_full_blocks,
                                                                                    selected_logical_block_ids_np=self.selected_logical_block_ids_np,
                                                                                    num_init_blocks=self.num_top_k_blocks // 2,
                                                                                    num_local_blocks=self.num_top_k_blocks // 2)
            return top_k_logical_block_ids_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np

        if select_method == SelectMethod.SelectTopK:
            # decode batch
            if self.attn_state == AscendAttentionState.DecodeOnly:
                top_k_logical_block_ids_tensor = gen_topk_blocks_decode(
                    topk=self.num_top_k_blocks,
                    num_reqs=self.curr_batch_size,
                    num_kv_heads=self.num_kv_heads,
                    heads_per_group=self.num_q_heads // self.num_kv_heads,
                    head_dim=self.head_dim,
                    block_repr_table=self.block_repr_table_this_layer,
                    q=query, # (num_batched_tokens, num_q_heads, head_dim))
                    scores=self.scores_narrowed,
                    score_mask=self.score_mask_builder.get_score_mask(),
                )

                # !!! 这里的 selected_logical_blocks 是 NPU tensor (调用 select_sparse_kernel), 会触发 NPU stream 同步
                # TODO: 性能优化, 如果将 allocate 逻辑实现在 NPU 侧可以避免 NPU->CPU 的同步
                selected_logical_blocks_np = top_k_logical_block_ids_tensor.cpu().numpy()
                top_k_block_scores_np = self.scores_narrowed.to(torch.float16).cpu().numpy()
                return selected_logical_blocks_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np
            else:
                # TODO: sparse select in prefill or chunked prefill batch on NPU
                top_k_logical_block_ids_np, top_k_block_scores_np = gen_local_choices(batch_size=self.curr_batch_size,
                                                                                  num_existing_full_blocks_per_seq=self.num_existing_full_blocks,
                                                                                    selected_logical_block_ids_np=self.selected_logical_block_ids_np,
                                                                                    num_init_blocks=self.num_top_k_blocks // 2,
                                                                                    num_local_blocks=self.num_top_k_blocks // 2)
                return top_k_logical_block_ids_np, top_k_block_scores_np, self.num_selected_existing_full_blocks_np

        raise NotImplementedError(f"Unsupported select method {self.select_method}")

    def allocate(self, layer_idx: int, selected_logical_block_ids_np: np.ndarray, selected_logical_block_scores_np: np.ndarray, num_selected_blocks: np.ndarray) -> Tuple[AscendMetadata, torch.Tensor, int, torch.Tensor, int]:
        """
        Parameters:
        layer_idx(int):  当前 Attention layer index
        selected_logical_block_ids_np: np.ndarray, [batch_size, max_num_seleced_blocks]: 稀疏注意力中当前 layer 选择的 KV 块位置。
        selected_logical_block_scores_np: np.ndarray 稀疏注意力中当前 layer 选择的 KV 块 attn score, 用于 cache 热度更新
        num_selected_blocks: np.ndarray, [batch_size]: 每个序列实际选择的 KV 块数量。
        Returns:
        Tuple[AscendMetadata, Dict[int, int], Dict[int, int]]:
            新 AscendMetadata 中包含的 block_tables 和 slot_mapping 映射到已分配的 NPU block
            swap_in_mapping: 需要 KV 换入的 CPU block ID: NPU block ID
            swap_out_mapping: 需要 KV 换出的 NPU block ID: CPU block ID
        """
        # TODO:
        # 1. AscendMetadata 更新适配
        # 2. mask 怎么更新开销最小?

        self.cache.add_timer()
        query_pos_in_slot_mapping = 0

        self.new_slot_mapping_np[:self.curr_num_actual_tokens] = self.in_block_offset_np

        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0

        for seq_idx in range(self.curr_batch_size):
            new_block_tables_next_pos = 0

            # 已经缓存的完整 KV 块稀疏化选择
            for j in range(num_selected_blocks[seq_idx]):
                logical_block_id = int(selected_logical_block_ids_np[seq_idx][j])
                if logical_block_id >= self.num_existing_full_blocks[seq_idx]:
                    continue  # 如果 seq 的 block 数量 < topk, 选择集中可能有溢出的 logical_block_id

                block_id = int(self.old_block_tables_np[seq_idx][logical_block_id])

                slot_id, hit = self.cache.get((layer_idx, block_id), float(selected_logical_block_scores_np[seq_idx][j]))
                self.new_block_tables_np[seq_idx][new_block_tables_next_pos] = slot_id

                new_block_tables_next_pos += 1

                if not hit:
                    # 只从 CPU 中 swap in cache miss的 NPU Cache Block
                    self.swap_in_np[self.num_swap_in_blocks] = [block_id, slot_id]
                    self.num_swap_in_blocks += 1

            # 新产生的完整 KV 块全部选择
            for logical_block_id in range(self.num_existing_full_blocks[seq_idx], self.num_new_full_blocks[seq_idx] + self.num_existing_full_blocks[seq_idx]):
                block_id = int(self.old_block_tables_np[seq_idx][logical_block_id])
                slot_id, hit = self.cache.get((layer_idx, block_id), 1.0)

                # 填满的 block 不需要 pin 在 NPU 中
                # timer 机制保证了这一层新分配的 NPU block 不会被 evict
                self.cache.unpin_block((layer_idx, block_id))

                self.new_block_tables_np[seq_idx][new_block_tables_next_pos] = slot_id
                new_block_tables_next_pos += 1

                # 等计算完成后 swap out 所有 NPU Cache Block 到 CPU 中
                self.swap_out_np[self.num_swap_out_blocks] = [slot_id, block_id]
                self.num_swap_out_blocks += 1

                # 将当前 block 中的 query_tokens 的 slot_mapping 设置为 slot 位置
                num_query_tokens_in_this_block = self.block_size
                # 减掉剩余 KV Cache
                num_query_tokens_in_this_block -= max(0, self.kv_lens_list[seq_idx] - logical_block_id * self.block_size)
                # 截断末尾
                num_query_tokens_in_this_block -= max(0, (logical_block_id + 1) * self.block_size - int(self.old_seq_lens_np[seq_idx]))

                self.new_slot_mapping_np[query_pos_in_slot_mapping: num_query_tokens_in_this_block + query_pos_in_slot_mapping] += slot_id * self.block_size
                query_pos_in_slot_mapping += num_query_tokens_in_this_block

            if self.num_partial_blocks[seq_idx] == 1:
                logical_block_id = self.num_full_blocks[seq_idx]  # = num_existing_full_blocks[i] + num_new_full_blocks[i]
                block_id = self.old_block_tables_np[seq_idx][logical_block_id]

                slot_id, hit = self.cache.get((layer_idx, block_id), 0.0)
                self.cache.pin_block((layer_idx, block_id))
                self.new_block_tables_np[seq_idx][new_block_tables_next_pos] = slot_id
                new_block_tables_next_pos += 1

                num_query_tokens_in_this_block = self.block_size
                # 减掉剩余 KV Cache
                num_query_tokens_in_this_block -= max(0, self.kv_lens_list[seq_idx] - logical_block_id * self.block_size)
                # 截断末尾
                num_query_tokens_in_this_block -= max(0, (logical_block_id + 1) * self.block_size - int(self.old_seq_lens_np[seq_idx]))

                self.new_slot_mapping_np[query_pos_in_slot_mapping: num_query_tokens_in_this_block + query_pos_in_slot_mapping] += slot_id * self.block_size
                query_pos_in_slot_mapping += num_query_tokens_in_this_block

            # 每个请求经过稀疏化后实际选择了多少 token
            num_not_selected_existing_full_blocks = self.num_existing_full_blocks[seq_idx] - int(num_selected_blocks[seq_idx])
            self.new_seq_lens_np[seq_idx] = int(self.old_seq_lens_np[seq_idx]) - num_not_selected_existing_full_blocks * self.block_size

        # commit slot_mapping, block_tables, seq_lens 到 NPU
        self.new_slot_mapping_narrowed.copy_(self.new_slot_mapping_cpu_narrowed, non_blocking=True)
        self.new_block_tables_narrowed.copy_(self.new_block_tables_cpu_narrowed, non_blocking=True) 

        # 修改 attn_mask
        if layer_idx == 0 and self.attn_state == AscendAttentionState.ChunkedPrefill:
            assert isinstance(self.origin_attn_metadata.attn_mask, torch.Tensor)
            max_seq_len=int(max(self.new_seq_lens_np))
            self.cached_new_attention_mask = torch.full((self.curr_num_actual_tokens, max_seq_len),
                                                        dtype=self.origin_attn_metadata.attn_mask.dtype,
                                                        fill_value=-10000.0,
                                                        device=self.origin_attn_metadata.attn_mask.device)
            for seq_idx in range(self.curr_batch_size):
                self.cached_new_attention_mask[seq_idx, :self.new_seq_lens_np[seq_idx]] = self.origin_attn_metadata.attn_mask[seq_idx, self.old_seq_lens_np[seq_idx]-self.new_seq_lens_np[seq_idx]:self.old_seq_lens_np[seq_idx]]

        new_attn_metadata = AscendMetadata(
            num_actual_tokens=self.curr_num_actual_tokens,
            num_seqs=self.curr_batch_size,
            
            query_lens=self.origin_attn_metadata.query_lens,  # 无需更改 CPU Tensor
            query_lens_np=self.query_lens_np,

            seq_lens=self.new_seq_lens_narrowed,  # 需要更改 CPU Tensor
            seq_lens_np=self.new_seq_lens_np,

            max_query_len=self.origin_attn_metadata.max_query_len,  # 无需更改

            block_tables=self.new_block_tables_narrowed,  # 需要更改
            block_tables_cpu=self.new_block_tables_cpu,
            block_tables_np=self.new_block_tables_np,

            slot_mapping=self.new_slot_mapping_narrowed,  # 需要更改
            slot_mapping_cpu=self.new_slot_mapping_cpu,
            slot_mapping_np=self.new_slot_mapping_np,

            attn_mask=self.origin_attn_metadata.attn_mask if self.attn_state!= AscendAttentionState.ChunkedPrefill else self.cached_new_attention_mask,

            attn_state=self.attn_state  # 无需更改
        )

        return new_attn_metadata, self.swap_in_cpu_tensor, self.num_swap_in_blocks, self.swap_out_cpu_tensor, self.num_swap_out_blocks

    def gen_repr(self, layer_idx: int, gpu_k_cache: torch.Tensor, swap_out_mapping_cpu: torch.Tensor, num_swap_out_mapping: int):
        valid_req_ids = self.new_full_logical_blocks[0][:num_swap_out_mapping]
        valid_logical_ids = self.new_full_logical_blocks[1][:num_swap_out_mapping]

        # 获取所有新 block 的 GPU 位置
        gpu_block_ids = swap_out_mapping_cpu[:num_swap_out_mapping, 0].to(self.device)

        # 批量取出 K Cache blocks 并计算特征向量
        selected_blocks = gpu_k_cache[gpu_block_ids]  # (total_num_new_blocks, block_size, num_kv_heads, head_dim)
        block_features = selected_blocks.mean(dim=1)   # (total_num_new_blocks, num_kv_heads, head_dim)

        # (total_num_new_blocks, num_kv_heads, head_dim) -> (total_num_new_blocks, num_kv_heads, 1, head_dim)
        block_features.unsqueeze_(2)

        # 根据 req_id, logical_block_id 的逻辑位置批量写回当前 layer 的 block_repr_table
        self.block_repr_table_this_layer[valid_req_ids, valid_logical_ids] = block_features
        pass