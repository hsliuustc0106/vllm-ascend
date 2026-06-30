from typing import Tuple, List, Dict, Optional
from enum import Enum
import torch
import numpy as np

from vllm.debug_config import global_debug_config, DebugConfig


class SelectMethod(Enum):
    SelectALL = 1
    SelectLOCAL = 2
    SelectTopK = 3
    SelectNone = 4

def gen_local_choices(batch_size: int, num_existing_full_blocks_per_seq: List[int],
                    selected_logical_block_ids_np: np.ndarray, num_init_blocks: int = 1, num_local_blocks: int = 1,) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 block_repr 选择头尾的 local KV 块。
    Args:
        batch_size: int
        num_existing_full_blocks: List[int] 每个 seq 已经存在的完整 KV 块数量
        selected_logical_block_ids_np: np.ndarray, [max_batch_size, max_num_seleced_blocks] 选择的 KV 块位置
        num_init_blocks: int, 选择的初始 KV 块数量
        num_local_blocks: int, 选择的末尾 KV 块数量
    Returns:
        selected_logical_block_ids_np: batch 中每个 seq 选择的 KV 块在 seq 中的位置
        top_k_block_scores_np: batch 中每个 seq 选择的 KV 块的 attn score
    """
    cur_max_num_selected_blocks = num_init_blocks + num_local_blocks
    for i in range(batch_size):
        num_existing_full_blocks = num_existing_full_blocks_per_seq[i]  # 当前 seq 的完整 kv 块数量

        if cur_max_num_selected_blocks > num_existing_full_blocks:  # 选择所有的完整 kv 块
            selected_logical_block_ids_np[i, :num_existing_full_blocks] = np.arange(0, num_existing_full_blocks, dtype=np.int32)
        else:
            selected_logical_block_ids_np[i, :num_init_blocks] = np.arange(0, num_init_blocks, dtype=np.int32)
            selected_logical_block_ids_np[i, num_init_blocks:cur_max_num_selected_blocks] = np.arange(num_existing_full_blocks - num_local_blocks, num_existing_full_blocks, dtype=np.int32)

    top_k_block_scores_np = np.ones_like(selected_logical_block_ids_np, dtype=np.float16)

    return selected_logical_block_ids_np, top_k_block_scores_np

def gen_all_choices(batch_size: int, num_existing_full_blocks_per_seq: List[int], selected_logical_block_ids_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 block_repr 选择 top-k 个 KV 块。同时, 生产的新 KV 块的 block_repr 会被缓存。
    Args:
        batch_size: int
        num_existing_full_blocks: List[int] 每个 seq 已经存在的完整 KV 块数量
        selected_logical_block_ids_np: np.ndarray, [max_batch_size, max_num_seleced_blocks] 选择的 KV 块位置
    Returns:
        selected_logical_blocks_np: batch 中每个 seq 选择的 KV 块在 seq 中的位置
        top_k_block_scores_np: batch 中每个 seq 选择的 KV 块的 attn score
    """
    for i in range(batch_size):
        num_existing_full_blocks = num_existing_full_blocks_per_seq[i]  # 当前 seq 的完整 kv 块数量
        selected_logical_block_ids_np[i, :num_existing_full_blocks] = np.arange(0, num_existing_full_blocks, dtype=np.int32)

    top_k_block_scores_np = np.ones_like(selected_logical_block_ids_np, dtype=np.float16)
    return selected_logical_block_ids_np, top_k_block_scores_np

def gen_topk_blocks_decode(topk: int, num_reqs: int, num_kv_heads: int, heads_per_group: int, head_dim: int, 
                               block_repr_table: torch.Tensor, q: torch.Tensor, scores: torch.Tensor, score_mask: torch.Tensor) -> torch.Tensor:
    """
    计算每个请求的topk块序号, 支持Group-Query-Attention
    
    Args:
        topk: int - topk的k值
        num_reqs: int - 请求数量
        num_kv_heads: int - KV头数量
        heads_per_group: int - 每组的query头数量 = num_q_heads // num_kv_heads
        block_repr_table: (num_reqs, max_num_blocks_per_req, num_kv_heads, 1, head_dim) - 每个块的特征向量
        q: (num_batched_tokens num_q_heads, head_dim) - 查询向量
        scores: (num_reqs, max_num_blocks_per_req, num_kv_heads, head_dim) 
        score_mask: (num_reqs, max_num_blocks_per_req) - 分数掩码, 无效块位置为 -inf, 有效块位置为0, 必选块位置为 inf
        k: int - topk的k值
    Returns:
        topk_indices: (num_reqs, k) - 每个请求的topk块序号
    """
    # 重新组织q tensor以匹配GQA结构：每个kv_head对应一组q_head
    # decode batch 中 num_batched_tokens = num_reqs
    # (num_reqs, num_q_heads, head_dim) -> (num_reqs, num_kv_heads, heads_per_group, head_dim)
    q_grouped = q.view(num_reqs, num_kv_heads, heads_per_group, head_dim)
    
    # 扩展q_grouped维度以支持与block_repr的广播计算
    # (num_reqs, num_kv_heads, heads_per_group, head_dim) ->
    # (num_reqs, 1, num_kv_heads, heads_per_group, head_dim)  
    q_expanded = q_grouped.unsqueeze(1)
    
    # 计算每个block与每个q_head的点积注意力分数
    # 广播逐元素相乘: (num_reqs, max_num_blocks_per_req, num_kv_heads, heads_per_group, head_dim)
    # 然后在head_dim维度求和: (num_reqs, max_num_blocks_per_req, num_kv_heads, heads_per_group)
    scores_per_head = torch.sum(block_repr_table[:num_reqs] * q_expanded, dim=-1)

    # 对 num_kv_heads, heads_per_group 维度求平均, 得到每个 block 的综合分数
    torch.mean(scores_per_head, (2, 3), out=scores) 
    
    # 将无效块的分数设为负无穷，确保topk时不会选中它们
    # masked_scores: (num_reqs, max_num_blocks_per_req)
    scores.add_(score_mask)
    
    # 对每个请求独立计算topk块序号
    # topk_values: (num_reqs, k) - topk分数值
    # topk_indices: (num_reqs, k) - 对应的块序号
    _, topk_indices = torch.topk(scores, k=topk, dim=1)
    
    return topk_indices

class ScoreMaskBuilder:
    def __init__(self, max_num_reqs: int, max_num_blocks_per_req: int, device, dtype):
        self.device = device
        # (max_num_blocks_per_req,)
        self.indices = torch.arange(max_num_blocks_per_req, device=device).unsqueeze(0)
        # (max_num_blocks_per_req,) -> (max_num_reqs, max_num_blocks_per_req)
        self.indices = self.indices.expand(max_num_reqs, -1)

        self.mask=torch.zeros((max_num_reqs, max_num_blocks_per_req), dtype=dtype, device=device)
        self.mask_narrowed = self.mask

        self._min_value=torch.tensor(-float('inf'), dtype=dtype, device=device)
        self._max_value=torch.tensor(float('inf'), dtype=dtype, device=device)
        self._zero_value=torch.tensor(0.0, dtype=dtype, device=device)

        self.num_blocks_per_req_cached: Optional[torch.Tensor] = None
        

    def make_score_mask(self, num_blocks_per_req: torch.Tensor) -> torch.Tensor:
        """
        高效地创建一个 GPU block mask tensor.

        参数:
        - num_blocks_per_req: 形状为 (num_reqs,) 的一维 GPU tensor, 包含每个请求的 block 数量。

        返回:
        - mask: 形状为 (num_reqs, max_num_blocks_per_req) 的 GPU tensor, 
                有效位置为 0, 无效位置为 -inf。
        """
        if self.num_blocks_per_req_cached is not None and torch.equal(num_blocks_per_req, self.num_blocks_per_req_cached):
            return self.mask_narrowed
        
        self.num_blocks_per_req_cached = num_blocks_per_req.clone().detach()

        # 1. 确保 num_blocks_per_req 在目标设备上，并调整形状用于广播
        #    形状从 (num_reqs,) 变为 (num_reqs, 1)
        num_seqs = num_blocks_per_req.size(0)
        num_blocks_per_req.unsqueeze_(1)

        indices = self.indices[:num_seqs]
        self.mask_narrowed = self.mask[:num_seqs]

        # 2. 比较生成布尔 mask
        #    num_blocks_gpu (num_reqs, 1) 也会被广播成 (num_reqs, max_num_blocks_per_req)
        #    最终得到一个形状为 (num_reqs, max_num_blocks_per_req) 的布尔 tensor
        mask_bool = indices < num_blocks_per_req

        # 4. 使用 torch.where 根据布尔 mask 生成最终结果
        #    条件为 True 的位置填充 0.0，为 False 的位置填充 -inf
        torch.where(mask_bool, self._zero_value, self._min_value, out=self.mask_narrowed)
        return self.mask_narrowed
    
    def get_score_mask(self) -> torch.Tensor:
        """
        获取当前 batch 的 score mask
        Returns:
            mask: (num_reqs, max_num_blocks_per_req) GPU Tensor
        """
        return self.mask_narrowed