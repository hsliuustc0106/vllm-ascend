#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.ops.attention import vanilla_chunked_prefill

from vllm.debug_config import DebugConfig, global_debug_config
import numpy as np

import time

class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]


class AscendAttentionState(Enum):
    PrefillOnly = 0
    DecodeOnly = 1
    ChunkedPrefill = 2


@dataclass
class AscendMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    num_seqs: int
    
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: torch.Tensor
    block_tables_cpu: torch.Tensor
    block_tables_np: np.ndarray
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    # query_lens 本来就是 CPU Tensor
    query_lens: torch.Tensor
    query_lens_np: np.ndarray
    # seq_lens 本来就是 CPU Tensor
    seq_lens: torch.Tensor
    seq_lens_np: np.ndarray
    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: Optional[torch.Tensor] = None
    slot_mapping_cpu: Optional[torch.Tensor] = None
    slot_mapping_np: Optional[np.ndarray] = None
    # TODO: Indicates whether there are only prefill requests.
    # FlashAttention can be used when there are only prefill requests.
    # FlashAttention has better performance than PageAtttention,
    # but it does not support decode requests.
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    attn_mask: Optional[torch.Tensor] = None

    # 每个 query token 的 in-block offset = slot_mapping % block_size
    in_block_offset_np: Optional[np.ndarray] = None  


class AscendAttentionMetadataBuilder:

    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs, num_actual_tokens, max_query_len,
              common_prefix_len):
        block_tables = self.runner.input_batch.block_table
        block_table = (
            block_tables.get_device_tensor()[:num_reqs])
        query_lens = self.runner.query_lens
        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        attn_mask = self.runner.attn_mask
        attn_state = self.runner.attn_state

        attn_metadata = AscendMetadata(num_actual_tokens=num_actual_tokens,
                                       num_seqs=num_reqs,
                                       block_tables=block_table,
                                       block_tables_cpu=block_tables.get_cpu_tensor(),
                                       block_tables_np=block_tables.get_numpy_array(),
                                       query_lens=query_lens,  # CPU Tensor
                                       query_lens_np=self.runner.query_lens_np,
                                       seq_lens=seq_lens,  # CPU Tensor
                                       seq_lens_np=self.runner.seq_lens_np,
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       slot_mapping_cpu=self.runner.slot_mapping_cpu,
                                       slot_mapping_np=self.runner.slot_mapping_np,
                                       in_block_offset_np=self.runner.block_offsets,
                                       attn_mask=attn_mask,
                                       attn_state=attn_state)
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        kv_swap_in_event: Optional[torch.npu.Event] = None,
        kv_update_event: Optional[torch.npu.Event] = None,
        main_stream: Optional[torch.npu.Stream] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads * head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads * head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx)
        else:
            num_tokens = query.shape[0]
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                          "encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if kv_cache.numel() > 0:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(key=key,
                                                 value=value,
                                                 key_cache=self.key_cache,
                                                 value_cache=self.value_cache,
                                                 slot_indices=slots)
            if kv_update_event is not None:
                kv_update_event.record(main_stream)

            # reshape 与 swap in 不应该冲突，swap in 的是已经生成过的 full blocks
            if kv_swap_in_event is not None:
                kv_swap_in_event.wait(main_stream)

            if hasattr(layer, 'quant_method'):
                # TODO: Add attr (num_prefills, prefill_metadata, decode_metadata) to AscendMetadata
                pass
            # V0-Style scheduler situation.
            elif attn_metadata.attn_state == AscendAttentionState.PrefillOnly:
                assert attn_metadata is not None
                assert attn_metadata.attn_mask is not None
                mask = attn_metadata.attn_mask
                torch_npu._npu_flash_attention(query=query,
                                               key=key,
                                               value=value,
                                               mask=mask,
                                               seq_len=attn_metadata.seq_lens,
                                               scale_value=self.scale,
                                               num_heads=self.num_heads,
                                               num_kv_heads=self.num_kv_heads,
                                               out=output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                block_tables = attn_metadata.block_tables
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
            # Normal V1 situation.
            else:
                # use chunked prefill for head size 192 scenario, like deepseek
                # paged_attention_splitfuse maybe crash at such scenario
                # TODO: vanilla path will be removed after the kernel support
                # head_size 192 scenario
                if self.head_size == 192:
                    cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
                    cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
                    cu_seqlen_q = torch.tensor(cu_seqlen_q, device="npu")
                    cu_seqlen_k = torch.tensor(cu_seqlen_k, device="npu")
                    cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                    cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                    max_seqlen_q = torch.max(attn_metadata.query_lens)
                    max_seqlen_k = torch.max(attn_metadata.seq_lens)
                    vanilla_chunked_prefill(output, query, self.key_cache,
                                            self.value_cache,
                                            attn_metadata.block_tables,
                                            cu_seqlen_q, cu_seqlen_k,
                                            max_seqlen_q, max_seqlen_k,
                                            self.scale, None, True)
                else:
                    # use paged attention
                    torch_npu._npu_paged_attention_splitfuse(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        mask=attn_metadata.attn_mask,
                        block_table=attn_metadata.block_tables,
                        seq_len=attn_metadata.query_lens,
                        context_lens=attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)
        return output.view(num_tokens, self.hidden_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    layer_idx: Optional[int] = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    if forward_context.gpu_cache_manager is None:
        # for profile run
        self.impl.forward(self,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        layer_idx=layer_idx,
                        trace_flag=False)
        return
    
    cpu_kv_cache = self.cpu_kv_cache[forward_context.virtual_engine]

    time1 = time.perf_counter()
    forward_context.gpu_cache_manager.layer_prepare(layer_idx=layer_idx, attn_metadata=attn_metadata)
    time2 = time.perf_counter()
    # key [num_tokens, num_kv_heads, head_size]
    # query [num_tokens, num_heads, head_size]
    
    # select 和 reshape 操作需等待 swap out 完成
    forward_context.gpu_cache_manager.kv_swap_out_event.wait(forward_context.gpu_cache_manager.main_stream)
    selected_logical_blocks_np, selected_logical_block_scores_np, num_selected_blocks = forward_context.gpu_cache_manager.select(
                                                                                                       layer_idx=layer_idx,
                                                                                                       query=query)
    
    time3 = time.perf_counter()
    # 要将选择的 block 结果同步到 CPU 侧以后, 才能进行 allocate
    try:
        new_attn_metadata, swap_in_mapping, num_swap_in_mapping, swap_out_mapping, num_swap_out_mapping = forward_context.gpu_cache_manager.allocate(layer_idx=layer_idx,
                                                                                                                                                     selected_logical_block_ids_np=selected_logical_blocks_np,
                                                                                                                                                     selected_logical_block_scores_np=selected_logical_block_scores_np,
                                                                                                                                                     num_selected_blocks=num_selected_blocks)
    except RuntimeError as e:
        print(f"has no cpu space.{e}")
        new_attn_metadata, swap_in_mapping, num_swap_in_mapping, swap_out_mapping, num_swap_out_mapping = None, None, 0, None, 0
    time4 = time.perf_counter()

    if global_debug_config.management_breakdown and layer_idx % 20 == 0:
        print(f"layer {layer_idx} prepare {(time2 - time1)*1000:.4f} ms  select {(time3 - time2)*1000:.4f} ms allocate {(time4 - time3)*1000:.4f} ms ")

    if global_debug_config.miss_rate:
        total_num_selected_blocks = 0
        for i in range(new_attn_metadata.num_seqs):
            total_num_selected_blocks += num_selected_blocks[i]
        if total_num_selected_blocks == 0:
            miss_rate = 0
        else:
            miss_rate = num_swap_in_mapping / total_num_selected_blocks
        print(f"layer {layer_idx} swap in {num_swap_in_mapping} / {total_num_selected_blocks} = {miss_rate * 100}%", end="  " if global_debug_config.hit_log else "\n")
    
    if global_debug_config.hit_log:
        for i in range(new_attn_metadata.num_seqs):
            num_selected_blocks_seq = num_selected_blocks[i]
            print(f"layer {layer_idx} seq {i} select: {selected_logical_blocks_np[i, :num_selected_blocks_seq]}")

    if num_swap_in_mapping > 0 and not global_debug_config.bypass_swap_in:
        with torch.npu.stream(forward_context.gpu_cache_manager.kv_swap_in_stream):
            # swap out 的 KV 块写回完毕, 才能开始 swap in, 防止冲突
            forward_context.gpu_cache_manager.kv_swap_out_event.wait(forward_context.gpu_cache_manager.kv_swap_in_stream)
            self.impl.swap_blocks(cpu_kv_cache, kv_cache, swap_in_mapping.narrow(0, 0, num_swap_in_mapping))
            # 记录 swap in 事件
            forward_context.gpu_cache_manager.kv_swap_in_event.record(forward_context.gpu_cache_manager.kv_swap_in_stream)

    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      new_attn_metadata,
                      output=output,
                      layer_idx=layer_idx,
                      kv_swap_in_event=forward_context.gpu_cache_manager.kv_swap_in_event,
                      kv_update_event=forward_context.gpu_cache_manager.kv_cache_update_event,
                      main_stream=forward_context.gpu_cache_manager.main_stream,
                      trace_flag=False)

    if num_swap_out_mapping > 0:
        # 在 gen_repr 操作完成之前, swap_in 操作不能开始
        with torch.npu.stream(forward_context.gpu_cache_manager.kv_swap_out_stream):
            # 等待 FlashAttention 将新产生的完整 KV 写入 GPU Cache 块, 才能 swap out
            forward_context.gpu_cache_manager.kv_cache_update_event.wait(forward_context.gpu_cache_manager.kv_swap_out_stream)
            # TODO: 优化 swap blocks kernel, block 数量太多时, 会造成 GPU 任务队列阻塞
            self.impl.swap_blocks(kv_cache, cpu_kv_cache, swap_out_mapping.narrow(0, 0, num_swap_out_mapping))
            # 先发射 swap out 操作, 再发射 gen_repr 操作, swap out 可以与 attn 计算并行
            forward_context.gpu_cache_manager.gen_repr(layer_idx, kv_cache[0], swap_out_mapping, num_swap_out_mapping)
            forward_context.gpu_cache_manager.kv_swap_out_event.record(forward_context.gpu_cache_manager.kv_swap_out_stream)
            # 在 swap_out 操作完成之前, swap_in 操作不能开始


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
