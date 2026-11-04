# Copyright 2024 Big Vision Authors.
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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy": # 测试用的小模型
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m": # 动作专家   约3亿参数
        # 311M params
        return Config(
            width=1024,       # token 的特征宽度，Block 输入与输出均为 1024 维
            depth=18,         # Transformer Block 的层数
            mlp_dim=4096,    # FFN 升维后的中间特征宽度
            num_heads=8,     # Query 头数
            num_kv_heads=1,  # Key、Value 各有 1 个头，由 8 个 Query 头共享
            head_dim=256,    # 每个 Q、K、V 头的特征维度
        )
    if variant == "gemma_2b": # VLM   约20亿参数
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora": # VLM+LoRA
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora": # 动作专家+LoRA
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
# RMSNorm（均方根归一化）用于调整每个 token 的特征尺度，让进入注意力或 FFN 的数值更稳定，同时保持张量形状不变
# 均方根归一化：对每个 token 的特征维独立归一化，不混合不同 token。
class RMSNorm(nn.Module):
    # 允许在前向方法中创建可学习参数和 Dense 子模块。
    @nn.compact
    # x 通常为 [B, T, D]；cond=None 使用普通 RMSNorm，否则使用条件调制。
    def __call__(self, x, cond):
        # 记录输入类型（如 bfloat16），归一化采用 float32 计算，返回前再转回原类型。
        dtype = x.dtype  # original dtype, could be half-precision
        # 沿最后一维 D 求平方均值：[B, T, D] → [B, T, 1]，便于后续广播。
        # 注意：var 实际是 mean(x²)，并非方差；RMSNorm 不先减去均值。
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        # 等价于 x / sqrt(mean(x²) + ε)；ε=1e-6 防止除零，输出形状仍为 [B, T, D]。
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        # 普通分支：原始 π₀ 不传入条件向量，使用这一分支。
        if cond is None:
            # regular RMSNorm
            # 创建或获取每个特征维的缩放参数，形状 [D]；所有样本和 token 共享。
            # scale 初始为 0，实际乘数是 1+scale，因此初始不会把特征缩成零。
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            # 归一化后逐元素乘可学习缩放 γ=1+scale，形状不变。
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            # 返回归一化特征及 None；外层收到 None 时采用直接残差相加。
            return normed_inputs.astype(dtype), None  # return in original dtype

        # 自适应分支：例如 π₀.₅ 动作专家用时间条件调制归一化与外层残差。
        # adaptive RMSNorm
        # 将条件向量 [B, C] 映射为 [B, 3D]，分别用于缩放、偏移和残差门控。
        # 权重初始化为零，Dense 默认偏置也为零，因此此处三种调制量初始均为零。
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        # 先插入 token 轴：[B, 3D] → [B, 1, 3D]；再拆成三个 [B, 1, D]。
        # 同一样本的调制量广播到所有 token；不同样本可由不同条件产生不同调制。
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        # 条件缩放与偏移：(1+scale) × 归一化特征 + shift，输出 [B, T, D]。
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        # gate 在此处只返回，不参与上面的归一化；由外层用于 residual + gate × 分支输出。
        # gate 不是经过 sigmoid 的概率，不限制在 0～1；它不同于 FFN 内部的 GELU 门控。
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder 负责把语言 token 的整数编号转换成特征向量，也提供将特征映射回词表分数的功能
       文本 → 分词器 → token ID → Embedder.encode → token 特征向量"""
    # 词表大小 V：嵌入表的行数，每个 token ID 对应一行。
    vocab_size: int
    # 嵌入宽度 D：每个 token 的向量长度，默认 π₀ 的图文分支为 2048。
    embed_dim: int

    # 创建或获取可学习的词嵌入表 E；加载检查点时可恢复预训练权重。
    def setup(self):
        # 所有样本和 token 位置共享这张表；表本身不含批次维或序列维。
        self.input_embedding_table = self.param(
            # 参数名称，用于在参数树和检查点中定位嵌入表。
            "input_embedding",
            # 新建参数时使用正态分布随机初始化，不是每次查表都重新初始化。
            nn.initializers.normal(),
            # 权重形状 [V, D]，即 V 个 token 各有一个 D 维向量。
            (self.vocab_size, self.embed_dim),
        )

    # 输入是分词器产生的整数 token ID，通常为 [B, T]；此处不负责分词。
    def encode(self, x):
        # 按 ID 索引表的第 0 轴：[B, T] → [B, T, D]。
        # 例如 ID 为 [2, 0, 1]，就依次取第 2、0、1 行；(x,) 是单元素索引元组。
        x = self.input_embedding_table[(x,)]
        # 所有嵌入乘以 √D，调整输入特征尺度；保持数据类型与形状不变。
        # 这个缩放不是位置编码，也不是 RMSNorm。
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        # 返回语言 token 特征，供后续 Transformer 使用。
        return x

    # 将输入特征映射成词表 logits；通常 [B, T, D] → [B, T, V]，供后续选择要输出的token
    def decode(self, x):
        # 计算 x @ Eᵀ，其中 Eᵀ 为 [D, V]；每个输出分数对应一个词表 token。
        # 此处未做 softmax、采样或文本解码；原始 π₀ 的动作输出使用单独的动作投影。
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
# 联合注意力模块：图文分支与动作专家各用独立权重投影，再在兼容的注意力头空间中交互。
# 注意力分块不等于参数分支：状态、动作虽属不同块，但共享动作专家的投影参数。
class Attention(nn.Module):
    """Attention module."""

    # 两组分支配置；默认 VLM 宽度 2048、动作专家宽度 1024，均为 8 个 Q 头、1 个 KV 头、头维 256。
    configs: Sequence[Config]

    # Flax Linen 装饰器：允许在调用方法内部定义带参数的子模块。
    @nn.compact
    # xs 为已在外层 Block 中归一化的两组特征；positions 用于 RoPE；attn_mask 指定可见性。
    # kv_cache 为本层缓存；推理时可以让 xs 的图文分支为 None，读取预先算好的前缀 K/V。
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 隐藏宽度可以不同，但头维、Q 头数和 KV 头数必须相同，才能沿序列维拼接并联合计算。
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
            
        # 记录第一个有效分支的数值类型；后面注意力分数用 float32 计算，再转回此类型。
        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        # 逐分支收集 (Q, K, V)；_name(..., i) 区分参数名称，两个分支不共享投影权重。
        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            # 跳过未参与本次计算的分支；例如缓存前缀后，仅运行状态与动作后缀。
            if x is None:
                continue

            if config.num_kv_heads == config.num_heads:
                 # Q、K、V 头数相同：标准多头注意力（MHA）
                qkv_einsum = lora.Einsum(
                    # 权重形状：[Q/K/V 三组, 头数, 输入宽度, 每头维度]。   
                    # VLM	   [8, 2048, 256]	8 个头，每个头把 2048 维特征投影到 256 维
                    # 动作专家	[8, 1024, 256]	 8 个头，每个头把 1024 维特征投影到 256 维
                    shape=(3, config.num_heads, config.width, config.head_dim), 
                    # 给当前的 QKV 投影模块命名 
                    # VLM：qkv_einsum    专家动作：qkv_einsum_1
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),  # 决定是否加入 LoRA 分支
                )
                '''
                输入 x：       [B, S, D]         [2, 100, 2048]
                投影权重：     [3, K, D, H]       [3, 8, 2048, 256]
                                ↓ 对 D 维求和   
                输出 QKV：     [3, B, S, K, H]   [3, 2, 100, 8, 256]
                '''
                # 对输入特征维 D 求和，得到 [3, B, S, K, H]，表示 Q、K、V 三组数据，每组包含 B 个样本、每个样本 S 个 token，每个 token ，有K个注意力头，每个注意力头的特征向量长度为H。
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                # MQA/GQA：Q 头多于 KV 头。默认 π₀ 为 MQA，8 个 Q 头共用 1 个 KV 头。
                # ① 定义投影模块：配置权重形状、初始化方式和可选 LoRA
                q_einsum = lora.Einsum(
                    # Q 权重：[8, 分支宽度, 256]；按输入维和输出维进行 LeCun 初始化。
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i), # VLM：q_einsum    专家动作：q_einsum_1
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # ② 调用投影模块：传入计算规则和输入，得到 Q
                '''
                "BTD"  → 第一个输入 x
                "NDH"  → 第二个输入 W
                "BTNH" → 要求的输出顺序
                '''
                q = q_einsum("BTD,NDH->BTNH", x) # Q：[B, T, D] → [B, T, 8, 256]；N 表示 Query 头数。
                kv_einsum = lora.Einsum(
                    # K/V 合并存储的权重：[2, 1, 分支宽度, 256]，两组权重分别生成 K 和 V。
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i), # VLM：kv_einsum    专家动作：kv_einsum_1
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # K、V 各为 [B, S, 1, 256]；这里大写 K 在 einsum 中表示 KV 头数。
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                # 保留该分支的投影结果，随后与其他有效分支一起拼接。
                qkvs.append((q, k, v))

        '''
        Q：[B, T_p + T_r, 8, 256]
        K：[B, T_p + T_r, 1, 256]
        V：[B, T_p + T_r, 1, 256]
                    ↓
        Q、K 加位置编码，计算匹配分数
                    ↓
        掩码 + softmax
                    ↓
        根据注意力权重汇总 V
        '''
        # 默认完整前向：816 个前缀 + 51 个后缀；Q=[B,867,8,256]，K/V=[B,867,1,256]。
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # Q 和 K 使用 RoPE 编码序列位置；V 不加 RoPE。
        q = _apply_rope(q, positions=positions)
        # 预先将 Q 除以 sqrt(头维)，等价于后续 QKᵀ / sqrt(d_head)。
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # 检查投影和位置编码没有改变 Q/K/V 的数值类型。
        assert q.dtype == k.dtype == v.dtype == dtype

        '''
        缓存 K：[B, 816, 1, 256]  ← 图像 + 语言
        当前 K：[B,  51, 1, 256]  ← 状态 + 动作
                        ↓ 沿 axis=1 拼接
        完整 K：[B, 867, 1, 256]
        '''
        if kv_cache is not None:  # 如果提供了缓存
            cache_k, cache_v = kv_cache  # 取出缓存的 K、V
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        # 将 Q 头分组：K=KV 头数，G=每个 KV 头对应的 Q 头数，H=头维。默认 K=1、G=8、H=256。
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # 沿头维 H 计算 Q 与 K 的点积，得 [B,K,G,T,S]；用 float32 提高数值稳定性。
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # 检查掩码为 [B,1,T,S]，头之间共享；外层 Module 已将 [B,T,S] 扩展为此形状。
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        # 用极小有限负数替代禁止访问位置的分数；不是直接将注意力分数乘以 0。
        big_neg = -2.3819763e38  # See gemma/modules.py
        # 掩码扩展为 [B,1,1,T,S]，广播到全部 KV 组及 Q 头；True 保留分数，False 屏蔽。
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        # 对 Key 序列维 S 做 softmax；有效 Query 对被屏蔽位置的权重接近 0。
        # 全被屏蔽的 Query 行不会自动得到全零概率；无效位置的输出不用于有效动作预测。
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 按注意力权重汇总 V，对 S 求和；每个 Q 头得到自己的输出，形状 [B,T,K,G,H]。
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # 合并头的分组维，恢复为 [B,T,8,256]。
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # 按各分支的输入长度切回序列，并使用各自的输出投影恢复原隐藏宽度。
        out = []
        # 当前有效分支在联合 Query 序列中的起始位置；不是缓存 K/V 的位置。
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # 切片 [start:end] 对应当前分支的全部 Query 输出。
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    # 输出权重：[8,256,分支宽度]；分别恢复 VLM 的 2048 维或专家的 1024 维。
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                # 沿所有头和头内特征求和，得到该分支的输出 [B,T_branch,D_branch]。
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                # 下一有效分支从当前切片终点开始。
                start = end
            else:
                # 未运行的分支仍返回 None，保持输出列表与 xs 一一对应。
                out.append(None)

        # 返回两分支注意力输出及本层 K/V；有输入缓存时，返回 K/V 包含缓存和当前输入。
        # 残差连接、后续 RMSNorm 和 FFN 由外层 Block 完成，不在本类内部执行。
        return out, (k, v)

@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    # GELU 门控 FFN：两路升维 → GELU 门控与内容逐元素相乘 → 降维。
    # 注意：当前 Block 调用的是 lora.FeedForward；本类是无 LoRA 分支的基础实现。
    features: int  # 输入与输出宽度 D：VLM 为 2048，动作专家为 1024。
    hidden_dim: int  # 中间宽度 M：VLM 为 16384，动作专家为 4096。

    @nn.compact  # 允许在 __call__ 中定义参数和子模块。
    def __call__(self, x):  # x 为每个 token 的特征，通常形状为 [B, T, D]。
        dtype = x.dtype  # 保留输入数据类型。
        # 创建或获取两套独立升维权重；所有样本和 token 共享它们，权重没有 B、T 维。
        w_gating = self.param(
            "gating_einsum",  # 参数名称：将 W_gate 和 W_up 存放在同一个张量中。
            # D 为输入轴，M 为输出轴；轴 0 区分两套权重，不计入输入维度。
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),  # [2, D, M]：索引 0 是 W_gate，1 是 W_up。
        ).astype(dtype)  # 计算时将权重转换为与输入一致的类型。
        ff_gate = jnp.dot(x, w_gating[0])  # 门控升维 X @ W_gate：[B, T, D] → [B, T, M]。
        gate_value = nn.gelu(ff_gate)  # 逐元素 GELU，生成门控系数；形状不变，值不限于 0～1。

        ff1 = jnp.dot(x, w_gating[1])  # 内容升维 X @ W_up：[B, T, D] → [B, T, M]。
        activations = gate_value * ff1  # 门控调节内容；逐元素乘法，结果仍为 [B, T, M]。

        # 创建或获取降维权重 W_down，将扩展后的特征融合回原宽度。
        w_linear = self.param(
            "linear",  # 降维权重的参数名称，对应公式中的 W_down。
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),  # 按输入 M、输出 D 初始化。
            (self.hidden_dim, self.features),  # 权重形状 [M, D]。
        ).astype(dtype)  # 与输入使用一致的计算类型。
        outputs = jnp.dot(activations, w_linear)  # 降维融合：[B, T, M] → [B, T, D]。
        assert outputs.dtype == dtype  # 检查输出类型与输入一致。
        # 三个投影均无偏置，且不直接混合 token；RMSNorm 和残差相加在外层 Block 完成。
        return outputs  # 返回 FFN 结果，形状与输入相同。


'''
输入 X
  │
  ├─────────────────────── 残差支路 ──┐
  ↓                                  │
RMSNorm                              │
  ↓                                  │
联合 Attention                       │
  ├─ 两分支分别投影 Q、K、V            │
  ├─ Q、K 应用 RoPE                  │
  ├─ 读取可选 KV 缓存                 │
  ├─ 缩放点积 → 掩码 → softmax        │
  ├─ 加权汇总 V                      │
  └─ 两分支分别做输出投影              │
  ↓                                 │
Dropout（默认关闭）                   │
  ↓                                 │
加上原始输入 X ←─────────────────────┘
  │
  ├─────────────────────── 残差支路 ──┐
  ↓                                  │
RMSNorm                              │
  ↓                                  │
各分支独立的门控 FFN                    │
  ├─ 两路升维                          │
  ├─ 一路 GELU → 两路逐元素相乘         │
  └─ 降维                             │
  ↓                                  │
Dropout（默认关闭）                    │
  ↓                                  │
加上 FFN 子层之前的输入 ←───────────────┘
  ↓
本层输出
'''
@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    # 一个 Block 对应一层联合 Transformer；默认两套配置分别属于 VLM 和动作专家。
    # 两分支通过 Attention 交换信息，各自使用独立的归一化、投影和 FFN 参数。
    configs: tuple[Config, ...]

    # 子层输出的随机丢弃比例；默认 0 表示不使用 dropout。
    dropout: float = 0.0
    # 指定 dropout 掩码在哪些轴上广播共享；默认不额外指定共享轴。
    dropout_bdims: tuple[int, ...] = ()

    # 允许在前向方法中定义 Attention、RMSNorm、FFN 等子模块。
    @nn.compact
    # xs 是分支特征列表：通常为 [图文特征 [B, Tp, 2048], 机器人特征 [B, Tr, 1024]]。
    # 分支可为 None，表示本次不计算它；例如缓存图文前缀后，只运行机器人后缀。
    # kv_cache 为本层可复用的 K/V；positions 为位置编号，attn_mask 控制可读取的位置。
    # adarms_cond 按分支提供条件向量；原始 π₀ 为 None，π₀.₅ 专家可接收时间条件。
    # deterministic=True 时禁用 dropout；本函数返回两路更新后的特征及本层 K/V。
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        # 施加激活张量的分片约束，用于设备间的数据布局，不改变这里的数学计算。
        xs = sharding.activation_sharding_constraint(xs)
        # 启用时使用 Dropout；比例为 0 时改用恒等函数，直接返回输入。
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # 建立本层联合注意力模块：两分支独立投影 QKV，按掩码执行联合注意力。
        attn = Attention(configs=self.configs, name="attn")

        # 第一段：先归一化，再做 Attention，最后与原输入残差相加（Pre-Norm）。
        # 列表保存各分支归一化后的特征，xs 中的原输入仍保留给残差连接。
        pre_attn = []
        # 保存注意力子层对应的残差 gate。
        gates = []
        for i, x in enumerate(xs): # 按分支依次处理：i=0 为 VLM，i=1 为动作专家
            if x is not None:
                # 使用该分支自己的注意力前归一化；普通 RMSNorm 返回 gate=None。
                # 条件分支还返回残差 gate，供 Attention 输出后的残差连接使用。
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            # 保留分支顺序；未参与计算的分支仍占一个 None 位置。
            pre_attn.append(x)
            # 保存对应子层的残差 gate；缺失分支记录 None。
            gates.append(gate if x is not None else None)

        # 为归一化结果设置分片约束，再送入注意力。
        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        # 完成 QKV 投影、RoPE、掩码注意力及各分支输出投影，返回分支输出列表和 K/V。
        # 图文输出恢复为 2048 维、专家输出恢复为 1024 维；此时尚未加残差。
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        # 对各有效分支的注意力输出应用 dropout；tree.map 保留列表结构和 None。
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        # 为注意力子层输出设置分片约束。
        post_attn = sharding.activation_sharding_constraint(post_attn)
        # 第一次残差：普通 π₀ 为 原输入 + Attention 输出；有 gate 时为 原输入 + gate × 输出。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]# strict=True 要求三个列表长度一致；_gated_residual 会保留未计算分支的 None。
        # 施加激活张量的分片约束，用于设备间的数据布局，不改变这里的数学计算。
        xs = sharding.activation_sharding_constraint(xs)

        # 第二段：对注意力残差后的特征做 RMSNorm → 门控 FFN → 再次残差相加。
        out = []
        # 重新收集 FFN 子层的残差 gate，不复用注意力子层的 gate。
        gates = []
        # 分支特征与各自配置一一对应；FFN 不直接混合不同 token 或不同分支。
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # 使用独立的 FFN 前归一化参数；条件分支产生的是本次 FFN 残差使用的 gate。
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                # 调用 lora.py 中的 GELU 门控 FFN，包含两路升维、逐元素相乘和输出降维。
                # FFN 内部 GELU 门控与外层残差 gate 是两种不同的运算。
                x = lora.FeedForward(  # noqa: PLW2901
                    # 输入与输出宽度 D：VLM 为 2048，动作专家为 1024。
                    features=config.width,
                    # 中间宽度 M：VLM 为 16384，动作专家为 4096。
                    hidden_dim=config.mlp_dim,
                    # 分支命名为 mlp / mlp_1，以区分两套 FFN 参数。
                    name=_name("mlp", i),
                    # 提供配置则加入 FFN 的 LoRA 增量；否则执行普通门控 FFN。
                    lora_config=config.lora_configs.get("ffn"),
                # 将归一化后的特征传入 FFN，输出形状保持 [B, T, D]。
                )(x)
            # 保存 FFN 结果；未计算分支保持 None，xs 仍保存残差支路的输入。
            out.append(x)
            # 保存对应子层的残差 gate；缺失分支记录 None。
            gates.append(gate if x is not None else None)

        # 为 FFN 输出设置分片约束。
        out = sharding.activation_sharding_constraint(out)
        # 对各有效分支的 FFN 输出应用 dropout，推理或比例为 0 时不丢弃。
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        # 第二次残差：注意力残差结果 + FFN 输出；条件分支先用对应 gate 调节 FFN 输出。
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        # 施加激活张量的分片约束，用于设备间的数据布局，不改变这里的数学计算。
        xs = sharding.activation_sharding_constraint(xs)

        # 返回本层更新后的分支特征，以及 Attention 产生的本层 K/V；FFN 不改写此缓存。
        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


'''
Attention	一层内的注意力计算
Block	    一个完整 Transformer 层
gemma.Module	组织多层 Transformer、嵌入和最终归一化
Pi0	            组织完整模型，并实现训练损失和动作采样
'''
@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    # 联合 Transformer 的分支配置；π₀ 中分别对应 VLM 和动作专家。
    configs: Sequence[Config]  # list of configs, one for each expert
    # 嵌入及层间特征使用的数据类型。
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    # 建立词嵌入、多层 Block 和各分支的最终归一化。
    def setup(self):
        # all experts must have the same depth
        # 两分支在每一层联合计算，因此要求层数相同。
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # 语言 token 嵌入使用第一个分支（VLM）的特征宽度。
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        # 梯度检查点：反向传播时重算部分中间结果，以计算换显存。
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        # 按层扫描 Block：特征依次传入下一层，默认共 18 层。
        self.layers = nn.scan(
            block_cls,
            # 参数沿层轴存储；各层有独立权重，并非所有层共享参数。
            variable_axes={"params": 0},
            # 为不同层拆分参数初始化及 dropout 的随机密钥。
            split_rngs={"params": True, "dropout": True},
            # KV 缓存沿第 0 维逐层取用；位置、掩码、条件等参数在各层复用。
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        # 所有 Block 之后，每个分支再使用各自的最终 RMSNorm。
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        # 语言 ID [B, T] → 特征 [B, T, D]，并转换为指定类型。
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        # 输入已嵌入的分支特征；None 表示本次跳过该分支（例如已有前缀缓存）。
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        # 可选的各层 K/V 缓存；推理时用于复用图文前缀。
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        # 统一各有效分支的计算类型，保留 None 分支。
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        # 掩码 [B, T, S] → [B, 1, T, S]，增加用于头维广播的轴。
        mask = jnp.asarray(mask)[:, None, :, :]
        # 未提供自适应条件时，各分支均使用普通 RMSNorm。
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # 依次执行所有 Block，得到最终特征和各层返回的 K/V。
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        # 检查有效分支的输出数据类型。
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # 对各分支做最终归一化，只取特征结果；同时返回 KV 缓存。
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        # 用最小虚拟输入触发词嵌入参数初始化。
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # 再运行所有分支以建立其参数；use_adarms 决定初始化普通或条件归一化。
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
