import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.

    矩阵回答的是：第 i 个 token 能不能关注第 j 个 token
    位置：        0     1      2       3      4      5
    内容：       图像    指令   PAD     状态   动作1  动作2
    input_mask：  T     T      F       T      T      T
    ar_mask：     F     F      F       T      T      F

    ar_mask：     0  0  0   1   1  0
    累加结果：     0  0  0   1   2  2
                 └─────┘   ↑   └──┘
                 图像语言  状态  动作
    结果矩阵：
    | 查询方 ↓ / 被关注方 → | 图像 | 指令 | PAD | 状态 | 动作1 | 动作2 |
    |       图像          |  ✓   | ✓   |  —  |  —  |   —   |  —   |
    |       指令          |  ✓   | ✓   |  —  |  —  |   —   |  —   |
    |       PAD          |  —   |  —  |  —  |  —  |   —   |   —  |
    |       状态          |   ✓  |  ✓  |  —  |  ✓  |   —  |   —  |
    |       动作1         |   ✓  |  ✓  |  —  |  ✓  |   ✓  |   ✓  |
    |       动作2         |   ✓  |  ✓  |  —  |  ✓  |   ✓  |   ✓  |
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    '''
    [B, 1, S]：被关注方的块编号
    [B, S, 1]：查询方的块编号
                    ↓ 比较
    [B, S, S]：块之间是否允许关注
    '''
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


'''
Pi0：完整模型
├─ 图像编码器：将图像转换为视觉 token
├─ 输入嵌入与投影
│  ├─ 语言 token 嵌入
│  ├─ 状态投影
│  └─ 动作投影与时间特征融合
├─ gemma.Module：联合 Transformer
│  ├─ 词嵌入模块 Embedder
│  ├─ 多层 Block（默认 18 层）
│  │  ├─ RMSNorm → 联合 Attention → 残差
│  │  └─ RMSNorm → 各分支 FFN → 残差
│  └─ 各分支 Final RMSNorm
└─ 动作输出投影：将专家特征映射为动作向量场
'''
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)    # 获取 VLM 结构配置    
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # 获取动作专家结构配置
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config], # 分别传入 VLM 和动作专家的配置
                embed_dtype=config.dtype,  # 嵌入特征的数据类型
                adarms=config.pi05,   # 是否启用 π₀.₅ 条件调制
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear( # 创建一个可训练的线性层，将动作专家的隐藏特征映射回动作维度
                action_expert_config.width,  # 输入维度，默认 1024
                config.action_dim,           # 输出维度，默认 32
                rngs=rngs,                  # 用于初始化参数的随机密钥
)
        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    # 图像 + 语言 → VLM 输入 token
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []  # [B, S, E]：图像与语言的嵌入序列
        ar_mask = []     # [B, S]   ：每个 token 是否有效
        tokens = []      # [S]      ：注意力分块标记
        # embed images
        for name in obs.images:
            # 通过视觉编码器得到图像 token [B, N_img:token数, E]
            # 默认每个 patch 为 14 × 14 像素，16 × 16 = 256 个 patch，保留全部 patch token，不池化成单个向量
            # 因此通过视觉编码器得到图像 token：[B, N_img:token数, E] = [B, 256, 2048]
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False) 

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s", # 视觉编码器把一张图像变成了多个 token，因此需要把标记复制到这些 token 上
                    s=image_tokens.shape[1], # token数量
                )  # 假设批量大小为 2，每张图像生成 4 个 token：[True, False] → [True, True, True, True, False, False, False, False] 复制到每个 token 上
            )
            
            '''
            为图像 token 添加注意力分块标记
            True：在这个位置开启一个新的注意力块。
            False：与前一个位置属于同一个注意力块。
            每个相机都添加 False，因此不同相机的图像 token 也属于同一个块，并没有按相机隔开。
            '''
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            '''
            将语言 token ID 转为嵌入向量
            语言指令 → 分词器 → token ID → 嵌入层 → 语言向量
                     ↑          ↑
                  上游完成    本行完成
                  [B, L]  → [B, L, E]
            '''
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed") # 通过语言编码器得到语言 token
            tokens.append(tokenized_inputs)
            '''
            语言 token：[fold, the, shirt, PAD, PAD]
            有效掩码： [True, True, True, False, False]
            PAD 是 padding（填充）, 用于对齐不同长度的语言指令，使得批量处理时每个样本的长度一致。
            填充位置使用 False 标记，表示这些位置不参与计算和注意力机制。
            '''
            input_mask.append(obs.tokenized_prompt_mask)  # 语言掩码已有 [B, L] 的形状，直接加入即可
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        '''
        [B, N₁, E]  主相机
        [B, N₂, E]  左腕相机
        [B, N₃, E]  右腕相机
        [B, L,  E]  语言
            ↓
        [B, N₁ + N₂ + N₃ + L, E]
        '''
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)  # [B, N₁ + N₂ + N₃ + L]
        ar_mask = jnp.array(ar_mask) # 语言部分也全部设置为 False;因此，图像与语言共同构成一个注意力块：有效的图像 token 和语言 token 之间可以双向关注
        return tokens, input_mask, ar_mask

    @at.typecheck
    # 状态 + 带噪动作 + 时间 → 动作专家输入 token
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[                        # 返回值的类型注解
        at.Float[at.Array, "b s emb"], # suffix_tokens  [b, s, emb]  编码后的状态与动作 token
        at.Bool[at.Array, "b s"],      # suffix_mask  [b, s]  表示哪些 token 是有效的，哪些是 padding
        at.Bool[at.Array, " s"],       # suffix_ar_mask  [s]  表示哪些 token 可以相互注意，用于构造最终注意力掩码
        at.Float[at.Array, "b emb"] | None, # adarms_cond  [b, emb]  用于 adaRMSNorm 的条件输入，如果 pi05 为 False，则为 None
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]  # 把整个机器人状态编码成一个 token
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]  # 表示状态 token 开启一个新的注意力块，这里状态单独作为一个注意力块，但它可以关注前面所有有效的图像和语言 token

        action_tokens = self.action_in_proj(noisy_actions) # 将带噪动作编码成动作 token  [B, 50, 32] → [B, 50, 1024]
        # 将标量时间编码成向量 [B] → [B, 1024]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0) # 正弦余弦编码
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # [B, 1024] → [B, 50, 1024]
            '''
            动作特征：[B, 50, 1024]
            时间特征：[B, 50, 1024]
                     ↓
            拼接结果：[B, 50, 2048]
            '''
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # 沿特征维拼接动作和时间
            action_time_tokens = self.action_time_mlp_in(action_time_tokens) # [B, 50, 2048] mlp→ [B, 50, 1024]
            action_time_tokens = nnx.swish(action_time_tokens)               # Swish 非线性激活 [B, 50, 1024]
            action_time_tokens = self.action_time_mlp_out(action_time_tokens) # [B, 50, 1024] mlp→ [B, 50, 1024]
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        '''
        [
        state_token,           # [B, 1, 1024]
        action_expert_tokens,  # [B, 50, 1024]
        ]
        '''
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        '''
        [True, False, False, ..., False]
        ↑             ↑
        第一个动作      其余 49 个动作
        开启新块       属于同一个块
        '''
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)         # [B, 1 + 50, 1024] = [B, 51, 1024]
        input_mask = jnp.concatenate(input_mask, axis=1) # [B, 51]，全为 True
        ar_mask = jnp.array(ar_mask)                     # [51]，内容为 [True, True, False, …, False]
        return tokens, input_mask, ar_mask, adarms_cond

    @override  #重写了父类接口
    def compute_loss(  # 计算流匹配损失
        # 输入：随机密钥、观测、真实动作；train 必须通过关键字传入
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:  # 返回批量维度 × 动作序列长度的浮点损失
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)  # 将一个随机数密钥rng 分出预处理、噪声和时间采样的随机密钥
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # 统一观测尺寸，训练时启用数据增强

        batch_shape = actions.shape[:-2]  # 获取批量维度，actions.shape = (B, H, D) -2表示取出形状中除最后两维 排除动作序列长度H和动作维度D
        noise = jax.random.normal(noise_rng, actions.shape)  # 生成与动作形状相同的标准高斯噪声
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # 为每个样本采样偏向噪声端的时间 t
        time_expanded = time[..., None, None]  # 增加两个维度，将时间广播到每个动作元素
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 插值构造带噪动作：t=0 为动作，t=1 为噪声
        u_t = noise - actions  # 目标速度：插值路径对 t 的导数

        # 前缀和后缀在同一次 Transformer 前向计算中处理
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # 编码图像和语言，生成前缀及掩码
        # 编码状态、带噪动作和时间；原始 π₀ 的 adarms_cond 为 None
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # 沿序列维拼接有效 token 掩码
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # 拼接注意力分块标记
        attn_mask = make_attn_mask(input_mask, ar_mask)  # 构造 token 之间的注意力可见性矩阵
        positions = jnp.cumsum(input_mask, axis=1) - 1  # 有效 token 从 0 编号，无效位置不增加计数
        #图像与语言   状态与动作  输出保持各分支的序列长度和特征维度
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(  # 执行 VLM 与动作专家，接收输出并忽略返回的 KV 缓存
            # 传入图像与语言， 状态与动作token，  注意力掩码，     位置编号，             各分支的条件向量
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # 取出动作 token，映射为预测速度  [B,50,1024] → [B,50,32]

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # 对动作维度求均方误差，保留批量维和序列维

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
