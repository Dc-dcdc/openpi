# math 用于计算秩的平方根；re 用于解析 einsum 表达式。
import math
import re

# Flax 提供模块与参数管理，JAX NumPy 提供可求导的张量运算。
import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


# LoRA 的配置对象；下面只描述低秩分支，不决定哪些基座参数被冻结。
@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    # 低秩维度 r：A、B 中间相连的维度，通常远小于原投影的输入和输出维度。
    rank: int
    # LoRA scaling factor.
    # 增量缩放的分子 α；Einsum 中实际乘的是 scaling_value。
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    # A、B 默认都用小幅随机数初始化，因此这里不保证初始增量严格为零。
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    # False 使用 α/r；True 使用 α/√r。
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    # 指定要分解的两个权重轴，默认是最后两轴；其他轴（例如头数）保留。
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    # 用 L 标记低秩维度，例如把输入 D → 输出 H 拆成 D → L → H。
    label: str = "L"

    # 把方法作为属性访问，例如 config.scaling_value，无需加括号。
    @property
    def scaling_value(self) -> float:
        # 根据配置返回低秩增量的缩放系数。
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


# ① 定义投影模块：配置权重形状、初始化方式和可选 LoRA
class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    # 原始权重 W 的形状；例如单个分支的 Q 投影为 [头数 N, 输入宽度 D, 输出头维 H]。
    shape: tuple[int, ...]
    # Initialization function for the weight.
    # 原始 W 的默认初始化器；调用方可以传入 LeCun 等其他初始化器。
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    # None 表示普通投影；有配置时额外创建 A、B。冻结 W 由外部训练过滤器控制。
    lora_config: LoRAConfig | None = None

    # Flax 初始化模块时建立参数；不是每次前向计算都重新随机初始化。
    def setup(self):
        # 创建或获取基座权重 W；参数名称为 w。
        self.w = self.param("w", self.init_fn, self.shape)

        # 海象运算符 := 同时取得配置并判断是否启用 LoRA。
        if config := self.lora_config:
            # Setup LoRA parameters.
            # 先复制 W 的形状，再分别替换两个分解轴。
            shape_a, shape_b = list(self.shape), list(self.shape)
            # A 的输出轴改为 r：例如 [N, D, H] → [N, D, r]。
            shape_a[config.axes[1]] = config.rank
            # B 的输入轴改为 r：例如 [N, D, H] → [N, r, H]。
            shape_b[config.axes[0]] = config.rank
            # 低秩参数 A：负责把输入映射到 r 维。
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            # 低秩参数 B：负责从 r 维映射到原输出维度。
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    # Flax 的紧凑模块方法装饰器，允许在方法内定义子模块或参数。
    @nn.compact
    # ② 调用投影模块：传入计算规则和输入，得到 QKV的计算结果
    def __call__(self, eqn: str, x):
        
        dtype = x.dtype  # 保留输入精度，下面把参与计算的权重转成同一类型。

        '''
            X：[2, 51, 1024]
                    ↓ W：[8, 1024, 256]：8 个独立的 1024 → 256 线性投影层
            Q_base：[2, 51, 8, 256]
        '''
        # 先计算原始投影 XW；头数、批次、序列等轴如何处理由 eqn 决定。
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        # 海象运算符 := 同时取得配置并判断是否启用 LoRA。
        if config := self.lora_config:
            '''
            X：[2, 51, 1024]
                    ↓ A：[8, 1024, 32]: 8 个独立的 [1024, 32] 矩阵，同一个 1024 维输入分别与它们相乘
            中间结果：[2, 51, 8, 32]
                    ↓ B：[8, 32, 256]
            ΔQ：[2, 51, 8, 256]
            '''
            # 将一次投影拆成 X → XA → (XA)B 两个表达式，避免显式构造完整增量矩阵 AB。
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            # 第一次低秩投影；Q 示例：[B, T, D] → [B, T, N, r]。
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            # 第二次低秩投影；Q 示例：[B, T, N, r] → [B, T, N, H]，两次之间无激活。
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            # 将增量加到投影结果：XW + scaling × XAB；输出形状与原投影相同。
            result = result + lora * config.scaling_value

        # 未启用 LoRA 返回 XW，启用后返回带低秩增量的结果。
        return result

    # 构造低秩分支的两个 einsum 字符串，本方法本身不计算张量。
    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        # 当前实现固定检查原表达式中不能有 L，避免与默认低秩标签冲突。
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        # 解析“输入轴,权重轴->输出轴”；不符合此格式则报错。
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        # 例如 BTD,NDH->BTNH 被拆成 lhs=BTD、rhs=NDH、out=BTNH。
        lhs, rhs, out = m.groups()

        # 只有启用 LoRA 才应调用此辅助函数。
        assert self.lora_config is not None
        # 取出待分解轴的字符；默认 Q 示例中为 D 和 H。
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        # 低秩轴字符，默认 L。
        label = self.lora_config.label

        # A 的权重轴：NDH → NDL。
        a_rhs = rhs.replace(b_label, label)
        # 第一次投影输出轴：BTNH → BTNL。
        a_out = out.replace(b_label, label)
        # Q 示例的第一步：BTD,NDL->BTNL。
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        # B 的权重轴：NDH → NLH。
        b_rhs = rhs.replace(a_label, label)
        # Q 示例的第二步：BTNL,NLH->BTNH，恢复原输出形状。
        eqn_b = f"{a_out},{b_rhs}->{out}"

        # 返回两个表达式，供 __call__ 依次执行。
        return eqn_a, eqn_b


# 带可选 LoRA 的 GELU 门控 FFN；当前 gemma.Block 实际调用的是这个类。
class FeedForward(nn.Module):
    """Feed forward module."""

    # 输入与输出宽度 D：默认 VLM 为 2048，动作专家为 1024。
    features: int
    # 中间宽度 M：默认 VLM 为 16384，动作专家为 4096。
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    # 有配置时，门控升维、内容升维、输出降维都加入各自的 LoRA 参数。
    lora_config: LoRAConfig | None = None

    # 建立三套无偏置的投影权重，以及可选的低秩参数。
    def setup(self):
        # 两路升维权重合并存储：[0] 对应 W_gate，[1] 对应 W_up；不是共享同一矩阵。
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            # 形状 [2, D, M]；两套权重均由所有样本、所有 token 共享。
            (2, self.features, self.hidden_dim),
        )
        # 输出降维权重 W_down，把 M 维中间特征融合回 D 维。
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            # 形状 [M, D]。
            (self.hidden_dim, self.features),
        )
        # 默认不创建低秩分支；下面启用时再替换为 (A, B) 元组。
        self.w_gating_lora = None
        self.w_linear_lora = None
        # 仅在提供配置时创建 LoRA 参数；这一步本身不会冻结原始权重。
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            # 升维 LoRA：(A, B)，形状分别为 [2, D, r]、[2, r, M]。
            # 第一个索引选择 A/B，张量的第 0 轴再选择门控分支/内容分支。
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            # 降维 LoRA：(A, B)，形状分别为 [M, r]、[r, D]。
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    # 前向计算：对每个 token 独立执行相同的门控 FFN。
    @nn.compact
    # 输入一般为 [B, T, D]；不直接混合批次或 token 维。
    def __call__(self, x):
        # 记录输入精度，用于最后检查输出类型。
        dtype = x.dtype  # original dtype, could be half-precision
        # 门控升维：XW_gate + 可选 LoRA 增量，输出 [B, T, M]。
        ff_gate = self._dot(
            x,
            # 原始门控权重 W_gate：[D, M]。
            self.w_gating[0],
            # 分别取出门控分支的 A_gate、B_gate；未启用时传 None。
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        # 逐元素 GELU 生成门控系数；形状不变，输出值不限于 0～1。
        gate_value = nn.gelu(ff_gate)

        # 内容升维：XW_up + 可选 LoRA 增量，同样得到 [B, T, M]。
        ff1 = self._dot(
            x,
            # 原始内容权重 W_up：[D, M]。
            self.w_gating[1],
            # 分别取出内容分支的 A_up、B_up，与门控分支的参数独立。
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        # 逐元素相乘调节内容特征；不是矩阵乘法，输出仍为 [B, T, M]。
        activations = gate_value * ff1

        # 降维融合并加入可选 LoRA 增量：[B, T, M] → [B, T, D]。
        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        # 确保前向计算没有改变输出数据类型。
        assert outputs.dtype == dtype
        # 仅返回 FFN 结果；归一化和外层残差相加由 gemma.Block 负责。
        return outputs

    # 线性投影辅助函数：无 LoRA 返回 XW，有 LoRA 返回 XW + (XA)B。
    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        # 原始投影；jnp.dot 对输入最后一维与权重输入维求和。
        base = jnp.dot(x, w.astype(x.dtype))
        # 没有低秩参数时直接使用原始投影结果。
        if lora_weights is None:
            return base
        # 先计算 XA，再乘 B，最后与 XW 相加；A、B 之间没有激活函数。
        # 注意：当前 FFN 实现没有乘 scaling_value，和上面的 Einsum 不同。
        # 默认 LoRA 变体中 α=r 且 rslora=False，缩放值恰好为 1；修改配置时应注意此差异。
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
