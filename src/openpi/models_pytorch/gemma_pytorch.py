from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


# 组合 PaliGemma 图文主干与 Gemma 动作专家；两分支参数独立，通过注意力交互。
# 下文默认 π₀：图文宽度 2048、专家宽度 1024，均为 18 层、8 个 Q 头、1 个 KV 头。
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        # 分别控制图文分支和动作专家的自适应 RMSNorm；π₀ 默认均关闭。
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # 将 OpenPI 配置转为 Hugging Face PaliGemma 配置。
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        # 设置语言主干的特征宽度、FFN 宽度、注意力头数与层数。
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        # 可选时间条件用于自适应归一化；标准 π₀ 的图文分支不使用该条件。
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        # 配置 SigLIP 视觉侧：FFN 中间宽度 4304，视觉投影配置维度 2048。
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # 构建动作专家配置：结构与 Gemma 类似，但使用专家自己的宽度与参数。
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # 构建视觉语言模型：SigLIP + 视觉投影 + Gemma 图文主干；自带的语言输出头在 π₀ 中不使用。
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf) 
        # 构建动作专家：使用独立参数的 Gemma 主干处理状态与动作特征；自带的语言输出头不使用。
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)  
        # 移除动作专家的语言词嵌入层，π₀ 的动作专家接收的是外部已经编码好的状态和动作特征，不需要把词语编号转换成向量。
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision) # 为部分参数保留较高精度

    # 设置参数精度；bfloat16 模式下，指定的图像嵌入和归一化参数保留 float32。
    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        # 遍历参数路径，根据名称匹配需恢复为 float32 的参数。
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    '''
    输入图像
    [B, 3, 224, 224]
            ↓ SigLIP：图像分块 + ViT 编码
    [B, 256, 1152]
            ↓ 视觉投影：1152 → 2048
    [B, 256, 2048]
            ↓ 除以 √2048，形状不变
    输出视觉 token
    [B, 256, 2048]
    '''
    # 将图像转换为 Gemma 可以接收的视觉 token 特征。[B, 3, 224, 224]→[B, 256, 2048]
    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image) 

    # 语言 token ID 查嵌入表：[B, L] → [B, L, 2048]；缩放由外层完成。
    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)


    # 调用本模块实例 model(...) 时，PyTorch 自动执行此 forward 方法。
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        # 自适应 RMSNorm 的时间条件向量；None 表示不提供时间条件。
        if adarms_cond is None:
            adarms_cond = [None, None]  # [图文分支条件, 动作专家条件]
        
        # 只有前缀：用于推理时预先计算图文特征，并按 use_cache 返回各层 K/V。
        if inputs_embeds[1] is None:
            '''
            图像＋语言 token 特征 [B, P, 2048]
                    ↓
            18 层 Gemma Transformer
                    ↓
            最终 RMSNorm
                    ↓
            前缀输出特征 [B, P, 2048]＋ 可选的各层 KV 缓存
            '''
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],  # 图像＋语言特征，默认 [B, P, 2048]
                attention_mask=attention_mask,  # 控制 token 间的可见性，并屏蔽填充位置
                position_ids=position_ids,      # token 位置编号，用于 RoPE
                past_key_values=past_key_values,  # 已有的各层 KV 缓存；首次计算通常为 None
                use_cache=use_cache,            # 是否生成并返回 KV 缓存
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                # 图文分支的自适应归一化条件；π₀ 中通常为 None
            )
            prefix_past_key_values = prefix_output.past_key_values  # 各层 KV 缓存
            prefix_output = prefix_output.last_hidden_state   # 最后一层输出特征
            suffix_output = None
        # 只有后缀：动作专家可读取传入的前缀 K/V 缓存，完成当前去噪步计算。
        elif inputs_embeds[0] is None:

            '''
            状态＋动作 token 特征 [B, 51, 1024]
                    ↓
            18 层 Gemma 动作专家（读取各层图文 KV 缓存）
                    ↓
            最终 RMSNorm
                    ↓
            后缀输出特征 [B, 51, 1024]
                    ↓ 取动作 token＋外部输出投影
            预测速度场 [B, 50, 32]
            '''
            suffix_output = self.gemma_expert.model.forward( 
                inputs_embeds=inputs_embeds[1],  # 专家后缀特征，原始 π₀ 默认 [B, 51, 1024]
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values, # 图文分支提前计算好的各层 K/V
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            # 前后缀均存在：逐层联合计算图文分支和专家分支，通常用于训练。
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers


            # 检查训练时是否启用梯度检查点，以重算部分前向结果来节省显存。
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # 当前处于训练模式，且动作专家具有 gradient_checkpointing 属性
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                # 强制启用梯度检查点，以增加部分计算量为代价节省显存
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                # 让后续联合计算使用梯度检查点
                use_gradient_checkpointing = True

            # 打印检查点状态
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}") # 当前联合计算是否使用梯度检查点
                print(f"Model training mode: {self.training}") # 模型是否处于训练模式
                # 动作专家是否具有梯度检查点开关属性
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                # 如果该属性存在，则打印开关的实际值
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                # 标记已经打印，后续调用不再重复打印
                self._debug_gc_printed = True

            # 单层流程：各自归一化与 QKV 投影 → 联合注意力 → 各自输出投影、残差和 FFN。
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                '''
                图文特征 [B, P, 2048]       专家特征 [B, S, 1024]
                        ↓                          ↓
                    各自 RMSNorm、使用各自参数生成 QKV
                                    ↓
                        拼接 QKV，按掩码计算联合注意力
                                    ↓
                            按序列位置拆回两分支
                        ↓                          ↓
                    各自输出投影 → 残差 → RMSNorm → FFN → 残差
                        ↓                          ↓
                    [B, P, 2048]              [B, S, 1024]
                '''
                models = [self.paligemma.language_model, self.gemma_expert.model] #[图文分支的 Gemma 主干, 动作专家的 Gemma 主干]
                
                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    # 各分支独立做 RMSNorm；启用 adaRMS 时同时返回条件残差门控。
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1] # [B, T, D] → [B, T]
                    # 先拆出头维，再交换序列与头轴：Q 为 [B, 8, T, 256]，K/V 为 [B, 1, T, 256]。
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim) # 这里的 -1 表示头数，后续.view会自动计算并替换
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [2, 51, 8, 256] 
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [2, 51, 1, 256]
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # [2, 51, 1, 256]

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)


                # 沿序列轴拼接两分支：Q 为 [B, 8, P+S, 256]，K/V 为 [B, 1, P+S, 256]。
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # 占位张量提供 RoPE 计算所需的形状、设备和类型，不作为注意力特征。
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                # 根据 token 位置生成旋转角度，对 Q、K 应用 RoPE；V 不旋转。
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                # 注意力分数缩放系数为 1/sqrt(head_dim)。
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # 共享 KV 头扩展到各 Q 头；计算缩放 QKᵀ、加掩码、softmax，再加权汇总 V。
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                # 合并注意力头：[B, P+S, 8, 256] → [B, P+S, 2048]；这里代码写死了 8 个头。
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    # 按原来的序列长度，将联合注意力输出切回图文段或专家段。
                    end_pos = start_pos + hidden_states.shape[1]

                    # 如果注意力输出与输出投影权重的数据类型不同，则转换为权重的类型。
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    # 使用各分支自己的输出投影，恢复图文宽度 2048 或专家宽度 1024。
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    # 第一次残差：输入 + 注意力输出；若存在 gate，则先对注意力输出做门控。
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    # FFN 前再次独立归一化，并取得该子层的可选残差 gate。
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    # 门控 FFN：两路升维，一路 GELU，再逐元素相乘并降回分支宽度。
                    out_emb = layer.mlp(out_emb)
                    # second residual
                    # 第二次残差：FFN 输入 + 可选门控后的 FFN 输出。
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            # 依次执行全部 Transformer 层，默认循环 18 次
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # 使用梯度检查点：前向少保存中间激活，反向时按需重算。
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,  # 要执行的单层计算函数
                        layer_idx,               # 当前层编号
                        inputs_embeds,           # 当前图文和专家两分支的特征
                        attention_mask,          # 注意力掩码
                        position_ids,            # token 位置编号
                        adarms_cond,             # 可选的自适应归一化条件
                        use_reentrant=False,     # 使用非重入检查点实现
                        preserve_rng_state=False,  # 不保存、恢复随机数状态
                    )
                else: # 普通方式：直接计算当前层。
                    inputs_embeds = compute_layer_complete(  
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            # 所有 Block 之后，各分支分别执行最终 RMSNorm。
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # 联合路径返回两分支特征；此路径不生成 KV 缓存。
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        # 返回 [前缀特征, 后缀特征] 和前缀缓存；未执行的分支输出为 None。
        return [prefix_output, suffix_output], prefix_past_key_values
