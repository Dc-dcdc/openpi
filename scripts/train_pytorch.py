"""
π₀ / π₀.₅ 的 PyTorch 训练入口，支持单卡、多卡及多机分布式数据并行训练。
复用 training/config.py 的配置和 training/data_loader.py 的数据处理流程。

单卡示例：
  python scripts/train_pytorch.py pi0_libero --exp_name libero_run

单机多卡示例：
  torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi0_libero --exp_name libero_run

断点续训：在相同训练命令末尾添加 --resume。
多机训练：使用 torchrun，并设置 --nnodes、--nproc_per_node、--node_rank、--master_addr 和 --master_port。
"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    """设置控制台日志格式。"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化或恢复 W&B 实验记录；未启用时使用禁用模式。"""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

# DDP：分布式数据并行，负责建立进程间通信并选择当前进程的设备。
def setup_ddp():
    # 读取全局训练进程数（通常由 torchrun 设置），默认单进程。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 进程数大于 1 时启用分布式数据并行（DDP）。
    use_ddp = world_size > 1
    # 需要分布式训练且进程组尚未初始化时，建立通信进程组。
    if use_ddp and not torch.distributed.is_initialized():
        # 有 CUDA 时使用 NCCL 通信后端，否则使用支持 CPU 的 Gloo。
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # 从环境变量读取 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT 等通信配置。
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # 未指定分布式调试级别时，将环境变量设为 INFO（此处位于初始化之后）。
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # 读取本机进程编号；缺失时回退到全局 RANK，再缺失则取 0。
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    # 有 CUDA 时选择编号对应的 GPU，否则选择 CPU；多机训练应正确设置 LOCAL_RANK。
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # 若使用 CUDA，将选定 GPU 设为当前进程的默认 CUDA 设备。
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # use_ddp：是否启用分布式数据并行
    # local_rank：当前进程在本机的编号
    # device：当前进程使用的计算设备
    return use_ddp, local_rank, device


def cleanup_ddp():
    """同步各训练进程，并销毁分布式通信进程组。"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    """为当前进程设置随机数种子。"""
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    """创建训练数据加载器，并返回其数据配置。"""
    # 创建统一数据加载器，返回 PyTorch 张量。
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """获取模型状态字典，兼容 DDP 包装。"""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """获取模型参数迭代器，兼容 DDP 包装。"""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """由主进程按保存条件写入模型、优化器、训练元信息及归一化统计量。"""
    if not is_main:
        return

    # 在指定保存间隔或 global_step 等于 num_train_steps - 1 时保存。
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # 先写入临时目录，保存完成后再重命名。
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # 清理同名临时目录并重新创建。
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 去除 DDP 包装，使用 safetensors 保存模型并处理共享张量。
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # 保存优化器状态，以便恢复训练。
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # 保存训练步数、配置字典和时间戳。
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # 保存归一化统计量，供部署时还原输入输出。
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # 替换已有目标目录，将完整临时目录重命名为正式检查点。
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # 向 W&B 记录检查点步数。
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """恢复最新检查点的模型和优化器，并返回训练步数。"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # 加载前回收对象并释放空闲显存缓存。
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # 读取并恢复模型权重，缺少文件时抛出异常。
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # 读取并恢复优化器状态，缺少文件时抛出异常。
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # 读取训练元信息，恢复已记录的训练步数。
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # 显存不足时清理缓存并补充错误说明。
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """返回检查点目录中的最大数字步数；没有检查点时返回 None。"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """记录当前显存分配、预留量及峰值；free 表示预留但尚未分配的缓存。"""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # 读取显存分配和预留的历史峰值。
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # 分布式训练时补充全局进程编号和总进程数。
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    """组织模型初始化、数据加载、训练更新及检查点保存。"""

    # ==========================================
    # 初始化设备和分布式训练
    # ==========================================
    # 初始化分布式训练环境，获取当前进程的设备编号及计算设备。
    use_ddp, local_rank, device = setup_ddp()
    # 单进程或全局 rank=0 为主进程，负责主要日志与检查点保存。
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    # 按配置种子和进程编号设置随机数种子。
    set_seed(config.seed, local_rank)


    # ==========================================
    # 准备实验目录和日志记录
    # ==========================================
    # 初始化检查点目录与实验记录。
    resuming = False
    # 断点续训：检查实验目录是否存在可恢复的检查点。
    if config.resume:
        # 根据配置确定当前实验的检查点目录。
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # 按数字目录名查找最新步数；此处尚未验证文件内容。
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    # 未续训且允许覆盖时，删除已有实验检查点目录。
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")


    # 新训练创建实验目录，续训沿用已有目录。
    if not resuming:
        # 为新实验创建检查点目录。
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # 续训沿用配置中指定的实验目录。
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    if is_main:
        # 仅主进程初始化 W&B 实验记录，并遵循是否启用日志的配置。
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)



    # ==========================================
    # 创建数据加载器
    # ==========================================
    # 计算进程数及每卡批量；config.batch_size 表示全局批量。
    world_size = torch.distributed.get_world_size() if use_ddp else 1  # 全局训练进程数，通常每张卡对应一个进程
    effective_batch_size = config.batch_size // world_size  # 每进程批量 = 全局批量 / 进程数
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # 构建训练数据加载器与数据配置；多卡数据划分由加载器处理。
    loader, data_config = build_datasets(config)

    # 新训练时额外取一个样本批次，将最多 5 个样本的多相机图像记录到 W&B。
    if is_main and config.wandb_enabled and not resuming:
        # 单独创建可视化数据加载器，避免消耗主加载器的数据。
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # 拆出观测和动作，将观测对象转为字典以便提取图像。
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # 收集要上传到 W&B 的图像。
        images_to_log = []
        # 从第一个相机的图像张量读取当前批量大小。
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # 将同一样本的各相机图像横向拼接。
            # 单张图由 [C,H,W] 转为 [H,W,C]，再沿宽度拼接各相机视角。
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # 删除可视化临时对象，再进行垃圾回收并释放空闲 CUDA 缓存。
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # 同时释放可视化数据加载器
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")


    # ==========================================
    # 创建模型并配置显存优化
    # ==========================================
    # 统一为 Pi0Config，并使用 PyTorch 训练精度配置。
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # 从现有配置提取参数，构造 Pi0Config。
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # 直接更新配置对象的 dtype，适用于冻结的 dataclass。
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # 创建 π₀/π₀.₅ 模型并搬到设备；预训练权重在下方单独加载。
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # 支持时启用梯度检查点：反向重算部分激活，以计算换显存。
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # 记录模型创建后的初始显存占用。
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 8 个及以上进程时，设置 cuDNN、TF32 和 CUDA 内存分配选项。
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 设置 CUDA 内存分配选项。
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # DDP 包装模型，在反向传播中同步各进程的参数梯度。
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # 检测未参与损失计算的参数
            gradient_as_bucket_view=True,  # 让梯度与通信桶共享存储以节省显存
            static_graph=world_size >= 8,  # 进程数不少于 8 时启用静态计算图模式
        )


    # ==========================================
    # 加载预训练权重
    # ==========================================
    # 从指定目录加载 model.safetensors 作为初始权重；此处不恢复优化器。
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")


    # ==========================================
    # 创建优化器和准备学习率参数
    # ==========================================
    # 读取学习率预热步数、峰值、衰减终点步数及最终学习率。
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # 创建 AdamW；传入模型参数，有梯度的参数才参与当前更新。
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


    # ==========================================
    # 按需恢复训练状态
    # ==========================================
    # 初始化训练步数，按需恢复检查点。
    global_step = 0
    # 续训时恢复模型参数、优化器状态和步数，覆盖前面的初始化状态。
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")


    # ==========================================
    # 定义学习率调度
    # ==========================================
    # 按当前训练步数计算学习率：先线性预热，再余弦衰减到 end_lr。
    def lr_schedule(step: int):
        if step < warmup_steps:
            # 与 JAX 训练保持一致，从较小的初始学习率开始预热。
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # 预热结束后执行余弦衰减。
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos


    # ==========================================
    # 准备训练进度和指标记录
    # ==========================================
    # 切换到训练模式；实际反向传播与参数更新仍需在循环中执行。
    model.train()
    start_time = time.time()
    infos = []  # 收集当前日志区间的统计指标
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # 主进程显示训练进度，续训时从已完成步数开始。
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )


    # ==========================================
    # 循环训练：读取数据 → 计算损失 → 更新参数
    # ==========================================
    # 反复遍历数据，直到达到配置的总训练步数。
    while global_step < config.num_train_steps:
        # 加载器支持时更新分布式采样轮次，改变各轮的数据顺序。
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))


        # ==========================================
        # 读取一批数据并搬到设备
        # ==========================================
        # 每批读取观测（图像、语言、状态等）及真实动作序列。
        for observation, actions in loader:
            # 达到目标训练步数后退出当前数据遍历。
            if global_step >= config.num_train_steps:
                break

            # 遍历观测结构，将其中的 PyTorch 张量搬到设备；不是转为 JAX 数组。
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            # 真实动作使用 float32，默认形状为 [B, 50, 32]，B 为当前进程的批量。
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # 在本次前向计算前，为各参数组更新学习率。
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)


            # ==========================================
            # 前向计算损失
            # ==========================================
            # 调用 PI0Pytorch.forward，返回逐元素流匹配损失，默认 [B, 50, 32]。
            losses = model(observation, actions)
            # 兼容不同的损失返回类型，统一为张量。
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            # 对批次、动作步及动作维求均值，得到标量训练损失。
            loss = losses.mean()


            # ==========================================
            # 反向传播与梯度裁剪
            # ==========================================
            # 反向传播，将梯度写入参与计算且需梯度的参数 .grad；此时还未更新参数。
            loss.backward()

            # 前 5 步由主进程记录反向传播后的 CUDA 内存占用。
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 按总梯度范数裁剪，限制更新幅度；返回值为裁剪前的总范数。
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)


            # ==========================================
            # 更新模型参数并清空梯度
            # ==========================================
            # 根据当前梯度及 AdamW 状态更新模型参数。
            optim.step()
            # 清空梯度，避免下一批反向传播时累加上一批梯度。
            optim.zero_grad(set_to_none=True)

            # 额外清理仍存在的梯度引用；上一步通常已将它们设为 None。
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None


            # ==========================================
            # 记录训练日志
            # ==========================================
            # 收集主进程的训练指标。
            if is_main:
                # 主进程收集本批次损失、学习率和梯度范数；这里未额外跨进程平均日志值。
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            # 每隔指定步数汇总日志区间内的指标，并输出耗时。
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # 计算当前日志区间内的平均损失和学习率。
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # 启用 W&B 时上传统计指标，随后重置日志计时与收集列表。
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # 清空当前日志区间的统计指标


            # ==========================================
            # 累计训练步数并按条件保存检查点
            # ==========================================
            # 本批参数更新完成，将训练步数加一。
            global_step += 1
            # 调用保存函数；是否实际保存由函数内部的主进程和步数条件决定。
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # 刷新进度条及当前批次的损失、学习率和步数。
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )


    # ==========================================
    # 结束训练并释放资源
    # ==========================================
    # 训练结束后关闭进度条。
    if pbar is not None:
        pbar.close()

    # 主进程结束 W&B 运行，随后清理分布式进程组。
    if is_main and config.wandb_enabled:
        wandb.finish()

    # 释放分布式训练资源。
    cleanup_ddp()



# ==========================================
# 训练总入口：读取配置并启动训练
# ==========================================
def main():
    """读取命令行训练配置并启动训练。"""
    init_logging()  # 初始化控制台日志

    # 
    '''
    根据命令行选择预设训练配置，并用命令行参数覆盖默认设置。
    python scripts/train_pytorch.py pi0_libero
    在 config.py 的 _CONFIGS 中找到
    TrainConfig(name="pi0_libero", ...)从而获取配置
    '''
    config = _config.cli()  
    train_loop(config)  # 执行完整训练流程


if __name__ == "__main__":
    main()
