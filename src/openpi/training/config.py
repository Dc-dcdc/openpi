"""可用配置列表见 _CONFIGS。"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 绕过 tyro 直接使用 nnx.filterlib.Filter 时的问题。
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """定义数据处理流程所需资源（例如归一化统计量）的存放位置。

    这些资源会复制到检查点内的 `assets/asset_id` 目录中。

    可从其他检查点（如基础模型检查点）或集中存储位置加载资源。
    例如，在微调时从基础模型检查点加载 Trossen 机器人的归一化统计量：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # 资源目录；未指定时使用当前配置的 assets_dirs。可用于从其他检查点
    # （例如基础模型检查点）或集中存储位置加载资源。
    assets_dir: str | None = None

    # 资源标识；未指定时使用数据集仓库 ID。可据此引用
    # 不同机器人平台对应的资源。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot 数据集仓库 ID；为 None 时创建模拟数据。
    repo_id: str | None = None
    # 资源目录下存放该数据集资源的子目录标识。
    asset_id: str | None = None
    # 预先计算的归一化统计量；为 None 时不执行归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 将数据集特有的输入格式重新组织为
    # 后续数据变换所需的统一格式。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据变换，通常包含机器人平台特有的处理，在归一化之前执行。
    # 归一化后的数据结构可参考 `model.Observation` 和
    # `model.Actions` 中的定义。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型专用变换，在数据归一化之后执行。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 为 True 时使用分位数归一化，否则使用标准 z-score 归一化。
    use_quantile_norm: bool = False

    # 数据加载器用于构造动作序列的字段名。序列长度由模型配置中的
    # `action_horizon` 决定；如果 LeRobot 数据集使用其他字段表示动作，
    # 需要相应修改此项。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 为 True 时，使用 LeRobot 数据集中的任务信息生成指令。
    prompt_from_task: bool = False

    # 仅供 RLDS 数据加载器使用（当前仅用于 DROID）。
    rlds_data_dir: str | None = None
    # DROID 数据集的动作空间。
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # 参与采样的数据集列表：名称、版本、权重及可选的 filter_dict_path。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建一组变换。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准 π₀ 系列模型创建模型专用变换。"""

    # 指定时，作为模型使用的默认指令。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot 数据集仓库 ID。
    repo_id: str = tyro.MISSING
    # 定义资源的加载方式。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 供工厂方法进一步更新的基础配置。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # 创建数据变换的工厂。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 创建模型专用变换的工厂。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # 为 True 时，将关节动作转换为相对于当前状态的增量，再传给模型。
    # 夹爪动作仍保留绝对值。
    use_delta_joint_actions: bool = True
    # 指定后，若输入数据缺少 "prompt" 字段，则注入此默认指令。
    default_prompt: str | None = None
    # 为 True 时，将标准 Aloha 的关节和夹爪数值转换到
    # 训练基础模型时使用的 π 内部运行时表示空间。
    # 使用标准 Aloha 数据时应设为 True。
    adapt_to_pi: bool = True

    # 数据字段重组变换。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 从数据集中读取动作序列所使用的字段名。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    配置数据处理流程中各阶段使用的变换。
    适配自己的数据集时，可复制此类，并按照下方注释修改相应变换。
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 字段重组变换只作用于数据集输入，
        # 推理时不执行。它使数据集输入尽可能与
        # 推理环境提供的数据格式一致，例如统一字段名。
        # 下面将数据集中的字段（由数据转换脚本定义）映射为
        # 推理流程中的字段（由 LIBERO 推理脚本定义）。
        # 使用自定义数据集时，先确认环境向策略服务器传入哪些字段，
        # 再修改下面的映射，将数据集字段对应到目标字段。
        # 此处的重组变换仅执行字段名映射。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换既用于训练数据，也用于推理数据。
        # 下面分别定义进入模型的数据变换（inputs）和
        # 模型输出的数据变换（outputs，后者仅在推理时使用）。
        # 这些变换定义在 `libero_policy.py` 中，可参考其中的详细注释，
        # 了解如何适配自定义数据集。定义好自己的变换后，
        # 将下面的变换替换为相应实现即可。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 额外的数据变换：π₀ 使用相对于每个动作块起始状态的增量动作训练。
        # 若数据使用绝对动作，例如目标关节角，
        # 可按需启用下方的增量变换，将其转换为增量动作。
        # 夹爪动作是例外，始终保留绝对值。
        # 下面的示例对前 6 个关节动作应用增量变换，
        # 第 7 个动作（夹爪）保持不变，仍使用绝对值。
        # LIBERO 数据集的原始动作已经是增量动作，通常不需要
        # 再次执行增量转换；应根据数据原本使用的
        # 绝对动作或增量动作表示决定是否启用该变换。

        # LIBERO 已使用增量动作，但部分旧版 π₀ 检查点训练时
        # 额外应用了这一步增量变换。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型专用变换包括对指令和动作目标进行分词等操作。
        # 适配自定义数据集时，通常无需修改这里。
        model_transforms = ModelTransformFactory()(model_config)

        # 返回训练和推理所需的全部数据变换，通常无需修改。
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    使用 RLDS 格式在 DROID 数据集上训练的配置，适用于高效处理大规模数据。
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # 过滤选项：可提供一个字典文件路径，将轨迹映射到需保留的时间步区间。
    # 区间使用 (start, end) 元组表示；轨迹的唯一标识为
    # f"{recording_folderpath}--{file_path}"，这两个字段均位于 RLDS 轨迹元数据中。

    # 参与采样的数据集列表：名称、版本、权重及可选的 filter_dict_path。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 数据加载器返回绝对关节位置动作，训练前转换为增量动作。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    使用 LeRobot 格式的自定义 DROID 数据集配置示例。
    将自定义 DROID 数据（少于数十小时）转换为 LeRobot 格式，参见 examples/droid/convert_droid_data_to_lerobot.py。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 此处假定动作表示关节速度，因此不应额外应用增量变换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置名称，必须唯一，用于引用该配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称，用于命名元数据和检查点目录。
    exp_name: str = tyro.MISSING

    # 模型配置。action_dim、action_horizon、max_token_len 等属性由各模型共用，
    # 定义见 BaseModelConfig。具体模型配置（如 Pi0Config）继承 BaseModelConfig，
    # 并可定义额外属性。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 可选的权重加载器，在模型初始化后从磁盘加载全部或部分权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 用于加载初始权重的 PyTorch 检查点目录，可不指定。
    pytorch_weight_path: str | None = None

    # PyTorch 训练使用的数值精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定训练时需要冻结的参数。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 定义训练使用的数据及其处理方式。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资源（例如归一化统计量）的根目录。
    assets_base_dir: str = "./assets"
    # 检查点根目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 训练中随机数生成器使用的种子。
    seed: int = 42
    # 全局批量大小。
    batch_size: int = 32
    # 数据加载器的工作进程数。增大此值可以加快数据加载，
    # 但会增加内存和 CPU 占用。
    num_workers: int = 2
    # 训练总步数，即处理的批次数。
    num_train_steps: int = 30_000

    # 每隔多少训练步记录一次指标。
    log_interval: int = 100
    # 每隔多少训练步保存一次检查点。
    save_interval: int = 1000
    # 指定后，满足 step % keep_period == 0 的已有检查点将被保留，不参与删除。
    keep_period: int | None = 5000

    # 为 True 时，允许覆盖已有检查点目录。
    overwrite: bool = False
    # 为 True 时，从最近的检查点恢复训练。
    resume: bool = False

    # 为 True 时，启用 W&B 日志记录。
    wandb_enabled: bool = True

    # 传递给策略服务器的元数据。
    policy_metadata: dict[str, Any] | None = None

    # 大于 1 时启用 FSDP，并在指定数量的设备之间分片，
    # 以降低每个设备的显存占用，但训练速度可能下降。
    # 例如，共有 4 个设备且 fsdp_devices=2 时，模型在每组的 2 个设备间分片，
    # 两个设备组之间再执行数据并行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取当前配置的资源目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取当前配置的检查点目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve() # 检查点根目录/配置名称/实验名称

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取筛选可训练参数的过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 在代码中按名称获取配置时，请使用 `get_config`。
_CONFIGS = [
    #
    # Aloha 推理配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # DROID 推理配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # LIBERO 微调配置。
    #
    # 这些训练配置定义在自定义数据集上微调基础模型所需的超参数。
    # 配置包含训练数据集、初始化使用的基础检查点，
    # 以及训练步数、学习率等设置。
    # 适配自己的数据集时，可复制相应配置，按照下面的注释
    # 修改数据集名称和数据变换。
    TrainConfig(
        # 修改名称，使其反映使用的模型和数据集。
        name="pi0_libero",
        # 在此定义模型配置。本例使用 π₀ 架构进行全参数微调；
        # 后面的示例展示如何改为低显存的 LoRA 微调，
        # 以及如何使用 π₀-FAST 作为替代架构。
        model=pi0_config.Pi0Config(),
        # 在此定义训练数据集，本例使用 LIBERO。
        # 对于自己的数据集，将 repo_id 改为对应仓库 ID，
        # 并使用前面为该数据集定义的新 DataConfig 配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 此标志决定是否从 LeRobot 数据集的 task 字段
                # 读取任务指令。设为 True 时，指令会放入输入字典的
                # prompt 字段中；建议设为 True。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 在此指定用于初始化模型的预训练检查点。
        # 它应与上面的模型配置匹配；本例使用 π₀ 基础模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 下方可设置学习率、训练步数等其他超参数。
        # 完整的可用超参数列表见 TrainConfig 基类。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 加载 π₀ 模型并进行 LoRA 微调的示例。
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # 冻结过滤器用于指定训练时不更新的参数。
        # 模型配置提供了便捷方法，可返回对应 LoRA 微调配置的
        # 默认冻结过滤器；请确保这里的模型配置
        # 与上面创建模型时使用的配置一致。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # LoRA 微调时关闭指数移动平均（EMA）。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 加载 π₀-FAST 模型并进行全参数微调的示例。
        # 根据数据集调整 action_dim 和 action_horizon，
        # 其中 action_horizon 为期望的动作块长度。
        # max_token_len 表示模型可处理的非图像 token 最大数量，
        # 包括分词后的指令、本体状态和 FAST 编码后的动作 token。
        # 设置过小可能截断序列末尾的 token，此时代码会发出警告；
        # 设置过大则会浪费内存，因为每个批次样本都会被填充到
        # max_token_len。经验上，单臂机器人可设为约 180，
        # 双臂机器人可设为约 250。通常先尝试较小值，
        # 若训练中频繁出现截断警告，再适当增大。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 注意：这里加载的是 π₀-FAST 基础模型检查点。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 加载 π₀-FAST 模型并进行 LoRA 微调的示例。
        # action_dim、action_horizon 和 max_token_len 的设置方法见上面的注释。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 获取 LoRA 微调的冻结过滤器时，
        # 同样应确保模型配置与上面保持一致。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # LoRA 微调时关闭指数移动平均（EMA）。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Aloha 微调配置。
    #
    # 此测试配置演示如何使用自定义 LeRobot 数据集进行训练。
    # 转换和训练自定义 Aloha 数据集的说明见 examples/aloha_real/README.md。
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # DROID 微调配置。
    #
    TrainConfig(
        # 此配置用于在完整 DROID 数据集上微调 π₀-FAST-base。
        # 使用 RLDS 加载数据，以支持这一大规模数据集的训练。
        # 在自定义 DROID 数据集上微调的示例见下文。
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 设为 DROID RLDS 数据集的路径，即 droid 目录的父目录。
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 原示例说明：10 万步通常足够，使用 8 张 H100 约需 2 天。
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 注意：RLDS DataLoader 在内部处理多进程，因此要求 num_workers=0。
    ),
    TrainConfig(
        # 此配置用于在完整 DROID 数据集上微调 π₀.₅。
        # 使用 RLDS 加载数据，以支持这一大规模数据集的训练。
        # 在自定义 DROID 数据集上微调的示例见下文。
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 设为 DROID RLDS 数据集的路径，即 droid 目录的父目录。
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # 注意：RLDS DataLoader 在内部处理多进程，因此要求 num_workers=0。
    ),
    TrainConfig(
        # 此配置用于在规模较小的自定义 DROID 数据集上微调 π₀.₅-DROID。
        # 与其他微调示例一样，这里使用 LeRobot 数据格式。
        # 将自定义 DROID 数据（少于数十小时）转换为 LeRobot 格式，参见 examples/droid/convert_droid_data_to_lerobot.py。
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # π₀.₅ 使用 32 维动作进行训练。
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # 替换为自定义 DROID LeRobot 数据集的仓库 ID。
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 注意：微调时应复用原始 DROID 归一化统计量。
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA 仿真配置，用于演示如何在简单仿真环境中训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena 与 PolaRiS 配置。
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称获取配置。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
