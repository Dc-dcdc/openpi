import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """可选择的机器人或仿真环境类型。"""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """通过配置名称和检查点目录加载策略。"""

    # 训练配置名称，用于确定模型结构和数据处理方式。
    config: str
    # 检查点目录，包含权重及推理所需的归一化统计量等资源。
    dir: str


@dataclasses.dataclass
class Default:
    """使用指定环境的默认策略配置。"""


@dataclasses.dataclass
class Args:
    """策略服务启动参数，由命令行解析。"""

    # 环境类型，仅在选择默认策略时生效。
    env: EnvMode = EnvMode.ALOHA_SIM

    # 输入缺少 prompt 字段时，使用这里指定的默认任务指令。
    default_prompt: str | None = None

    # 服务监听端口，客户端需要连接相同端口。
    port: int = 8000
    # 是否保存策略输入输出，便于调试。
    record: bool = False

    # 策略来源：指定检查点，或采用当前环境的默认配置。
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 各环境对应的默认训练配置与检查点地址。
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """查询环境默认配置，并加载对应策略。"""
    # 查找该环境的默认检查点，读取配置并创建可推理的 Policy。
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据启动参数选择策略加载方式。"""
    match args.policy:
        case Checkpoint():
            # 使用用户指定的配置和权重目录。
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            # 未指定检查点时，按环境选择默认策略。
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    """加载策略并启动 WebSocket 推理服务。"""
    # 加载模型、归一化统计量以及输入输出变换。
    policy = create_policy(args)
    policy_metadata = policy.metadata  # 提供给客户端的策略元信息

    # 启用时将每次推理的输入输出保存到 policy_records，便于回放分析。
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 查询本机名称和地址，仅用于打印启动日志。
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 创建服务：接收客户端观测，调用策略推理并返回动作。
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",  # 监听本机所有 IPv4 网络接口
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()  # 持续运行，等待并处理客户端请求


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)  # 初始化日志
    main(tyro.cli(Args))  # 解析命令行参数并启动服务
