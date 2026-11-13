import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 末端不移动，夹爪保持打开的等待动作
LIBERO_ENV_RESOLUTION = 256  # 与训练数据一致的仿真渲染分辨率


@dataclasses.dataclass
class Args:
    # ==========================================
    # 模型服务连接与推理参数
    # ==========================================
    host: str = "0.0.0.0"  # 模型服务地址，运行时可指定为 127.0.0.1 或服务器地址
    port: int = 8000  # 与模型服务监听端口一致
    resize_size: int = 224  # 发送给模型的图像尺寸
    replan_steps: int = 5  # 每次只执行动作块的前几步，然后重新获取观测并预测

    # ==========================================
    # LIBERO 仿真评测参数
    # ==========================================
    task_suite_name: str = (
        "libero_spatial"  # 任务套件，可选 libero_spatial、libero_object、libero_goal、libero_10、libero_90
    )
    num_steps_wait: int = 10  # 开始执行任务前，等待场景物体稳定的步数
    num_trials_per_task: int = 50  # 每个任务的测试次数

    # ==========================================
    # 视频输出与随机种子
    # ==========================================
    video_out_path: str = "data/libero/videos"  # 回放视频保存目录

    seed: int = 7  # 随机种子，用于复现实验


def eval_libero(args: Args) -> None:
    """连接模型服务，在 LIBERO 中执行任务并记录成功率与视频。"""
    # 设置随机种子。
    np.random.seed(args.seed)

    # 根据名称创建任务套件，读取任务总数。
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # 创建视频输出目录。
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 为不同任务套件设置单次测试的动作步数上限。
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # 对应训练示教最长为 193 步
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # 对应训练示教最长为 254 步
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # 对应训练示教最长为 270 步
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # 对应训练示教最长为 505 步
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # 对应训练示教最长为 373 步
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 建立模型服务连接；本脚本负责仿真，服务端负责模型推理。
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 逐个任务评测，累计测试次数和成功次数。
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # 读取当前任务定义。
        task = task_suite.get_task(task_id)

        # 读取该任务预设的初始状态。
        initial_states = task_suite.get_task_init_states(task_id)

        # 创建离屏渲染环境，获取任务文字指令。
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # 使用不同初始状态，重复测试当前任务。
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # 重置环境，并创建待执行动作队列。
            env.reset()
            action_plan = collections.deque()

            # 设置本次测试对应的初始状态并获取观测。
            obs = env.set_init_state(initial_states[episode_idx])

            # 初始化步数和视频帧列表。
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # 前几步执行空动作，等待物体落稳。
                    # 这段等待时间不计入任务动作步数上限。
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # 读取外部相机与腕部相机图像。
                    # 图像旋转 180 度，与训练预处理保持一致。
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    # 等比例缩放并填充至目标尺寸，统一为 uint8 像素。
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # 保存处理后的外部视角图像，用于生成回放视频。
                    replay_images.append(img)

                    if not action_plan:
                        # 动作队列为空时，请求模型生成新的动作块。
                        # 组装图像、机器人状态和语言指令。
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            # 状态共 8 维：末端位置 3 维、轴角 3 维、夹爪关节位置 2 维。
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # 通过网络调用模型服务，取得预测动作序列。
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        # 仅缓存本轮需要执行的动作，其余预测动作不执行。
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()  # 按时间顺序取出一个动作

                    # 在仿真环境中执行动作，获得新观测和任务完成标志。
                    obs, reward, done, info = env.step(action.tolist())
                    if done:  # 在该 LIBERO 环境中，任务成功时结束当前测试
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                # 当前测试发生异常时记录错误，并退出本次动作循环。
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存本次测试视频；同一任务、同一结果的文件名相同，重复测试会覆盖。
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # 记录本次结果及累计成功次数。
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # 记录当前任务成功率和所有已完成测试的总成功率。
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # 全部任务结束后输出最终汇总。
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """创建 LIBERO 仿真环境，返回环境对象和任务文字指令。"""
    task_description = task.language
    # 定位任务场景描述文件，并设置离屏渲染尺寸。
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # 即使使用固定初始状态，种子仍可能影响物体位置
    return env, task_description


def _quat2axisangle(quat):
    """
    将四元数转换为三维轴角向量。实现来源于 robosuite： https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # 限制四元数的标量分量，避免反余弦计算越界。
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    # 由四元数标量分量计算轴角转换所需的分母。
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 旋转角接近零时，直接返回零向量。
        return np.zeros(3)

    # 返回“旋转轴方向 × 旋转角度”，角度单位为弧度。
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)  # 解析命令行参数并启动评测
