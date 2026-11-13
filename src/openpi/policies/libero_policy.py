import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """生成 LIBERO 随机输入示例，用于接口调试。"""
    return {
        "observation/state": np.random.rand(8),  # 8 维机器人状态
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 外部相机图像
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 腕部相机图像
        "prompt": "do something",  # 任务指令示例
    }


def _parse_image(image) -> np.ndarray:
    """将图像统一为 uint8 类型和 [H, W, C] 布局。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)  # 将 [0, 1] 浮点像素转换为 [0, 255]
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")  # 通道维从最前移到最后
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """将 LIBERO 数据转换为模型统一的输入格式，训练和推理共用。"""

    model_type: _model.ModelType  # 模型类型，决定缺失图像的掩码处理方式

    def __call__(self, data: dict) -> dict:
        # 读取外部和腕部图像，统一像素类型与通道布局。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 使用模型约定的字段名；此处只整理状态，归一化与维度补齐由后续变换完成。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # LIBERO 没有右腕相机，用同尺寸黑图占位。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,  # 外部图像有效
                "left_wrist_0_rgb": np.True_,  # 腕部图像有效
                # π₀-FAST 保留占位图像，其余模型将其标记为无效。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 训练时传递参考动作；此处不补齐维度，补齐由后续模型输入变换完成。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 传递文字指令，后续由分词器转换为 token ID。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """将模型输出转换为 LIBERO 动作格式，仅用于推理。"""

    def __call__(self, data: dict) -> dict:
        # 保留每个动作向量的前 7 维，去除填充维；动作序列长度不变。
        return {"actions": np.asarray(data["actions"][..., :7])}
