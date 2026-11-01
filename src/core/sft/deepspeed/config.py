"""
DeepSpeed SFT 训练配置
"""
from typing import Optional, List
from pydantic import Field

from ...schema.training import TrainingConfig


class DeepSpeedSFTConfig(TrainingConfig):
    """
    DeepSpeed + TRL SFT 训练配置，在通用 TrainingConfig 基础上扩展。

    TrainingConfig 已提供:
        model, batch_size, epochs, learning_rate, output_dir, model_name, platform_config
    """

    # ── 模型本地路径 ───────────────────────────────────────────────────
    model_path: str = Field(..., description="本地模型权重路径（HuggingFace checkpoint 目录）")
    attn_implementation: Optional[str] = Field(
        default=None,
        description="注意力实现：flash_attention_2 / sdpa / eager",
    )

    # ── DeepSpeed ────────────────────────────────────────────────────
    deepspeed_config: Optional[str] = Field(
        default=None,
        description="DeepSpeed JSON 配置文件路径，如 assets/ds_config_zero3.json",
    )
    num_gpus: Optional[int] = Field(
        default=None,
        description="参与训练的 GPU 数量；None 表示自动检测全部可用卡",
    )
    master_port: int = Field(
        default=28500,
        description="分布式训练 rendezvous 端口；避免与默认 29500 冲突",
    )

    # ── 训练超参 ─────────────────────────────────────────────────────
    per_device_batch_size: int = Field(default=1, ge=1, description="每卡 micro batch size")
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_length: int = Field(default=4096, ge=1, description="序列最大长度（截断/packing 单元）")
    pad_to_multiple_of: int = Field(default=512, ge=1, description="序列长度对齐到该值的倍数")
    completion_only_loss: bool = Field(
        default=True,
        description="只对 assistant 回复部分计算 loss；messages 格式数据会在线转为 prompt-completion 格式",
    )
    packing: bool = Field(default=False, description="是否将短序列拼包到 max_length")
    gradient_checkpointing: bool = Field(default=True)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    lr_scheduler_type: str = Field(default="cosine")
    save_steps: int = Field(default=100, ge=1, description="save_strategy=steps 时生效")
    save_strategy: str = Field(default="steps", description="steps=按步数保存 | epoch=每 epoch 结束时保存")
    save_only_model: bool = Field(
        default=True,
        description="仅保存模型权重，不保存优化器/调度器/rng（TRL/Transformers 内置；省空间，无法从 checkpoint 断点续训）",
    )
    logging_steps: int = Field(default=10, ge=1)
    bf16: bool = Field(default=True)

    # ── PEFT / LoRA ──────────────────────────────────────────────────
    use_peft: bool = Field(default=False)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: Optional[List[str]] = Field(
        default=None,
        description="LoRA 作用的模块名列表；None 表示由 PEFT 自动选择",
    )

    # ── 数据 ─────────────────────────────────────────────────────────
    dataset_text_field: str = Field(
        default="text",
        description="JSONL 中文本字段名（conversational 格式用 messages 字段时可忽略）",
    )
    eval_file: Optional[str] = Field(default=None, description="验证集 JSONL 路径（可选）")
    eval_strategy: str = Field(default="epoch", description="验证策略：steps / epoch（有 eval_file 时生效）")
