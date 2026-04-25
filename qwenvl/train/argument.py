from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B-Instruct")

    # ── 视觉模块 ──────────────────────────────────────────────────────
    tune_mm_vision: bool  = field(default=False)
    tune_mm_mlp:    bool  = field(default=True)
    tune_mm_llm:    bool  = field(default=True)

    # ── LoRA ──────────────────────────────────────────────────────────
    use_lora:            bool  = field(default=False)
    lora_r:              int   = field(default=128)
    lora_alpha:          float = field(default=256)
    lora_dropout:        float = field(default=0.1)
    lora_target_modules: str   = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    # ── SemanticSlotToken ────────────────────────────────────────────
    use_semantic_slot:   bool = field(default=False)
    slot_num_slots:      int  = field(default=9,
        metadata={"help": "slot 数量，建议设为完全平方数（4/9/16），对应规则空间网格"})
    slot_num_iterations: int  = field(default=2)
    use_position_info:   bool = field(default=True,
        metadata={"help": "开启空间位置偏置，让各 slot 对应固定空间区域，改善位置描述"})
    use_count_head:      bool = field(default=True,
        metadata={"help": "开启无监督数量感知模块，改善目标计数"})

    # ── StylePrefixToken ────────────────────────────────────────────
    use_style_prefix:       bool = field(default=False)
    style_num_tokens:       int  = field(default=12,
        metadata={"help": "style token 数量，建议 8-16"})
    style_init_sentences:   str  = field(default="")
    style_use_slot_attention: bool = field(default=True,
        metadata={"help": "让 style token 感知 SemanticSlot 空间信息，改善位置短语生成"})


@dataclass
class DataArguments:
    dataset_use:  str  = field(default="train_dataset")
    data_flatten: bool = field(default=False)
    image_processor: Optional[object] = field(default=None)
    model_type:      Optional[str]    = field(default=None)
    max_pixels: int = field(default=512 * 512)
    min_pixels: int = field(default=256 * 256)
    video_max_total_pixels: int = field(default=1664 * 28 * 28)
    video_min_total_pixels: int = field(default=256  * 28 * 28)
    video_max_frame_pixels: int = field(default=512  * 512)
    video_min_frame_pixels: int = field(default=256  * 256)
    base_interval:          int = field(default=4)
    video_min_frames:       int = field(default=4)
    video_max_frames:       int = field(default=8)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:        Optional[str] = field(default=None)
    model_max_length: int           = field(default=4096)

    mm_projector_lr: Optional[float] = field(default=None,
        metadata={"help": "Merger MLP 独立学习率"})
    vision_tower_lr: Optional[float] = field(default=None,
        metadata={"help": "ViT 独立学习率，建议 1e-6（比主 LR 低一个数量级）"})
    slot_lr:         Optional[float] = field(default=None,
        metadata={"help": "SemanticSlot + CountHead 独立学习率"})
    style_lr:        Optional[float] = field(default=None,
        metadata={"help": "StylePrefix 独立学习率，建议 1e-3（参数量极小）"})

    resume_training: bool = field(default=False,
        metadata={"help": "True=续训，False=全新训练（自动备份旧目录）"})