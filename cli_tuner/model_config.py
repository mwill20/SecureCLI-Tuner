"""
Model Configuration — QLoRA and model loading settings
"""
from dataclasses import dataclass
from typing import List


@dataclass
class QLoRAConfig:
    """QLoRA fine-tuning configuration (from V1)."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class QuantizationConfig:
    """4-bit quantization settings."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration."""
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: str = "training/checkpoints/phase2-final"
    qlora: QLoRAConfig = None
    quantization: QuantizationConfig = None
    max_length: int = 2048
    
    def __post_init__(self):
        if self.qlora is None:
            self.qlora = QLoRAConfig()
        if self.quantization is None:
            self.quantization = QuantizationConfig()
