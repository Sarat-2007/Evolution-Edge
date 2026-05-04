"""
Evolution Edge – Local Model Package
Local ONNX inference module running on CPU (AMD Ryzen / any x86).
No GPU or NPU required; uses ONNX Runtime CPUExecutionProvider.
"""
from .inference import local_inference, compute_confidence
from .router import SymbolicRouter
from .model_manager import ModelManager

__all__ = ["local_inference", "compute_confidence", "SymbolicRouter", "ModelManager"]
