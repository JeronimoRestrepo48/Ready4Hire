"""
Machine Learning Infrastructure for Ready4Hire.
"""
from .gpu_detector import GPUDetector, get_gpu_detector, GPUType

__all__ = [
    'GPUDetector',
    'get_gpu_detector',
    'GPUType'
]
