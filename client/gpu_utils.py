"""
GPU detection and configuration utilities.

Detects available GPU backends and provides configuration for GPU acceleration.
"""

import os
import sys
from typing import Dict, Optional, Tuple
from logger import get_logger

logger = get_logger("GPU Utils")


class GPUDetector:
    """Detects and reports available GPU acceleration backends."""
    
    def __init__(self):
        self.gpu_info: Dict[str, any] = {}
        self._detect_gpu_backends()
    
    def _detect_gpu_backends(self):
        """Detect all available GPU backends."""
        self.gpu_info = {
            'pytorch_cuda': self._check_pytorch_cuda(),
            'pytorch_mps': self._check_pytorch_mps(),
            'pytorch_rocm': self._check_pytorch_rocm(),
            'opencv_cuda': self._check_opencv_cuda(),
            'opencv_opencl': self._check_opencv_opencl(),
            'onnx_cuda': self._check_onnx_cuda(),
            'onnx_tensorrt': self._check_onnx_tensorrt(),
            'mediapipe_gpu': self._check_mediapipe_gpu(),
        }
        
        # Log detection results
        available = [k for k, v in self.gpu_info.items() if v]
        if available:
            logger.info(f"GPU acceleration available: {', '.join(available)}")
        else:
            logger.info("No GPU acceleration detected - using CPU")
    
    def _check_pytorch_cuda(self) -> bool:
        """Check if PyTorch CUDA is available."""
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                logger.info(f"PyTorch CUDA: {device_count} device(s) - {device_name}")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch CUDA check failed: {e}")
        return False
    
    def _check_pytorch_mps(self) -> bool:
        """Check if PyTorch MPS (Apple Silicon) is available."""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("PyTorch MPS (Apple Silicon) available")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch MPS check failed: {e}")
        return False
    
    def _check_pytorch_rocm(self) -> bool:
        """Check if PyTorch ROCm (AMD GPU) is available."""
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                logger.info("PyTorch ROCm (AMD GPU) available")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch ROCm check failed: {e}")
        return False
    
    def _check_opencv_cuda(self) -> bool:
        """Check if OpenCV CUDA is available."""
        try:
            import cv2
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                logger.info(f"OpenCV CUDA: {device_count} device(s)")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"OpenCV CUDA check failed: {e}")
        return False
    
    def _check_opencv_opencl(self) -> bool:
        """Check if OpenCV OpenCL is available."""
        try:
            import cv2
            if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                logger.info("OpenCV OpenCL available")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"OpenCV OpenCL check failed: {e}")
        return False
    
    def _check_onnx_cuda(self) -> bool:
        """Check if ONNX Runtime CUDA provider is available."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                logger.info("ONNX Runtime CUDA provider available")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"ONNX Runtime CUDA check failed: {e}")
        return False
    
    def _check_onnx_tensorrt(self) -> bool:
        """Check if ONNX Runtime TensorRT provider is available."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'TensorrtExecutionProvider' in available_providers:
                logger.info("ONNX Runtime TensorRT provider available")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"ONNX Runtime TensorRT check failed: {e}")
        return False
    
    def _check_mediapipe_gpu(self) -> bool:
        """Check if MediaPipe GPU delegate is available."""
        try:
            import mediapipe as mp
            # MediaPipe GPU support is typically available on mobile/desktop GPUs
            # Check if we can create a GPU-based solution
            # Note: MediaPipe GPU delegate requires specific initialization
            # This is a basic check - actual GPU usage depends on model initialization
            logger.debug("MediaPipe GPU delegate check - requires model initialization to verify")
            return False  # Conservative - will be enabled during model init if available
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"MediaPipe GPU check failed: {e}")
        return False
    
    def get_whisper_device(self) -> Tuple[str, Optional[str]]:
        """Get the best device for Whisper inference.
        
        Returns:
            Tuple of (device_type, device_name) where device_type is 'cuda', 'mps', 'cpu', etc.
        """
        if self.gpu_info.get('pytorch_cuda'):
            try:
                import torch
                device_name = torch.cuda.get_device_name(0)
                return ('cuda', device_name)
            except:
                pass
        
        if self.gpu_info.get('pytorch_mps'):
            return ('mps', 'Apple Silicon')
        
        if self.gpu_info.get('pytorch_rocm'):
            return ('rocm', 'AMD GPU')
        
        return ('cpu', None)
    
    def get_onnx_providers(self) -> list:
        """Get the best execution providers for ONNX Runtime.
        
        Returns:
            List of provider names in priority order
        """
        providers = []
        
        if self.gpu_info.get('onnx_tensorrt'):
            providers.append('TensorrtExecutionProvider')
        
        if self.gpu_info.get('onnx_cuda'):
            providers.append('CUDAExecutionProvider')
        
        # CPU is always available as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def enable_opencv_gpu(self) -> bool:
        """Enable OpenCV GPU acceleration if available.
        
        Returns:
            True if GPU acceleration was enabled
        """
        if self.gpu_info.get('opencv_cuda'):
            try:
                import cv2
                # OpenCV CUDA is automatically used when available
                logger.info("OpenCV CUDA acceleration enabled")
                return True
            except:
                pass
        
        if self.gpu_info.get('opencv_opencl'):
            try:
                import cv2
                cv2.ocl.setUseOpenCL(True)
                logger.info("OpenCV OpenCL acceleration enabled")
                return True
            except:
                pass
        
        return False
    
    def is_gpu_available(self) -> bool:
        """Check if any GPU acceleration is available."""
        return any(self.gpu_info.values())


# Global GPU detector instance
_gpu_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    """Get or create the global GPU detector instance."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    return _gpu_detector





