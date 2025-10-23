"""
GPU Detection and Configuration for Ready4Hire
Detecta automáticamente GPU disponible y configura el sistema óptimamente.
"""
import subprocess
import os
import logging
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """Tipos de GPU soportados."""
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    APPLE_METAL = "apple_metal"
    CPU_ONLY = "cpu_only"


class GPUDetector:
    """Detecta y configura GPU para inferencia LLM."""
    
    def __init__(self):
        self.gpu_type: GPUType = GPUType.CPU_ONLY
        self.gpu_info: Dict[str, Any] = {}
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detecta el tipo de GPU disponible."""
        # 1. Intentar detectar NVIDIA CUDA
        if self._check_nvidia_cuda():
            self.gpu_type = GPUType.NVIDIA_CUDA
            logger.info("✅ GPU detectada: NVIDIA CUDA")
            return
        
        # 2. Intentar detectar AMD ROCm
        if self._check_amd_rocm():
            self.gpu_type = GPUType.AMD_ROCM
            logger.info("✅ GPU detectada: AMD ROCm")
            return
        
        # 3. Intentar detectar Apple Metal (M1/M2/M3)
        if self._check_apple_metal():
            self.gpu_type = GPUType.APPLE_METAL
            logger.info("✅ GPU detectada: Apple Metal (Apple Silicon)")
            return
        
        # 4. No hay GPU, usar CPU
        self.gpu_type = GPUType.CPU_ONLY
        logger.warning("⚠️ No se detectó GPU. Usando CPU (rendimiento limitado)")
    
    def _check_nvidia_cuda(self) -> bool:
        """Verifica si hay GPU NVIDIA con CUDA."""
        try:
            # Verificar nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(',')
                self.gpu_info = {
                    'name': gpu_info[0].strip() if len(gpu_info) > 0 else 'Unknown',
                    'memory': gpu_info[1].strip() if len(gpu_info) > 1 else 'Unknown',
                    'driver': self._get_nvidia_driver_version()
                }
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"NVIDIA CUDA no disponible: {e}")
        
        return False
    
    def _get_nvidia_driver_version(self) -> str:
        """Obtiene versión del driver NVIDIA."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "Unknown"
    
    def _check_amd_rocm(self) -> bool:
        """Verifica si hay GPU AMD con ROCm."""
        try:
            # Verificar rocm-smi
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'GPU' in result.stdout:
                self.gpu_info = {'type': 'AMD ROCm'}
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"AMD ROCm no disponible: {e}")
        
        return False
    
    def _check_apple_metal(self) -> bool:
        """Verifica si es Apple Silicon con Metal."""
        try:
            # Verificar si es macOS con Apple Silicon
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'Apple' in result.stdout:
                self.gpu_info = {'type': 'Apple Silicon', 'metal': True}
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Apple Metal no disponible: {e}")
        
        return False
    
    def has_gpu(self) -> bool:
        """Retorna True si hay GPU disponible."""
        return self.gpu_type != GPUType.CPU_ONLY
    
    def get_recommended_model(self) -> str:
        """Retorna el modelo LLM recomendado según el hardware."""
        if self.gpu_type == GPUType.CPU_ONLY:
            # CPU: Modelo más pequeño para mejor performance
            return "llama3.2:1b"
        else:
            # GPU: Modelo mediano para mejor calidad
            return "llama3.2:3b"
    
    def get_ollama_env_vars(self) -> Dict[str, str]:
        """Retorna variables de entorno para Ollama según GPU."""
        env_vars = {}
        
        if self.gpu_type == GPUType.NVIDIA_CUDA:
            # Habilitar CUDA para Ollama
            env_vars['CUDA_VISIBLE_DEVICES'] = '0'
            env_vars['OLLAMA_GPU_DRIVER'] = 'cuda'
            
        elif self.gpu_type == GPUType.AMD_ROCM:
            # Habilitar ROCm para Ollama
            env_vars['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
            env_vars['OLLAMA_GPU_DRIVER'] = 'rocm'
            
        elif self.gpu_type == GPUType.APPLE_METAL:
            # Metal está habilitado por defecto en macOS
            env_vars['OLLAMA_GPU_DRIVER'] = 'metal'
        
        else:
            # CPU only
            env_vars['OLLAMA_NUM_GPU'] = '0'
            env_vars['OLLAMA_GPU_DRIVER'] = 'cpu'
        
        return env_vars
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Retorna configuración de performance según hardware."""
        if self.gpu_type == GPUType.CPU_ONLY:
            return {
                'model': 'llama3.2:1b',
                'num_ctx': 2048,  # Contexto más pequeño
                'num_thread': os.cpu_count() or 4,  # Usar todos los cores
                'num_gpu': 0,
                'expected_latency_ms': 30000,  # 30s en CPU
                'batch_size': 1,
                'use_cache': True  # Cache agresivo en CPU
            }
        else:
            return {
                'model': 'llama3.2:3b',
                'num_ctx': 4096,  # Contexto más grande
                'num_thread': 4,
                'num_gpu': 1,
                'expected_latency_ms': 3000,  # 3s en GPU
                'batch_size': 4,
                'use_cache': False  # Menos cache en GPU (más rápido)
            }
    
    def print_info(self):
        """Imprime información de GPU detectada."""
        print(f"\n{'='*70}")
        print(f"  🖥️  CONFIGURACIÓN DE HARDWARE DETECTADA")
        print(f"{'='*70}")
        
        if self.has_gpu():
            print(f"  Tipo:                 {GREEN}✅ GPU DISPONIBLE{NC}")
            print(f"  GPU Type:             {self.gpu_type.value}")
            
            if self.gpu_info:
                for key, value in self.gpu_info.items():
                    print(f"  {key.capitalize():20s}  {value}")
        else:
            print(f"  Tipo:                 {YELLOW}⚠️  CPU ONLY{NC}")
            print(f"  Cores:                {os.cpu_count() or 'Unknown'}")
        
        config = self.get_performance_config()
        print(f"\n  {'CONFIGURACIÓN OPTIMIZADA':^68}")
        print(f"  {'-'*68}")
        print(f"  Modelo LLM:           {config['model']}")
        print(f"  Latencia esperada:    {config['expected_latency_ms']/1000:.1f}s")
        print(f"  Contexto tokens:      {config['num_ctx']}")
        print(f"  Threads:              {config['num_thread']}")
        print(f"  Cache agresivo:       {'Sí' if config['use_cache'] else 'No'}")
        print(f"{'='*70}\n")


# Colores para output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'


# Instancia global
gpu_detector = GPUDetector()


def get_gpu_detector() -> GPUDetector:
    """Retorna la instancia global del detector de GPU."""
    return gpu_detector


if __name__ == "__main__":
    # Test del detector
    logging.basicConfig(level=logging.INFO)
    detector = GPUDetector()
    detector.print_info()
    
    print(f"\nVariables de entorno para Ollama:")
    for key, value in detector.get_ollama_env_vars().items():
        print(f"  export {key}={value}")

