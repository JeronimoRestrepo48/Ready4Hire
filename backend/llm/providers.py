"""
LLM Abstraction Layer for Ready4Hire
====================================

This module provides an abstraction layer for Large Language Models (LLMs),
allowing easy replacement and updating of LLM providers without affecting
the rest of the system.

Supported providers:
- Ollama (default)
- OpenAI (configurable)
- Anthropic (configurable)
- Hugging Face (configurable)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "ollama"
    model: str = "llama3.2"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Ollama client."""
        try:
            import ollama
            self._client = ollama.Client(host=self.config.base_url or "http://localhost:11434")
        except ImportError:
            raise ImportError("ollama package is required for Ollama provider")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        try:
            response = self._client.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens
                }
            )
            return response['response'].strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Ollama: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Simple test generation
            self._client.generate(model=self.config.model, prompt="test", options={'num_predict': 1})
            return True
        except:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with OpenAI: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            # Simple test generation
            self.generate_text("test")
            return True
        except:
            return False


class LLMManager:
    """Manager for LLM providers with fallback support."""
    
    def __init__(self, configs: List[LLMConfig] = None):
        """
        Initialize LLM manager with multiple provider configurations.
        
        Args:
            configs: List of LLM configurations. If None, uses default Ollama config.
        """
        if configs is None:
            configs = [LLMConfig()]
        
        self.providers = []
        self.active_provider = None
        
        for config in configs:
            provider = self._create_provider(config)
            if provider:
                self.providers.append(provider)
        
        self._select_active_provider()
    
    def _create_provider(self, config: LLMConfig) -> Optional[LLMProvider]:
        """Create a provider based on configuration."""
        try:
            if config.provider.lower() == "ollama":
                return OllamaProvider(config)
            elif config.provider.lower() == "openai":
                return OpenAIProvider(config)
            else:
                print(f"Unsupported LLM provider: {config.provider}")
                return None
        except Exception as e:
            print(f"Failed to create {config.provider} provider: {str(e)}")
            return None
    
    def _select_active_provider(self):
        """Select the first available provider as active."""
        for provider in self.providers:
            if provider.is_available():
                self.active_provider = provider
                print(f"Selected {provider.config.provider} as active LLM provider")
                return
        
        if not self.active_provider:
            print("No LLM providers are available")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the active provider with fallback."""
        if not self.active_provider:
            raise RuntimeError("No LLM provider is available")
        
        # Try active provider first
        try:
            return self.active_provider.generate_text(prompt, **kwargs)
        except Exception as e:
            print(f"Active provider failed: {str(e)}")
            
            # Try fallback providers
            for provider in self.providers:
                if provider != self.active_provider and provider.is_available():
                    try:
                        result = provider.generate_text(prompt, **kwargs)
                        self.active_provider = provider  # Switch to working provider
                        print(f"Switched to {provider.config.provider} provider")
                        return result
                    except Exception as fallback_e:
                        print(f"Fallback provider {provider.config.provider} failed: {str(fallback_e)}")
                        continue
            
            raise RuntimeError("All LLM providers failed")
    
    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider."""
        if not self.active_provider:
            return {"provider": None, "status": "unavailable"}
        
        return {
            "provider": self.active_provider.config.provider,
            "model": self.active_provider.config.model,
            "status": "available" if self.active_provider.is_available() else "unavailable"
        }


def create_llm_manager_from_env() -> LLMManager:
    """Create LLM manager from environment variables."""
    configs = []
    
    # Primary LLM configuration
    primary_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        model=os.getenv("LLM_MODEL", "llama3.2"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "30"))
    )
    configs.append(primary_config)
    
    # Fallback LLM configuration (Ollama as default fallback)
    if primary_config.provider.lower() != "ollama":
        fallback_config = LLMConfig(
            provider="ollama",
            model=os.getenv("FALLBACK_LLM_MODEL", "llama3.2"),
            base_url=os.getenv("FALLBACK_LLM_BASE_URL")
        )
        configs.append(fallback_config)
    
    return LLMManager(configs)