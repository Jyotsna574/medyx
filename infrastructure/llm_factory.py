"""
LLM Backend Factory for MedicalAgentDiagnosis-MAD.

Provides a unified factory function to instantiate LLM backends based on
configuration. Supports both cloud APIs (Gemini, OpenAI, Anthropic) via
CAMEL-AI and local HuggingFace models with 4-bit quantization.

Usage:
    from infrastructure.llm_factory import get_llm_backend
    
    # Returns CAMEL-AI compatible model backend based on config
    model = get_llm_backend()
    
    # Use with CAMEL-AI ChatAgent
    agent = ChatAgent(system_message=..., model=model)

Environment Variables:
    ACTIVE_PROVIDER: 'gemini', 'openai', 'anthropic', 'local', 'ollama'
    LOCAL_ACTIVE_MODEL: 'biomistral_7b', 'med42_8b', 'meditron_70b', etc.
    LOCAL_MODEL_PATH: Path to pre-downloaded model (skips HuggingFace download)
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from camel.models.base_model import BaseModelBackend
from camel.types import ModelType

from config.settings import model_config


# Lazy imports for heavy dependencies
_transformers = None
_torch = None


def _lazy_import_torch():
    """Lazily import PyTorch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _lazy_import_transformers():
    """Lazily import transformers and related libraries."""
    global _transformers
    if _transformers is None:
        try:
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                GenerationConfig,
            )
            _transformers = {
                "module": transformers,
                "AutoModelForCausalLM": AutoModelForCausalLM,
                "AutoTokenizer": AutoTokenizer,
                "BitsAndBytesConfig": BitsAndBytesConfig,
                "GenerationConfig": GenerationConfig,
            }
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers not installed. "
                "Install with: pip install transformers accelerate bitsandbytes"
            ) from e
    return _transformers


@dataclass
class ChatMessage:
    """Simple chat message structure compatible with CAMEL-AI."""
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletionResponse:
    """Response structure mimicking CAMEL-AI/OpenAI format."""
    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    
    @property
    def msg(self):
        """Compatibility property for CAMEL-AI's response.msg access."""
        return self


class HuggingFaceTokenCounter:
    """
    Token counter for HuggingFace models, compatible with CAMEL-AI's expectations.
    
    Implements the count_tokens_from_messages interface required by CAMEL-AI's
    memory system for context management.
    """
    
    def __init__(self, tokenizer: Any):
        """Initialize with a HuggingFace tokenizer."""
        self._tokenizer = tokenizer
    
    def count_tokens_from_messages(self, messages: List[Any]) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of messages (dicts or CAMEL message objects).
            
        Returns:
            Total token count.
        """
        if self._tokenizer is None:
            return 0
        
        total_tokens = 0
        for msg in messages:
            content = ""
            if isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "")
            elif hasattr(msg, "content"):
                content = str(msg.content)
                role = getattr(msg, "role", "")
            else:
                content = str(msg)
                role = ""
            
            # Tokenize and count
            text = f"{role}: {content}" if role else content
            try:
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
            except Exception:
                # Fallback: estimate 4 chars per token
                total_tokens += len(text) // 4
        
        # Add overhead for message formatting (approx 4 tokens per message)
        total_tokens += len(messages) * 4
        
        return total_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self._tokenizer is None:
            return len(text) // 4
        
        try:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            return len(text) // 4


class HuggingFaceLocalBackend:
    """
    Local HuggingFace model backend compatible with CAMEL-AI ChatAgent.
    
    Loads models with BitsAndBytes 4-bit quantization for efficient GPU usage.
    Implements the same interface expected by CAMEL-AI's ChatAgent.
    
    Args:
        model_path_or_id: HuggingFace repo ID or local path to model.
        load_in_4bit: Enable 4-bit quantization (default: True).
        device_map: Device mapping strategy (default: "auto").
        torch_dtype: PyTorch dtype for model weights.
        bnb_4bit_compute_dtype: Compute dtype for 4-bit ops.
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4').
        use_double_quant: Enable nested quantization.
        generation_params: Default generation parameters.
    """
    
    def __init__(
        self,
        model_path_or_id: str,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        use_double_quant: bool = True,
        generation_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_path_or_id = model_path_or_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype_str = torch_dtype
        self.bnb_4bit_compute_dtype_str = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_double_quant = use_double_quant
        self.generation_params = generation_params or {
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
        }
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
        
        print(f"[HuggingFaceLocalBackend] Initialized")
        print(f"  Model: {model_path_or_id}")
        print(f"  4-bit Quantization: {load_in_4bit}")
        print(f"  Device Map: {device_map}")
    
    def _get_torch_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype."""
        torch = _lazy_import_torch()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def load(self) -> bool:
        """Load the model and tokenizer into memory."""
        if self._loaded:
            return True
        
        torch = _lazy_import_torch()
        tf = _lazy_import_transformers()
        
        try:
            print(f"[HuggingFaceLocalBackend] Loading model: {self.model_path_or_id}")
            
            # Configure 4-bit quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = tf["BitsAndBytesConfig"](
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._get_torch_dtype(self.bnb_4bit_compute_dtype_str),
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.use_double_quant,
                )
                print(f"  BitsAndBytes Config: 4-bit, {self.bnb_4bit_quant_type}, double_quant={self.use_double_quant}")
            
            # Load tokenizer
            print("  Loading tokenizer...")
            self._tokenizer = tf["AutoTokenizer"].from_pretrained(
                self.model_path_or_id,
                trust_remote_code=True,
            )
            
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            print("  Loading model weights...")
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "device_map": self.device_map,
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = self._get_torch_dtype(self.torch_dtype_str)
            
            self._model = tf["AutoModelForCausalLM"].from_pretrained(**model_kwargs)
            
            self._loaded = True
            
            # Report memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU Memory Used: {allocated:.2f} GB")
            
            print("[HuggingFaceLocalBackend] Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[HuggingFaceLocalBackend] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        Run inference on the model with chat messages.
        
        This method is compatible with CAMEL-AI's expected interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional generation parameters.
            
        Returns:
            ChatCompletionResponse with generated content.
        """
        if not self._loaded:
            if not self.load():
                return ChatCompletionResponse(
                    content="Error: Failed to load model",
                    finish_reason="error",
                )
        
        torch = _lazy_import_torch()
        
        # Merge generation params with kwargs
        gen_params = {**self.generation_params, **kwargs}
        
        try:
            # Format messages for the model
            prompt = self._format_messages(messages)
            
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self._model.device)
            
            # Generate
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_params.get("max_new_tokens", 2048),
                    temperature=gen_params.get("temperature", 0.7),
                    do_sample=gen_params.get("do_sample", True),
                    top_p=gen_params.get("top_p", 0.95),
                    repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            
            return ChatCompletionResponse(
                content=response_text.strip(),
                usage={
                    "prompt_tokens": input_length,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_length + len(generated_tokens),
                },
            )
            
        except Exception as e:
            print(f"[HuggingFaceLocalBackend] Generation error: {e}")
            import traceback
            traceback.print_exc()
            return ChatCompletionResponse(
                content=f"Error during generation: {str(e)}",
                finish_reason="error",
            )
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt string.
        
        Uses the model's chat template if available, otherwise falls back
        to a generic format.
        """
        # Try to use the tokenizer's chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # Fallback: manual formatting
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"### System:\n{content}\n")
            elif role == "user":
                formatted_parts.append(f"### User:\n{content}\n")
            elif role == "assistant":
                formatted_parts.append(f"### Assistant:\n{content}\n")
        
        formatted_parts.append("### Assistant:\n")
        return "\n".join(formatted_parts)
    
    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        torch = _lazy_import_torch()
        
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("[HuggingFaceLocalBackend] Model unloaded")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this backend."""
        torch = _lazy_import_torch()
        
        info = {
            "backend": "HuggingFaceLocal",
            "model_id": self.model_path_or_id,
            "loaded": self._loaded,
            "4bit_quantization": self.load_in_4bit,
            "device_map": self.device_map,
        }
        
        if self._loaded and torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1024**3, 2
            )
        
        return info


class HuggingFaceModelBackend(BaseModelBackend):
    """
    CAMEL-AI compatible model backend for local HuggingFace models.
    
    Inherits from CAMEL-AI's BaseModelBackend to ensure full compatibility
    with ChatAgent. Wraps our HuggingFaceLocalBackend for actual inference.
    """
    
    def __init__(self, backend: HuggingFaceLocalBackend):
        """Initialize the CAMEL-compatible backend wrapper."""
        # Store backend reference before calling super().__init__
        self._hf_backend = backend
        self._hf_token_counter = None
        
        # Call parent __init__ with required parameters
        super().__init__(
            model_type=ModelType.STUB,
            model_config_dict={
                "model_type": "HuggingFaceLocal",
                "model_path": backend.model_path_or_id,
                "load_in_4bit": backend.load_in_4bit,
            },
            api_key=None,
            url=None,
            token_counter=None,  # We'll set this after loading tokenizer
            timeout=300.0,  # 5 minute timeout for local models
            max_retries=1,
            extract_thinking_from_response=False,  # Not needed for local models
        )
        
        # Pre-load tokenizer to enable token counting before first inference
        self._ensure_tokenizer_loaded()
    
    def _ensure_tokenizer_loaded(self) -> None:
        """Ensure tokenizer is loaded for token counting."""
        if self._hf_backend._tokenizer is None:
            tf = _lazy_import_transformers()
            print(f"[HuggingFaceModelBackend] Loading tokenizer for token counting...")
            try:
                self._hf_backend._tokenizer = tf["AutoTokenizer"].from_pretrained(
                    self._hf_backend.model_path_or_id,
                    trust_remote_code=True,
                )
                if self._hf_backend._tokenizer.pad_token is None:
                    self._hf_backend._tokenizer.pad_token = self._hf_backend._tokenizer.eos_token
                print(f"[HuggingFaceModelBackend] Tokenizer loaded successfully")
            except Exception as e:
                print(f"[HuggingFaceModelBackend] Warning: Could not load tokenizer: {e}")
        
        # Create token counter with the tokenizer (use our own attribute to avoid conflict)
        if self._hf_token_counter is None and self._hf_backend._tokenizer is not None:
            self._hf_token_counter = HuggingFaceTokenCounter(self._hf_backend._tokenizer)
    
    
    def _extract_role(self, msg: Any) -> str:
        """Extract role string from various CAMEL-AI message formats."""
        if hasattr(msg, "role"):
            role = msg.role
            if isinstance(role, str):
                return role.lower()
            if hasattr(role, "value"):
                return str(role.value).lower()
            if hasattr(role, "name"):
                return str(role.name).lower()
            return str(role).lower()
        
        if hasattr(msg, "role_name"):
            role_name = msg.role_name
            if isinstance(role_name, str):
                role_lower = role_name.lower()
                if "system" in role_lower:
                    return "system"
                elif "assistant" in role_lower:
                    return "assistant"
                else:
                    return "user"
            return "user"
        
        if isinstance(msg, dict):
            return msg.get("role", "user")
        
        return "user"
    
    def _extract_content(self, msg: Any) -> str:
        """Extract content string from various CAMEL-AI message formats."""
        if hasattr(msg, "content"):
            return str(msg.content)
        if isinstance(msg, dict):
            return msg.get("content", str(msg))
        return str(msg)
    
    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Normalize CAMEL-AI messages to simple dicts."""
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                if not isinstance(role, str):
                    role = str(role).lower()
                normalized.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })
            else:
                normalized.append({
                    "role": self._extract_role(msg),
                    "content": self._extract_content(msg),
                })
        return normalized
    
    def _make_chat_completion(self, response: ChatCompletionResponse) -> Any:
        """Convert our response to OpenAI ChatCompletion format that CAMEL expects."""
        from types import SimpleNamespace
        
        message = SimpleNamespace(
            role="assistant",
            content=response.content,
            function_call=None,
            tool_calls=None,
        )
        
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason=response.finish_reason,
        )
        
        usage = SimpleNamespace(
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.usage.get("total_tokens", 0),
        )
        
        completion = SimpleNamespace(
            id="hf-local-completion",
            object="chat.completion",
            created=0,
            model=self._hf_backend.model_path_or_id,
            choices=[choice],
            usage=usage,
        )
        
        return completion
    
    def _run(self, messages: List[Any]) -> Any:
        """
        Required abstract method from BaseModelBackend.
        Synchronous inference method called by CAMEL-AI.
        """
        normalized_messages = self._normalize_messages(messages)
        response = self._hf_backend.run(normalized_messages)
        return self._make_chat_completion(response)
    
    async def _arun(self, messages: List[Any]) -> Any:
        """
        Required abstract method from BaseModelBackend.
        Async inference method - runs sync version in executor for compatibility.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, messages)
    
    def run(self, messages: List[Any]) -> Any:
        """Public run method that delegates to _run."""
        return self._run(messages)
    
    def check_model_config(self) -> None:
        """Check if model configuration is valid."""
        pass
    
    @property
    def stream(self) -> bool:
        """Return whether streaming is enabled."""
        return False
    
    @property 
    def token_counter(self) -> Any:
        """Return token counter for CAMEL-AI memory management."""
        if self._hf_token_counter is None:
            self._ensure_tokenizer_loaded()
        return self._hf_token_counter
    
    def unload(self) -> None:
        """Unload the underlying backend."""
        self._hf_backend.unload()
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return self._hf_backend.get_info()


def get_llm_backend(provider: Optional[str] = None):
    """
    Factory function to get the appropriate LLM backend.
    
    Returns a model backend compatible with CAMEL-AI's ChatAgent.
    The backend is determined by configuration (models.yaml) and
    environment variables (ACTIVE_PROVIDER, LOCAL_ACTIVE_MODEL, LOCAL_MODEL_PATH).
    
    Args:
        provider: Override the active provider. If None, uses config.
        
    Returns:
        Model backend compatible with CAMEL-AI ChatAgent.
        
    Raises:
        ValueError: If required API keys or configurations are missing.
        
    Example:
        # Use default provider from config
        model = get_llm_backend()
        agent = ChatAgent(system_message=..., model=model)
        
        # Override provider
        model = get_llm_backend(provider="local")
    """
    # Reload config to pick up any env var changes
    model_config.reload()
    
    active_provider = provider or model_config.active_provider
    
    print(f"[LLM Factory] Creating backend for provider: {active_provider}")
    
    if active_provider == "local":
        return _create_huggingface_backend()
    else:
        return _create_camel_backend(active_provider)


def _create_camel_backend(provider: str):
    """Create a CAMEL-AI model backend for cloud APIs."""
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    
    provider_config = model_config.get_provider_config(provider)
    api_key = model_config.get_api_key(provider)
    
    if not api_key:
        api_key_env = provider_config.get("api_key_env", "UNKNOWN")
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set the {api_key_env} environment variable."
        )
    
    # Map provider to CAMEL platform and model types
    platform_map = {
        "gemini": ModelPlatformType.GEMINI,
        "openai": ModelPlatformType.OPENAI,
        "openai_turbo": ModelPlatformType.OPENAI,
        "anthropic": ModelPlatformType.ANTHROPIC,
        "ollama": ModelPlatformType.OLLAMA,
    }
    
    # Model type mapping - use string-based lookup for flexibility
    model_name = provider_config.get("model", "")
    
    platform = platform_map.get(provider)
    if platform is None:
        raise ValueError(f"Unknown provider: {provider}")
    
    # For Gemini, use specific model type
    if provider == "gemini":
        model_type = ModelType.GEMINI_2_0_FLASH
    elif provider in ("openai", "openai_turbo"):
        # Map OpenAI models
        if "gpt-4-turbo" in model_name:
            model_type = ModelType.GPT_4_TURBO
        elif "gpt-4o-mini" in model_name:
            model_type = ModelType.GPT_4O_MINI
        elif "gpt-4o" in model_name:
            model_type = ModelType.GPT_4O
        else:
            model_type = ModelType.GPT_4O_MINI
    elif provider == "anthropic":
        model_type = ModelType.CLAUDE_3_5_SONNET
    else:
        # Default fallback
        model_type = ModelType.GEMINI_2_0_FLASH
    
    print(f"[LLM Factory] Creating CAMEL backend: {platform} / {model_type}")
    
    return ModelFactory.create(
        model_platform=platform,
        model_type=model_type,
        api_key=api_key,
    )


def _create_huggingface_backend() -> HuggingFaceModelBackend:
    """Create a HuggingFace local model backend compatible with CAMEL-AI."""
    local_config = model_config.get_local_model_config()
    
    if not local_config:
        raise ValueError(
            "Local model configuration not found. "
            "Check the 'local' provider section in config/models.yaml"
        )
    
    # Determine model path/ID
    local_path = model_config.get_local_model_path()
    if local_path:
        model_path_or_id = local_path
        print(f"[LLM Factory] Using local model path: {local_path}")
    else:
        model_path_or_id = local_config.get("repo_id", "")
        if not model_path_or_id:
            raise ValueError(
                "No repo_id specified for local model and LOCAL_MODEL_PATH not set."
            )
        print(f"[LLM Factory] Using HuggingFace repo: {model_path_or_id}")
    
    # Create the underlying HuggingFace backend
    backend = HuggingFaceLocalBackend(
        model_path_or_id=model_path_or_id,
        load_in_4bit=local_config.get("load_in_4bit", True),
        device_map=local_config.get("device_map", "auto"),
        torch_dtype=local_config.get("torch_dtype", "bfloat16"),
        bnb_4bit_compute_dtype=local_config.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_quant_type=local_config.get("bnb_4bit_quant_type", "nf4"),
        use_double_quant=local_config.get("use_double_quant", True),
        generation_params=local_config.get("generation_params", {}),
    )
    
    # Wrap in CAMEL-AI compatible BaseModelBackend
    return HuggingFaceModelBackend(backend)


def get_provider_info() -> Dict[str, Any]:
    """
    Get information about the current provider configuration.
    
    Returns:
        Dictionary with provider details, useful for debugging.
    """
    return {
        "active_provider": model_config.active_provider,
        "is_local": model_config.is_local_provider,
        "temperature": model_config.get_temperature(),
        "max_tokens": model_config.get_max_tokens(),
        "local_model_config": model_config.get_local_model_config() if model_config.is_local_provider else None,
        "local_model_path": model_config.get_local_model_path(),
    }
