#!/usr/bin/env python
"""
Dummy run test - validates imports, config, and LLM factory without GPU.
No model loading, no API calls. Uses mocks where needed.
"""
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all key modules import successfully."""
    print("1. Testing imports...")
    try:
        from config.settings import model_config, Settings
        from core.schemas import PatientCase
        print("   [OK] config, core.schemas")
    except Exception as e:
        print(f"   [FAIL] Import failed: {e}")
        return False
    return True


def test_config():
    """Test configuration loading."""
    print("2. Testing config...")
    try:
        os.environ["ACTIVE_PROVIDER"] = "local"
        os.environ["LOCAL_ACTIVE_MODEL"] = "med42_8b"
        os.environ["LOCAL_MODEL_PATH"] = "/fake/path/for/test"
        
        from config.settings import model_config
        model_config.reload()
        
        assert model_config.active_provider == "local"
        assert model_config.is_local_provider is True
        
        local_config = model_config.get_local_model_config()
        assert local_config, "local config should not be empty"
        assert "generation_params" in local_config
        assert "load_in_4bit" in local_config
        
        path = model_config.get_local_model_path()
        assert path == "/fake/path/for/test"
        
        print("   [OK] Config (active_provider=local, path from env)")
    except Exception as e:
        print(f"   [FAIL] Config failed: {e}")
        return False
    return True


def test_llm_factory_with_mock():
    """Test LLM factory creates correct backend type (with mocked tokenizer load)."""
    print("3. Testing LLM factory (mocked)...")
    try:
        from unittest.mock import patch, MagicMock

        mod = _import_llm_factory()
        get_llm_backend = mod.get_llm_backend
        get_provider_info = mod.get_provider_info
        HuggingFaceModelBackend = mod.HuggingFaceModelBackend

        with patch.dict(os.environ, {
            "ACTIVE_PROVIDER": "local",
            "LOCAL_ACTIVE_MODEL": "med42_8b",
            "LOCAL_MODEL_PATH": "/fake/test/path",
        }):
            with patch.object(mod, "_lazy_import_transformers") as mock_tf:
                mock_tokenizer = MagicMock()
                mock_tokenizer.pad_token = None
                mock_tokenizer.eos_token = "<eos>"
                mock_tokenizer.encode = lambda x, **kw: [1, 2, 3]
                
                mock_auto = MagicMock()
                mock_auto.from_pretrained = MagicMock(return_value=mock_tokenizer)
                mock_tf.return_value = {"AutoTokenizer": mock_auto}

                mod.model_config.reload()
                provider_info = get_provider_info()
                assert provider_info["active_provider"] == "local"
                assert provider_info["is_local"] is True
                print("   [OK] get_provider_info()")
                
                backend = get_llm_backend()
                assert backend is not None
                assert isinstance(backend, HuggingFaceModelBackend)
                print("   [OK] get_llm_backend() returns HuggingFaceModelBackend")
                
                token_counter = backend.token_counter
                assert token_counter is not None
                count = token_counter.count_tokens_from_messages([{"role": "user", "content": "hello"}])
                assert count >= 0
                print(f"   [OK] token_counter works (count={count})")
                
    except Exception as e:
        print(f"   [FAIL] LLM factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def _import_llm_factory():
    """Import llm_factory without loading full infrastructure (neo4j, vision)."""
    import importlib.util
    if "infrastructure.llm_factory" in sys.modules:
        return sys.modules["infrastructure.llm_factory"]
    spec = importlib.util.spec_from_file_location(
        "llm_factory",
        os.path.join(os.path.dirname(__file__), "infrastructure", "llm_factory.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infrastructure.llm_factory"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_huggingface_backend_init():
    """Test HuggingFaceLocalBackend init (no load)."""
    print("4. Testing HuggingFaceLocalBackend init...")
    try:
        mod = _import_llm_factory()
        HuggingFaceLocalBackend = mod.HuggingFaceLocalBackend
        
        backend = HuggingFaceLocalBackend(
            model_path_or_id="/fake/path",
            load_in_4bit=True,
            device_map="auto",
            generation_params={"temperature": 0.7, "max_new_tokens": 256},
        )
        
        assert backend.model_path_or_id == "/fake/path"
        assert backend.load_in_4bit is True
        assert backend.generation_params["max_new_tokens"] == 256
        assert backend._loaded is False
        
        print("   [OK] HuggingFaceLocalBackend init")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False
    return True


def test_chat_completion_response():
    """Test ChatCompletionResponse and _make_chat_completion."""
    print("5. Testing ChatCompletionResponse format...")
    try:
        mod = _import_llm_factory()
        ChatCompletionResponse = mod.ChatCompletionResponse

        resp = ChatCompletionResponse(
            content="Test response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        
        assert resp.content == "Test response"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage.get("completion_tokens", 0) == 5
        
        print("   [OK] ChatCompletionResponse")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False
    return True


def main():
    print("=" * 60)
    print("DUMMY RUN TEST - No GPU, No API keys required")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_config())
    results.append(test_llm_factory_with_mock())
    results.append(test_huggingface_backend_init())
    results.append(test_chat_completion_response())
    
    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED - System ready to go")
        return 0
    else:
        print("SOME TESTS FAILED - Check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
