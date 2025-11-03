#!/usr/bin/env python3
"""Debug script to reproduce the actual text generation failure."""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from server.llm_config.llm_models import get_llm_config, ModelConfig
from server.services.model_registry_service import get_model_registry
from server.services.text_generation_service import TextGenerationService
from server.llm.client import chat_complete_with_raw


def debug_model_loading():
    """Test model configuration loading."""
    print("=== Testing Model Configuration Loading ===")
    
    try:
        config = get_llm_config()
        print(f"✅ Config loaded successfully")
        print(f"   Default model: {config.default_model}")
        print(f"   Total models: {len(config.models)}")
        print(f"   Fallback chain: {config.fallback_chain}")
        
        # Test a specific model
        kimi_model = config.get_model_by_id("kimi-k2-0905")
        print(f"✅ Kimi model found: {kimi_model.display_name}")
        print(f"   API key env: {kimi_model.api_key_env}")
        print(f"   Base URL: {kimi_model.base_url}")
        print(f"   API key value: {'***SET***' if kimi_model.get_api_key() else 'NOT SET'}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def debug_model_registry():
    """Test model registry service."""
    print("\n=== Testing Model Registry Service ===")
    
    try:
        registry = get_model_registry()
        user_tier = "Free"
        
        available_models = registry.get_available_models(user_tier)
        print(f"✅ Found {len(available_models)} models for tier '{user_tier}'")
        
        for model in available_models:
            print(f"   - {model.display_name} ({model.id})")
        
        default_model = registry.get_default_model(user_tier)
        print(f"✅ Default model: {default_model.display_name}")
        
        return True
    except Exception as e:
        print(f"❌ Model registry failed: {e}")
        return False


def debug_llm_call():
    """Test actual LLM API call."""
    print("\n=== Testing LLM API Call ===")
    
    try:
        registry = get_model_registry()
        model = registry.get_model_by_id("kimi-k2-0905")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in exactly one word."}
        ]
        
        print(f"Attempting call to {model.display_name}...")
        print(f"Base URL: {model.base_url}")
        print(f"Model: {model.model}")
        
        text, resp_dict = chat_complete_with_raw(
            messages=messages,
            model_config=model,
            max_tokens=100
        )
        
        if text and text.strip():
            provider = "openrouter" if "openrouter" in model.base_url else "local"
            print(f"✅ LLM call succeeded")
            print(f"   Response: {text.strip()[:100]}...")
            print(f"   Provider: {provider}")
            print(f"   Model used: {model.model}")
            return True
        else:
            print(f"❌ LLM call returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        print(f"   Exception type: {type(e).__name__}")
        # Try to get more details about the exception
        if hasattr(e, 'response'):
            print(f"   Response status: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"   Response text: {getattr(e.response, 'text', 'N/A')[:200]}...")
        return False


def debug_text_generation_service():
    """Test text generation service with placeholder database."""
    print("\n=== Testing Text Generation Service ===")
    
    try:
        from unittest.mock import MagicMock
        from sqlalchemy.orm import Session
        
        # Mock database sessions to avoid needing real DB
        mock_account_db = MagicMock(spec=Session)
        mock_global_db = MagicMock(spec=Session)
        
        # Mock account lookup
        from server.auth import Account
        mock_account = MagicMock()
        mock_account.subscription_tier = "Free"
        mock_global_db.query.return_value.filter.return_value.first.return_value = mock_account
        
        service = TextGenerationService()
        
        # Just test the model selection and LLM call logic
        registry = get_model_registry()
        available_models = registry.get_available_models("Free")
        print(f"✅ Available models for Free tier: {len(available_models)}")
        
        if not available_models:
            print("❌ No models available for Free tier!")
            return False
        
        # Try the first available model
        test_model = available_models[0]
        print(f"   Testing with: {test_model.display_name}")
        
        # Mock the database operations
        from unittest.mock import patch
        from pathlib import Path
        
        with patch('server.services.text_generation_service.llm_call_and_log_to_file') as mock_llm_call:
            # Simulate successful LLM response
            mock_llm_call.return_value = (
                "Hello world! This is a test generated text.",
                {"usage": {"total_tokens": 10}},
                "openrouter",
                test_model.model
            )
            
            # Create the job directory
            job_dir = Path("/tmp/debug_text_gen")
            job_dir.mkdir(parents=True, exist_ok=True)
            
            result = service.generate_text_content(
                mock_account_db,
                mock_global_db,
                account_id=123,
                lang="es",
                text_id=456,
                job_dir=job_dir,
                messages=[
                    {"role": "system", "content": "Generate a short Spanish text."},
                    {"role": "user", "content": "Create text"}
                ]
            )
            
            if result.success:
                print(f"✅ Text generation succeeded")
                print(f"   Generated text: {result.text[:100]}...")
                print(f"   Provider: {result.provider}")
                print(f"   Model: {result.model}")
                return True
            else:
                print(f"❌ Text generation failed: {result.error}")
                return False
                
    except Exception as e:
        print(f"❌ Text generation service test failed: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Debugging Text Generation Failure")
    print("=" * 50)
    
    success = True
    success &= debug_model_loading()
    success &= debug_model_registry()
    success &= debug_llm_call()
    success &= debug_text_generation_service()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed - text generation should work!")
    else:
        print("❌ Some tests failed - this explains the generation issue")
    
    sys.exit(0 if success else 1)
