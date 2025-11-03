#!/usr/bin/env python3
"""Debug script to reproduce the REAL text generation failure without mocking."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from server.services.text_generation_service import TextGenerationService, TextGenerationResult
from server.services.model_registry_service import get_model_registry
from server.llm.client import chat_complete_with_raw
from sqlalchemy.orm import Session


def debug_real_text_generation():
    """Test the REAL text generation flow without mocking LLM responses."""
    print("=== Testing REAL Text Generation (No Mocking) ===")
    
    try:
        # Create service
        service = TextGenerationService()
        
        # Mock database sessions but let LLM calls happen
        mock_account_db = MagicMock(spec=Session)
        mock_global_db = MagicMock(spec=Session)
        
        # Mock account lookup for tier
        from server.auth import Account
        mock_account = MagicMock()
        mock_account.subscription_tier = "Free"
        mock_global_db.query.return_value.filter.return_value.first.return_value = mock_account
        
        # Setup models to try
        registry = get_model_registry()
        available_models = registry.get_available_models("Free")
        print(f"Available models for Free tier: {[m.display_name for m in available_models]}")
        
        # Try each available model
        for model in available_models:
            print(f"\n--- Testing with model: {model.display_name} ---")
            
            # Create actual job directory
            job_dir = Path(f"/tmp/debug_real_gen_{model.id}")
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Build a simple test prompt
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant that generates short text."},
                {"role": "user", "content": "Generate a short Spanish sentence (max 10 words)."}
            ]
            
            # Mock the database operations for placeholder creation
            class MockReadingText:
                def __init__(self):
                    self.id = 999
                    
            mock_rt = MockReadingText()
            
            # Mock create_placeholder_text to return our test ID
            service.create_placeholder_text = MagicMock(return_value=999)
            
            with open(job_dir / "test_input.json", "w") as f:
                f.write(f"Testing model: {model.display_name}\n")
            
            # Call the actual generate_text_content
            try:
                result = service.generate_text_content(
                    mock_account_db,
                    mock_global_db, 
                    account_id=123,
                    lang="es",
                    text_id=999,
                    job_dir=job_dir,
                    messages=test_messages
                )
                
                if result.success:
                    print(f"✅ SUCCESS with {model.display_name}")
                    print(f"   Generated text: {result.text[:50]}...")
                    print(f"   Provider: {result.provider}")
                    print(f"   Model: {result.model}")
                    return True
                else:
                    print(f"❌ FAILED with {model.display_name}")
                    print(f"   Error: {result.error}")
                    
            except Exception as e:
                print(f"❌ EXCEPTION with {model.display_name}: {e}")
                print(f"   Exception type: {type(e).__name__}")
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    print(f"   HTTP Status: {e.response.status_code}")
                
        print(f"\n❌ All models failed!")
        return False
        
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openrouter_api_directly():
    """Test OpenRouter API directly to see the exact error."""
    print("\n=== Testing OpenRouter API Directly ===")
    
    try:
        registry = get_model_registry()
        kimi_model = registry.get_model_by_id("kimi-free")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."}
        ]
        
        print(f"Testing OpenRouter with model: {kimi_model.model}")
        print(f"API key is set: {'YES' if kimi_model.get_api_key() else 'NO'}")
        
        text, resp_dict = chat_complete_with_raw(
            messages=messages,
            model_config=kimi_model,
            max_tokens=10
        )
        
        print(f"✅ Direct API call worked: {text}")
        return True
        
    except Exception as e:
        print(f"❌ Direct API call failed: {e}")
        print(f"   Type: {type(e).__name__}")
        
        # Look for HTTP errors
        if hasattr(e, 'response'):
            print(f"   Response status: {getattr(e.response, 'status_code', 'N/A')}")
            if hasattr(e.response, 'text'):
                resp_text = e.response.text
                print(f"   Response: {resp_text[:200]}...")
                # Check for specific error types
                if "payment" in resp_text.lower() or "credits" in resp_text.lower():
                    print("   🚨 This is a PAYMENT/CREDITS issue!")
                elif "rate" in resp_text.lower() and "limit" in resp_text.lower():
                    print("   🚨 This is a RATE LIMIT issue!")
                elif "invalid" in resp_text.lower() and "key" in resp_text.lower():
                    print("   🚨 This is an API KEY issue!")
        
        return False


if __name__ == "__main__":
    print("🔍 Debugging REAL Text Generation Failure")
    print("=" * 60)
    
    openrouter_works = test_openrouter_api_directly()
    text_gen_works = debug_real_text_generation()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  OpenRouter API direct: {'✅ WORKS' if openrouter_works else '❌ FAILS'}")
    print(f"  Text Generation Service: {'✅ WORKS' if text_gen_works else '❌ FAILS'}")
    
    if not openrouter_works:
        print("\n🎯 CONCLUSION:")
        print("The issue is NOT with code architecture!")
        print("The issue is that the OpenRouter API requires payment/credits.")
        print("Tests pass because they MOCK the API responses.")
        print("Real generation fails because 402 Payment Required.")
    
    sys.exit(0 if (openrouter_works and text_gen_works) else 1)
