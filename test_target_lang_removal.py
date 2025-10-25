#!/usr/bin/env python3

import requests
import json

def test_lookup_api():
    """Test that lookup API works without target_lang parameter."""
    
    # Test 1: Basic lookup without target_lang (should work with profile target_lang)
    data = {
        "source_lang": "es",
        "surface": "casa",
        "context": "La casa es azul."
    }
    
    try:
        response = requests.post("http://localhost:8000/api/lookup", json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Lookup API works without target_lang parameter")
            print(f"Translation found: {result.get('translations', 'No translations')}")
            return True
        else:
            print(f"❌ Lookup API failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to localhost:8000 - {e}")
        return False

def test_translate_api():
    """Test that translate API works without target_lang parameter."""
    
    # First create a text to translate
    text_data = {
        "lang": "es",
        "length": 50,
        "include_words": ["casa", "azul"]
    }
    
    try:
      # Generate a reading text first (for testing)
        print("Testing text generation...")
        gen_response = requests.post("http://localhost:8000/gen/reading", json=text_data, timeout=10)
        
        if gen_response.status_code != 200:
             print(f"Failed to generate text: {gen_response.status_code}")
             return False
        
        gen_text = gen_response.json()
        content = gen_text.get("content", "La casa es muy bonita.")
        print(f"Generated text: {content[:100]}...")
        
        # Now test translation without target_lang
        translate_data = {
    "lang": "es",
"unit": "sentence",
 "text": content
        }
        
        translate_response = requests.post("http://localhost:8000/translate", json=translate_data, timeout=10)
        
  if translate_response.status_code == 200:
          translate_result =translate_response.json()
 print("✅ Translate API works without target_lang parameter")
print(f"Translation: {translate_result.get('items', [])[:2] if translate_result.get('items') else 'No translation'}")
 return True
        else:
        print(f"❌ Translate API failed: {translate_response.status_code} - {translate_response.text}")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to localhost:8000 - {e}")
return False

if __name__ == "__main__":
    print("Testing target language parameter removal...")
 print("Note: Server must be running on localhost:8000")
    print("-" * 50)
    
    # Test lookup
    lookup_success = test_lookup_api()
    
    print("-" * 50)
    
    # Test translate
    translate_success = test_translate_api()
    
    print("-" * 50)
    
    if lookup_success and translate_success:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed - check server connection and logs")
