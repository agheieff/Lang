"""Unit tests for LLM JSON and text parsing - the most brittle logic."""

import pytest
from server.utils.nlp import extract_json_from_text, extract_word_translations, split_sentences


def test_extract_json_messy():
    """Test JSON extraction from messy LLM output with markdown."""
    messy_input = """
    Sure, here is your JSON:
    ```json
    {"text": "Hola mundo", "words": ["hola", "mundo"]}
    ```
    Hope this helps!
    """
    
    data = extract_json_from_text(messky_input)
    assert data["text"] == "Hola mundo"
    assert "hola" in data["words"]
    assert "mundo" in data["words"]


def test_extract_json_without_fences():
    """Test JSON extraction when LLM doesn't use markdown fences."""
    simple_input = '{"result": "success", "data": {"value": 42}}'
    
    data = extract_json_from_text(simple_input)
    assert data["result"] == "success"
    assert data["data"]["value"] == 42


def test_extract_json_with_extra_text():
    """Test JSON extraction when LLM includes explanatory text."""
    with_preamble = """
    Based on your request, here's the structured data:
    {"translations": [{"word": "casa", "meaning": "house"}]}
    Please let me know if you need anything else!
    """
    
    data = extract_json_from_text(with_preamble)
    assert len(data["translations"]) == 1
    assert data["translations"][0]["word"] == "casa"


def test_extract_json_with_xml_tags():
    """Test JSON extraction when LLM uses XML-like tags."""
    xml_wrapped = """
    <response>
    {"status": "completed", "items": ["item1", "item2"]}
    </response>
    """
    
    data = extract_json_from_text(xml_wrapped)
    assert data["status"] == "completed"
    assert len(data["items"]) == 2


def test_extract_json_malformed_fallback():
    """Test graceful handling of malformed JSON."""
    malformed = '{"text": "This is broken", "invalid": }'
    
    # Should either extract what's valid or return None gracefully
    try:
        data = extract_json_from_text(malformed)
        # If it doesn't raise an exception, check if partial extraction worked
        assert data.get("text") == "This is broken"
    except (ValueError, KeyError):
        # It's acceptable to raise an exception for malformed JSON
        pass


def test_extract_json_array_response():
    """Test JSON extraction when LLM returns an array instead of object."""
    array_input = '''
    Here are the translations:
    ```json
    [
        {"surface": "hola", "translation": "hello"},
        {"surface": "adios", "translation": "goodbye"}
    ]
    ```
    '''
    
    data = extract_json_from_text(array_input)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["surface"] == "hola"


def test_extract_word_translations_various_formats():
    """Test word translation extraction from different LLM response formats."""
    
    # Format 1: Simple object
    simple_format = '[{"surface": "gato", "translation": "cat"}]'
    translations = extract_word_translations(simple_format)
    assert len(translations) == 1
    assert translations[0]["surface"] == "gato"
    
    # Format 2: With additional fields
    with_metadata = '''
    [
        {"surface": "perro", "translation": "dog", "pos": "NOUN", "confidence": 0.95}
    ]
    '''
    translations = extract_word_translations(with_metadata)
    assert len(translations) == 1
    assert translations[0]["surface"] == "perro"
    assert translations[0].get("pos") == "NOUN"


def test_extract_word_translations_with_extra_text():
    """Test word translation extraction from messy LLM responses."""
    messy_translations = """
    Here are the translations you requested:
    ```json
    [
        {"surface": "agua", "translation": "water"},
        {"surface": "fuego", "translation": "fire"}
    ]
    ```
    These should help with your learning!
    """
    
    translations = extract_word_translations(messy_translations)
    assert len(translations) == 2
    
    surfaces = [t["surface"] for t in translations]
    assert "agua" in surfaces
    assert "fuego" in surfaces


def test_split_sentences_spanish():
    """Test Spanish sentence splitting."""
    text = "Hola mundo. ¿Cómo estás? Estoy bien, gracias."
    sentences = split_sentences(text, "es")
    
    # Should split on actual sentence boundaries
    assert len(sentences) >= 2
    assert "Hola mundo" in sentences[0] or "Hola mundo" == sentences[0]


def test_split_sentences_english():
    """Test English sentence splitting."""
    text = "Hello world. How are you? I'm fine, thank you."
    sentences = split_sentences(text, "en")
    
    assert len(sentences) >= 2
    assert "Hello world" in sentences[0] or "Hello world" == sentences[0]


def test_split_sentences_empty_text():
    """Test sentence splitting with empty or minimal input."""
    assert split_sentences("", "es") == []
    assert split_sentences(".", "es") == []  # Just punctuation
    assert split_sentences("palabra", "es") == ["palabra"]  # Single word


def test_llm_response_cleaning():
    """Test LLM response cleaning utilities."""
    from server.llm.client import _strip_thinking_blocks
    
    # Test thinking block removal
    thinking_text = """
    <think>
    I need to respond in Spanish. Let me think about this carefully.
    </think>
    
    Hola, ¿cómo estás?
    """
    
    cleaned = _strip_thinking_blocks(thinking_text)
    assert "Hola, ¿cómo estás?" in cleaned
    assert "<think>" not in cleaned
    assert "I need to respond" not in cleaned
    
    # Test markdown fence removal
    fenced_text = '''```json
{"text": "contenido"}
```
Contenido extra'''
    
    cleaned = _strip_thinking_blocks(fenced_text)
    assert cleaned.strip() == '{"text": "contenido"}'


def test_extract_json_partial_extraction():
    """Test partial JSON extraction when there are incomplete structures."""
    partial_json = '''
    {
        "valid": {"name": "test"},
        "invalid": 
    '''
    
    try:
        data = extract_json_from_text(partial_json)
        # If extraction works, check what we got
        if data and "valid" in data:
            assert data["valid"]["name"] == "test"
    except (ValueError, KeyError, json.JSONDecodeError):
        # Should fail gracefully for malformed JSON
        pass


def test_extract_json_unicode_content():
    """Test JSON extraction with Unicode characters."""
    unicode_json = '''
    {
        "spanish": "¡Hola! ¿Cómo estás?",
        "chinese": "你好吗？",
        "emoji": "👋🌍"
    }
    '''
    
    data = extract_json_from_text(unicode_json)
    assert "¡Hola!" in data.get("spanish", "")
    assert "你好" in data.get("chinese", "")
    assert "👋" in data.get("emoji", "")
