# Sample LLM Responses for Testing

This directory contains example LLM responses that demonstrate the various ways language models format their output. These are used in the parsing tests to ensure our JSON extraction and text processing functions can handle real-world LLM outputs.

## Files

- `messy_json_with_fences.txt` - LLM response with markdown code fences
- `partial_json_response.txt` - Incomplete JSON with trailing text
- `xml_wrapped_response.txt` - XML-like tags around JSON
- `translation_response.txt` - Word translation examples
- `mixed_language_text.txt` - Text with mixed language content

## How to Use

These examples should be copied into test functions or used as fixtures for testing the NLP parsing utilities in `server/utils/nlp.py`.
