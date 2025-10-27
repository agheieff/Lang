"""
JSON parsing utilities for LLM responses.
"""

import json
import re
from typing import Any, Dict, Optional


def extract_json_from_text(text: str, expected_key: str = "text") -> Optional[Any]:
    """
    Extract JSON object from text response.

    Args:
        text: The text response from LLM
        expected_key: The key to look for in the JSON object

    Returns:
        The extracted value if found, otherwise None
    """
    if not text or text.strip() == "":
        return None

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"' + re.escape(expected_key) + r'"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if isinstance(json_data, dict) and expected_key in json_data:
                return json_data[expected_key]
        except json.JSONDecodeError:
            # If JSON parsing fails, try to find the first valid JSON object
            pass

    # Alternative: try to find any JSON object
    try:
        # Look for JSON objects with the expected key
        json_objects = re.findall(r'\{[^{}]*"' + re.escape(expected_key) + r'"[^{}]*\}', text, re.DOTALL)
        for json_str in json_objects:
            try:
                json_data = json.loads(json_str)
                if isinstance(json_data, dict) and expected_key in json_data:
                    return json_data[expected_key]
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None




def extract_text_from_llm_response(response: str) -> str:
    """
    Extract text from LLM response, handling JSON format if present.

    Args:
        response: The raw response from LLM

    Returns:
        The extracted text
    """
    if not response or response.strip() == "":
        return ""

    # Try to extract text from JSON
    extracted_text = extract_json_from_text(response, "text")
    if extracted_text is not None:
        return str(extracted_text)

    # Fallback to original response
    return response


def extract_partial_json_string(buffer: str, key: str = "text") -> Optional[str]:
    """Best-effort extraction of a JSON string value for a given key from a partial/incomplete buffer.

    Handles unfinished JSON while streaming by locating the opening quote after "key": and
    scanning until the next unescaped matching quote. Returns the partial contents collected so far.
    """
    if not buffer:
        return None
    try:
        start_key = f'"{key}"'
        i = buffer.find(start_key)
        if i < 0:
            return None
        # find colon after key
        i = buffer.find(":", i)
        if i < 0:
            return None
        # skip spaces
        n = len(buffer)
        i += 1
        while i < n and buffer[i].isspace():
            i += 1
        if i >= n:
            return None
        q = buffer[i]
        if q not in ('"', "'"):
            return None
        i += 1
        out: list[str] = []
        esc = False
        while i < n:
            ch = buffer[i]
            i += 1
            if esc:
                # rudimentary unescape for common sequences
                if ch in ('"', "'", "\\", "/"):
                    out.append(ch)
                elif ch == "n":
                    out.append("\n")
                elif ch == "t":
                    out.append("\t")
                elif ch == "r":
                    out.append("\r")
                else:
                    out.append(ch)
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == q:
                # closing quote reached
                break
            out.append(ch)
        return "".join(out)
    except Exception:
        return None


def extract_structured_translation(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured translation from LLM response.

    Args:
        response: The raw response from LLM

    Returns:
        The structured translation data if found, otherwise None
    """
    if not response or response.strip() == "":
        return None

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"text"[^{}]*"paragraphs"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if (isinstance(json_data, dict) and
                "text" in json_data and
                "paragraphs" in json_data and
                isinstance(json_data["paragraphs"], list)):
                return json_data
        except json.JSONDecodeError:
            # If JSON parsing fails, try alternative patterns
            pass

    # Alternative: try to find any valid JSON with the expected structure
    try:
        # Look for JSON objects with text and paragraphs
        json_objects = re.findall(r'\{[^{}]*"text"[^{}]*"paragraphs"[^{}]*\}', response, re.DOTALL)
        for json_str in json_objects:
            try:
                json_data = json.loads(json_str)
                if (isinstance(json_data, dict) and
                    "text" in json_data and
                    "paragraphs" in json_data and
                    isinstance(json_data["paragraphs"], list)):
                    return json_data
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None


def extract_word_translations(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract word translations from LLM response.

    Args:
        response: The raw response from LLM

    Returns:
        The word translation data if found, otherwise None
    """
    if not response or response.strip() == "":
        return None

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"text"[^{}]*"words"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if (isinstance(json_data, dict) and
                "text" in json_data and
                "words" in json_data and
                isinstance(json_data["words"], list)):
                return json_data
        except json.JSONDecodeError:
            # If JSON parsing fails, try alternative patterns
            pass

    # Alternative: try to find any valid JSON with the expected structure
    try:
        # Look for JSON objects with text and words
        json_objects = re.findall(r'\{[^{}]*"text"[^{}]*"words"[^{}]*\}', response, re.DOTALL)
        for json_str in json_objects:
            try:
                json_data = json.loads(json_str)
                if (isinstance(json_data, dict) and
                    "text" in json_data and
                    "words" in json_data and
                    isinstance(json_data["words"], list)):
                    return json_data
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None