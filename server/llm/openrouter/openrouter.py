from __future__ import annotations
import os
import json
import base64
import mimetypes
import httpx
from typing import AsyncIterator, Dict, List, Optional, Literal, TypedDict, NotRequired, Union, IO, Callable
from contextlib import contextmanager
import asyncio
import time
import random

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Build request headers using the current env value.

    Delays API key lookup to call-time so .env can be loaded earlier
    by the application before using this client.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it or put it in .env before calling OpenRouter."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Optional headers recommended by OpenRouter
    # Allow override via env vars, with a sensible default X-Title
    app_title = os.getenv("OPENROUTER_APP_TITLE", "Arcadia AI Chat")
    headers["X-Title"] = app_title
    if ref := os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = ref
    return headers

_RETRIABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))

# Persistent clients to reuse connections and reduce handshake overhead
_sync_client: Optional[httpx.Client] = None
_async_client: Optional[httpx.AsyncClient] = None

def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _sync_client

def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _async_client

def _backoff_delay(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after and retry_after.isdigit():
        base = float(retry_after)
    else:
        base = float(2 ** attempt)
    jitter = random.uniform(0, base * 0.5)
    return base + jitter

class Chunk(TypedDict):
    kind: Literal["reasoning", "content", "usage"]
    text: str
    usage: NotRequired[dict]

class StreamController:
    """
    A controllable stream that allows stopping generation mid-stream.
    """
    def __init__(self, generator_factory):
        self.generator_factory = generator_factory
        self._cancelled = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def __aiter__(self):
        """Iterate over the stream, checking for cancellation."""
        self._task = asyncio.current_task()
        generator = self.generator_factory(cancellation_event=self._cancelled)
        async for chunk in generator:
            if self._cancelled.is_set():
                break
            yield chunk

    def stop(self):
        """Stop the stream immediately."""
        self._cancelled.set()
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def stopped(self) -> bool:
        """Check if the stream has been stopped."""
        return self._cancelled.is_set()

# ---------- sync helpers (keep old surface if you want) ----------
def complete(
    messages: List[dict],
    model: str,
    *,
    max_tokens: int = 4096,
    reasoning: Optional[Dict] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
) -> dict:
    payload: Dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _get_sync_client().post(_BASE_URL, headers=_headers(api_key), json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in _RETRIABLE_STATUS and attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt, e.response.headers.get("Retry-After"))
                time.sleep(delay)
                last_err = e
                continue
            raise
        except httpx.HTTPError as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_backoff_delay(attempt, None))
                last_err = e
                continue
            raise
    # Should not reach here, but just in case
    if last_err:
        raise last_err
    raise RuntimeError("OpenRouter request failed without specific error")

# ---------- new single async generator ----------
async def _astream_generator(
    messages: List[dict],
    model: str,
    *,
    max_tokens: int = 32_768,
    thinking: bool = False,
    cancellation_event: Optional[asyncio.Event] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    api_key: Optional[str] = None,
) -> AsyncIterator[Chunk]:
    """
    Internal generator that yields (kind, text) tuples while the model streams.
    kind is one of: "reasoning", "content", "usage".
    """
    payload: Dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "include_reasoning": thinking,
    }

    # Retry only around establishing the stream; once streaming begins, we don't retry mid-stream.
    for attempt in range(_MAX_RETRIES + 1):
        try:
            client = _get_async_client()
            async with client.stream(
                "POST", _BASE_URL, headers=_headers(api_key), json=payload
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    # Check for cancellation
                    if cancellation_event and cancellation_event.is_set():
                        break

                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    data = json.loads(chunk)
                    # Allow callers to capture the raw SSE JSON event
                    try:
                        if on_event is not None and isinstance(data, dict):
                            on_event(data)
                    except Exception:
                        pass

                    # usage block (sent once at the end)
                    if "usage" in data:
                        yield {"kind": "usage", "text": "", "usage": data["usage"]}
                        continue

                    delta = data["choices"][0].get("delta", {})

                    # reasoning text
                    reason = (delta.get("reasoning") or "") + "".join(
                        rd["text"]
                        for rd in delta.get("reasoning_details", [])
                        if rd.get("type") == "reasoning.text"
                    )
                    if reason:
                        yield {"kind": "reasoning", "text": reason}

                    # response text
                    if content := delta.get("content"):
                        yield {"kind": "content", "text": content}
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in _RETRIABLE_STATUS and attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt, e.response.headers.get("Retry-After"))
                await asyncio.sleep(delay)
                continue
            raise
        except httpx.HTTPError:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_backoff_delay(attempt, None))
                continue
            raise

def astream(
    messages: List[dict],
    model: str,
    *,
    max_tokens: int = 32_768,
    thinking: bool = False,
    on_event: Optional[Callable[[dict], None]] = None,
    api_key: Optional[str] = None,
) -> StreamController:
    """
    Return a controllable stream that allows stopping generation mid-stream.

    Returns:
        StreamController: A stream object with stop() method to cancel generation.
    """
    return StreamController(lambda cancellation_event=None: _astream_generator(
        messages, model, max_tokens=max_tokens, thinking=thinking, cancellation_event=cancellation_event, on_event=on_event, api_key=api_key
    ))

async def consume_and_drop(generator: AsyncIterator[Chunk]) -> None:
    """Consume and drop the rest of a stream."""
    async for _ in generator:
        pass


# ---------------- Content helpers (multimodal) ----------------

FileInput = Union[str, bytes, IO[bytes]]


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_b64(data)}"


def _guess_mime(filename: Optional[str], fallback: str = "application/octet-stream") -> str:
    if filename:
        mime, _ = mimetypes.guess_type(filename)
        if mime:
            return mime
    return fallback


def _is_pdf_url(url: str) -> bool:
    if not _is_http_url(url):
        return False
    # quick heuristic: extension or mimetype guess
    if url.lower().endswith(".pdf"):
        return True
    mime = _guess_mime(url)
    return mime == "application/pdf"


def _read_file_input(file: FileInput) -> tuple[bytes, Optional[str]]:
    if isinstance(file, bytes):
        return file, None
    if isinstance(file, str):
        # Caller must handle http(s) URLs separately; here we treat strings as paths.
        with open(file, "rb") as f:
            return f.read(), os.path.basename(file)
    # file-like object
    data = file.read()
    name = getattr(file, "name", None)
    if isinstance(name, str):
        name = os.path.basename(name)
    else:
        name = None
    return data, name


def _audio_format_from(mime: Optional[str], name: Optional[str]) -> Optional[str]:
    # Prefer MIME mapping, then extension
    m = (mime or "").lower()
    if m in ("audio/mpeg", "audio/mp3"):
        return "mp3"
    if m in ("audio/wav", "audio/x-wav"):
        return "wav"
    if m in ("audio/m4a", "audio/x-m4a", "audio/aac"):
        return "m4a" if m != "audio/aac" else "aac"
    if m in ("audio/ogg", "audio/ogg; codecs=opus"):
        return "ogg"
    if m == "audio/flac":
        return "flac"
    if m in ("audio/webm", "audio/opus"):
        return "webm" if m == "audio/webm" else "opus"
    # Try extension
    if name:
        n = name.lower()
        for ext, fmt in (
            (".mp3", "mp3"), (".wav", "wav"), (".m4a", "m4a"), (".aac", "aac"),
            (".ogg", "ogg"), (".flac", "flac"), (".webm", "webm"), (".opus", "opus"),
        ):
            if n.endswith(ext):
                return fmt
    return None


_TEXTUAL_APPLICATION_MIMES = {
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
    "application/yaml",
    "application/x-yaml",
    "application/sql",
}


def content_from_file(file: FileInput, mime: Optional[str] = None) -> dict:
    """Return a single content part dict for a file input.

    Rules:
    - Images: image_url with data URI (or pass-through if http(s) URL)
    - PDFs: file with base64 data
    - Audio: input_audio with base64 data and format
    - Text: decoded into a text part
    - Other: generic file with base64 data
    """
    # URL pass-through for images and PDFs
    if isinstance(file, str) and _is_http_url(file):
        guessed = _guess_mime(file)
        if guessed.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": file}}
        if _is_pdf_url(file):
            return {"type": "file", "mime_type": "application/pdf", "url": file}
        raise ValueError("Only image and PDF URLs are supported; provide local bytes/path for other files.")

    data, name = _read_file_input(file)
    use_mime = mime or _guess_mime(name)

    if use_mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": _to_data_uri(data, use_mime)}}

    if use_mime == "application/pdf":
        return {"type": "file", "mime_type": "application/pdf", "data": _b64(data)}

    if use_mime.startswith("audio/"):
        fmt = _audio_format_from(use_mime, name) or "mp3"
        return {"type": "input_audio", "audio": {"data": _b64(data), "format": fmt}}

    if use_mime.startswith("text/") or use_mime in _TEXTUAL_APPLICATION_MIMES:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return {"type": "text", "text": text}

    return {"type": "file", "mime_type": use_mime or "application/octet-stream", "data": _b64(data)}


def content_from(text: Optional[str] = None, files: Optional[List[FileInput]] = None) -> List[dict]:
    """Build a message content array from optional text and files.

    Example:
        content = content_from(
            text="Summarize the document",
            files=["/path/to/doc.pdf", "/path/to/image.png", "/path/to/notes.txt"],
        )
        messages = [{"role": "user", "content": content}]
    """
    parts: List[dict] = []
    if text:
        parts.append({"type": "text", "text": text})
    for f in files or []:
        parts.append(content_from_file(f))
    return parts


# ---------------- Higher-level helpers: env/key override, full message builder ----------------

@contextmanager
def with_api_key(key: str):
    """Temporarily override OPENROUTER_API_KEY for the duration of the context."""
    old = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = key
    try:
        yield
    finally:
        if old is None:
            try:
                del os.environ["OPENROUTER_API_KEY"]
            except Exception:
                pass
        else:
            os.environ["OPENROUTER_API_KEY"] = old


def build_or_messages(
    messages: List[Dict],
    *,
    attachments: Optional[List[Dict]] = None,
    resolver: Optional[Callable[[Dict], Optional[Union[str, bytes, IO[bytes]]]]] = None,
    max_inline_bytes: int = 8 * 1024 * 1024,
) -> List[Dict]:
    """Return OpenRouter messages with attachments appended to the last user message.

    - `resolver(att)` should return a file path, bytes, file-like, or None for a given attachment dict
    - Inlines files up to `max_inline_bytes`; otherwise appends a textual stub
    - Leaves system/assistant messages unchanged
    """
    if not attachments:
        return messages
    out = [dict(m) for m in messages]
    # find last user
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if isinstance(out[i], dict) and out[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return out
    umsg = dict(out[idx])
    base = umsg.get("content", "")
    parts: List[Dict] = []
    if isinstance(base, list):
        parts.extend(base)
    elif isinstance(base, str) and base:
        parts.append({"type": "text", "text": base})
    # map attachments
    for att in attachments:
        try:
            if not isinstance(att, dict):
                continue
            rel = (att.get("rel") or "").strip()
            name = (att.get("name") or os.path.basename(rel) or "file")
            mime = (att.get("content_type") or "").strip() or _guess_mime(name)
            size = int(att.get("size") or 0)
            # resolve source
            src: Optional[Union[str, bytes, IO[bytes]]] = None
            if resolver:
                src = resolver(att)
            # Determine bytes length if not provided
            src_path = None
            if isinstance(src, str):
                src_path = src
                if size <= 0:
                    try:
                        size = os.path.getsize(src)
                    except Exception:
                        size = 0
            elif isinstance(src, (bytes, bytearray)):
                if size <= 0:
                    size = len(src)
            elif src is not None:
                try:
                    pos = src.tell()
                except Exception:
                    pos = None
                try:
                    data = src.read()
                except Exception:
                    data = b""
                if hasattr(src, "seek") and pos is not None:
                    try:
                        src.seek(pos)
                    except Exception:
                        pass
                src = data
                size = len(data)

            if src is None or size > max_inline_bytes:
                human = f"{size/1024/1024:.1f}MB" if size else "unknown size"
                parts.append({"type": "text", "text": f"Attachment: {name} ({mime}, {human}) at {rel}"})
                continue

            # Inline via content_from_file
            if src_path:
                part = content_from_file(src_path, mime=mime)
            else:
                # bytes/file-like already loaded as bytes
                b = src if isinstance(src, (bytes, bytearray)) else bytes(src)
                part = content_from_file(b, mime=mime)
            parts.append(part)
        except Exception:
            continue
    umsg["content"] = parts
    out[idx] = umsg
    return out