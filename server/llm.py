from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from sqlalchemy.orm import Session

from .models import (
    User,
    Profile,
    Lexeme,
    LexemeInfo,
    UserLexeme,
    LexemeVariant,
)


def _http_json(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Any:
    body = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers=hdrs, method=method)
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_model(base_url: str, prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    try:
        data = _http_json(base_url.rstrip("/") + "/models")
        arr = data.get("data") or []
        if arr:
            return arr[0].get("id") or "local"
    except Exception:
        pass
    return "local"


def _strip_thinking_blocks(text: str) -> str:
    import re
    # Remove <think>...</think> or variants, case-insensitive, multiline
    text = re.sub(r"<\s*(think|thinking|analysis)[^>]*>.*?<\s*/\s*\1\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Drop leading lines that look like reasoning headers
    text = re.sub(r"^(?:\s*(?:Thoughts?|Thinking|Reasoning)\s*:?\s*\n)+", "", text, flags=re.IGNORECASE)
    # Remove fenced code blocks if the model wrapped the passage
    text = re.sub(r"^\s*```[\s\S]*?```\s*", lambda m: m.group(0).strip('`\n '), text, flags=re.MULTILINE)
    return text.strip()


def chat_complete(base_url: str, model: Optional[str], messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": resolve_model(base_url, model),
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        # Avoid using stop sequences that can prematurely cut output (e.g., when model starts with <think>)
        data = _http_json(url, method="POST", data=payload)
        content = data["choices"][0]["message"]["content"].strip()
        return _strip_thinking_blocks(content)
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")


@dataclass
class PromptSpec:
    lang: str
    unit: str  # "chars" or "words"
    approx_len: int
    user_level_hint: Optional[str]
    include_words: List[str]
    script: Optional[str] = None  # for zh: "Hans" or "Hant"


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    sys = (
        "You are a writing assistant. Output ONLY the final passage text. "
        "Do not include meta commentary, analysis, or <think> sections."
    )
    lines = []
    lines.append(f"Language: {spec.lang}")
    if spec.script:
        lines.append(f"Script: {spec.script}")
    lines.append(f"Length: ~{spec.approx_len} {spec.unit}")
    if spec.user_level_hint:
        lines.append(f"UserLevel: {spec.user_level_hint}")
    if spec.include_words:
        words = ", ".join(spec.include_words)
        lines.append(f"TargetWords (must appear naturally): {words}")
    lines.append("Constraints:")
    lines.append("- Do not include translations or vocabulary lists.")
    lines.append("- Avoid English unless the target language is English.")
    lines.append("- Keep the vocabulary consistent with the stated level; gently reinforce target words in context.")
    lines.append("- Output ONLY the passage text; no headings, no bullet points, no analysis.")
    content = "\n".join(lines)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": content},
    ]


def _profile_for_lang(db: Session, user: User, lang: str) -> Optional[Profile]:
    return db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()


def _script_from_lang(lang: str) -> Optional[str]:
    if lang.startswith("zh"):
        return "Hant" if lang.endswith("Hant") else "Hans"
    return None


def pick_words(db: Session, user: User, lang: str, count: int = 12) -> List[str]:
    prof = _profile_for_lang(db, user, lang)
    if not prof:
        return []
    pid = prof.id
    # Join user_lexemes with lexemes and variants
    rows = (
        db.query(UserLexeme, Lexeme)
        .join(Lexeme, UserLexeme.lexeme_id == Lexeme.id)
        .filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == pid)
        .all()
    )
    scored: List[Tuple[float, float, int, Lexeme]] = []
    for ul, lx in rows:
        a = ul.a_click or 0
        b = ul.b_nonclick or 0
        n = a + b
        p = (a / n) if n > 0 else 0.0
        S = float(ul.stability or 0.0)
        # heuristic: prioritize mid/low stability and higher p_click
        score = (1.0 - S) * 0.6 + p * 0.4
        scored.append((score, S, n, lx))
    scored.sort(key=lambda t: t[0], reverse=True)
    picked: List[str] = []
    script = _script_from_lang(lang)
    for _, S, n, lx in scored:
        form = lx.lemma
        if lx.lang == "zh":
            # choose variant by script if present
            if script:
                v = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lx.id, LexemeVariant.script == script).first()
                if v:
                    form = v.form
        if form not in picked:
            picked.append(form)
        if len(picked) >= count:
            break
    return picked


def estimate_level(db: Session, user: User, lang: str) -> Optional[str]:
    # For zh, approximate HSK by taking the most common level among user lexemes with moderate stability
    if not lang.startswith("zh"):
        return None
    prof = _profile_for_lang(db, user, lang)
    if not prof:
        return None
    pid = prof.id
    rows = (
        db.query(UserLexeme, Lexeme, LexemeInfo)
        .join(Lexeme, UserLexeme.lexeme_id == Lexeme.id)
        .outerjoin(LexemeInfo, LexemeInfo.lexeme_id == Lexeme.id)
        .filter(UserLexeme.user_id == user.id, UserLexeme.profile_id == pid)
        .all()
    )
    counts: Dict[str, int] = {}
    for ul, lx, li in rows:
        if not li or not li.level_code:
            continue
        S = float(ul.stability or 0.0)
        if S >= 0.3:  # seen at least somewhat
            counts[li.level_code] = counts.get(li.level_code, 0) + 1
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return top


# ---------- User level description (language-specific) ----------
def _bucket_zh(level_value: float) -> Tuple[str, str]:
    """Map numeric value (0..6) to HSK bucket with lower/mid/upper and a short description."""
    v = max(0.0, min(6.0, float(level_value)))
    if v <= 0.0:
        return ("pre-HSK1", "beginner: limited characters; keep sentences very short and concrete")
    # Determine HSK level 1..6 and sublevel 0:lower 1:mid 2:upper
    if v >= 6.0:
        return ("HSK6-upper", "advanced: broad vocabulary; natural register; nuanced connectors")
    base = int(v)  # 0..5
    hsk = base + 1  # 1..6
    frac = v - base
    sub = int(min(2, frac * 3))  # 0,1,2
    subname = ["lower", "mid", "upper"][sub]
    code = f"HSK{hsk}-{subname}"
    desc_map = {
        1: [
            "very simple daily phrases; pinyin support helpful",
            "short everyday sentences; frequent repetition",
            "connected simple sentences; basic function words",
        ],
        2: [
            "common topics; present/past actions; keep vocabulary basic",
            "short narratives with time words; simple connectors",
            "richer everyday contexts; a few new words ok",
        ],
        3: [
            "familiar themes; short paragraphs; simple relative clauses",
            "multi-paragraph stories; varied sentence starts",
            "mild abstraction; descriptive language; avoid rare idioms",
        ],
        4: [
            "broader topics; modest idiomatic usage; clear structure",
            "argument + examples; transitions (不过/因此/然而)",
            "more nuance; occasional 成语 if transparent",
        ],
        5: [
            "news-like style; more precise vocabulary; balanced complexity",
            "abstract topics; layered clauses; keep clarity",
            "near-native narrative flow; occasional literary turns",
        ],
        6: [
            "authentic style; concise yet expressive; diverse registers",
            "nuanced, persuasive or reflective prose; subtle cohesion",
            "native-like richness; cultural references acceptable",
        ],
    }
    desc = desc_map.get(hsk, ["appropriate difficulty"])[sub]
    return (code, desc)


def compose_level_hint(db: Session, user: User, lang: str) -> Optional[str]:
    prof = _profile_for_lang(db, user, lang)
    if prof and lang.startswith("zh"):
        code, desc = _bucket_zh(getattr(prof, "level_value", 0.0) or 0.0)
        return f"{code}: {desc}"
    # Fallback to inferred level codes
    return estimate_level(db, user, lang)
