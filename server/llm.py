from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import os

try:
    # Local OpenRouter client from libs/
    from openrouter import complete as _or_complete, resolve_model_id as _or_resolve_model_id  # type: ignore
except Exception:  # pragma: no cover - optional during dev
    _or_complete = None  # type: ignore
    _or_resolve_model_id = None  # type: ignore

from sqlalchemy.orm import Session
from sqlalchemy import select

from server.auth import Account as User
from .models import (
    Profile,
    Lexeme,
    LexemeInfo,
    UserLexeme,
    LexemeVariant,
)
from .level import update_level_if_stale


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


def _pick_openrouter_model(requested: Optional[str]) -> str:
    """Prefer non-thinking model variants by default.

    Order:
    1) explicit requested model
    2) env OPENROUTER_MODEL_NONREASONING (preferred override)
    3) env OPENROUTER_MODEL
    4) fallback 'openrouter/auto'
    """
    if requested:
        return requested
    m = os.getenv("OPENROUTER_MODEL_NONREASONING")
    if m:
        return m
    m2 = os.getenv("OPENROUTER_MODEL")
    return m2 or "openrouter/auto"


def _strip_thinking_blocks(text: str) -> str:
    import re
    original = text or ""
    # Remove <think>...</think> (and common variants), case-insensitive, multiline
    cleaned = re.sub(r"<\s*(think|thinking|analysis)[^>]*>.*?<\s*/\s*\1\s*>", "", original, flags=re.IGNORECASE | re.DOTALL)
    # Drop leading lines that look like reasoning headers
    cleaned = re.sub(r"^(?:\s*(?:Thoughts?|Thinking|Reasoning)\s*:?\s*\n)+", "", cleaned, flags=re.IGNORECASE)
    # If the model wrapped the whole passage in a fenced block, unwrap and keep the inner content
    # Support optional fence label like ```xml
    fenced_full = re.compile(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$", flags=re.DOTALL)
    m = fenced_full.match(cleaned.strip())
    if m:
        cleaned = m.group(1)
    # Final trim
    cleaned = cleaned.strip()
    # Last-resort: if stripping produced empty but original wasn't empty, preserve original
    if not cleaned and original.strip():
        return original.strip()
    return cleaned


def chat_complete(
    messages: List[Dict[str, str]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    base_url: str = "http://localhost:1234/v1",
) -> str:
    """Completion wrapper with provider switch.

    provider: "openrouter" | "lmstudio" (default inferred from env)
    """
    prov = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not prov:
        # Default to OpenRouter if available; fall back to LM Studio explicitly
        prov = "openrouter"
    if prov == "openrouter":
        if _or_complete is None:
            raise RuntimeError("openrouter client not available; install libs/openrouter or set LLM_PROVIDER=lmstudio")
        use_model = _pick_openrouter_model(model)
        # Allow resolving aliases/labels via catalog when available
        try:
            if _or_resolve_model_id is not None:
                use_model = _or_resolve_model_id(use_model)  # type: ignore
        except Exception:
            pass
        try:
            data = _or_complete(messages=messages, model=use_model, max_tokens=(max_tokens or 4096), temperature=temperature)
            content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            return _strip_thinking_blocks(content)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"LLM request (openrouter) failed: {e}")
    else:
        # LM Studio / OpenAI-compatible local endpoint
        url = base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": resolve_model(base_url, model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            data = _http_json(url, method="POST", data=payload, timeout=120)
            content = data["choices"][0]["message"]["content"].strip()
            return _strip_thinking_blocks(content)
        except Exception as e:
            raise RuntimeError(f"LLM request (lmstudio) failed: {e}")


@dataclass
class PromptSpec:
    lang: str
    unit: str  # "chars" or "words"
    approx_len: int
    user_level_hint: Optional[str]
    include_words: List[str]
    script: Optional[str] = None  # for zh: "Hans" or "Hant"
    ci_target: Optional[float] = None  # desired share of familiar tokens (0..1)


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    sys_tpl = _load_prompt("reading_system.txt") or (
        "You are a tutor of {lang_display}. Please generate a text for learning and comprehensible input practice, given the following parameters."
    )
    user_tpl = _load_prompt("reading_user.txt") or (
        "Write in {lang_display}.\n{script_line}{level_line}{length_line}{include_words_line}{ci_line}Constraints:\n- Do not include translations or vocabulary lists.\n- Avoid English unless the target language is English.\n- Gently reinforce the target words in context and keep the language natural and engaging.\n- Do not include meta commentary."
    )

    lang_display = _lang_display(spec.lang)
    script_line = ""
    if spec.script and spec.lang.startswith("zh"):
        if spec.script == "Hans":
            script_line = "Use simplified Chinese characters.\n"
        elif spec.script == "Hant":
            script_line = "Use traditional Chinese characters.\n"
    level_line = f"The student is around {spec.user_level_hint}; please use appropriate language for this level.\n" if spec.user_level_hint else ""
    length_line = (
        f"The text should be around {spec.approx_len} characters long.\n" if spec.unit == "chars" else f"The text should be around {spec.approx_len} words long.\n"
    )
    include_words_line = ""
    if spec.include_words:
        words = ", ".join(spec.include_words)
        include_words_line = f"Please include these words naturally: {words}.\n"
    ci_line = ""
    if isinstance(spec.ci_target, (int, float)) and spec.ci_target:
        pct = int(round(float(spec.ci_target) * 100))
        ci_line = f"Aim for about {pct}% of tokens to be familiar for the learner; limit new vocabulary.\n"

    sys_content = sys_tpl.format(lang_display=lang_display)
    user_content = user_tpl.format(
        lang_display=lang_display,
        script_line=script_line,
        level_line=level_line,
        length_line=length_line,
        include_words_line=include_words_line,
        ci_line=ci_line,
    )
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]


@dataclass
class TranslationSpec:
    lang: str
    target_lang: str
    unit: str  # "sentence" | "paragraph" | "text"
    content: Union[str, List[str]]
    continue_with_reading: bool = False
    script: Optional[str] = None  # for zh source formatting


def build_translation_prompt(spec: TranslationSpec, prev_messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    # XML-only output mode
    sys_tpl = _load_prompt("translation_system_xml.txt") or (
        "You are a professional translator from {src_lang} to {tgt_lang}. Respond with a single well-formed XML document only. {line_mode_line}{script_line}"
    )
    user_tpl = _load_prompt("translation_user_xml.txt") or ("{content}")

    src_lang = _lang_display(spec.lang)
    tgt_lang = _lang_display(spec.target_lang)
    line_mode_line = (
        "Translate each input line independently and return exactly one line per input line in the same order."
        if isinstance(spec.content, list) else ""
    )
    script_line = ""
    if spec.script and spec.lang.startswith("zh"):
        if spec.script == "Hans":
            script_line = "Source text may be in simplified Chinese."
        elif spec.script == "Hant":
            script_line = "Source text may be in traditional Chinese."

    sys_content = sys_tpl.format(src_lang=src_lang, tgt_lang=tgt_lang, line_mode_line=line_mode_line, script_line=script_line)
    lines = spec.content if isinstance(spec.content, list) else [spec.content]
    user_content = user_tpl.format(content="\n".join(lines))

    msgs: List[Dict[str, str]] = []
    if prev_messages and spec.continue_with_reading:
        tail = prev_messages[-4:]
        msgs.extend([{"role": m.get("role", "user"), "content": m.get("content", "")} for m in tail if isinstance(m, dict)])
    msgs.insert(0, {"role": "system", "content": sys_content})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _profile_for_lang(db: Session, user: User, lang: str) -> Optional[Profile]:
    return db.query(Profile).filter(Profile.account_id == user.id, Profile.lang == lang).first()


def _script_from_lang(lang: str) -> Optional[str]:
    if lang.startswith("zh"):
        # Default to simplified when no explicit preference present
        return "Hans"
    return None


def _variant_form_for_lang(db: Session, user: User, lx: Lexeme, lang: str) -> str:
    form = lx.lemma
    if lx.lang == "zh":
        script = None
        # Prefer user profile setting when available
        prof = _profile_for_lang(db, user, lang)
        if prof and getattr(prof, "preferred_script", None):
            script = prof.preferred_script
        if not script:
            script = _script_from_lang(lang)
        if script:
            v = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lx.id, LexemeVariant.script == script).first()
            if v:
                form = v.form
    return form


def _hsk_numeric(level_code: Optional[str]) -> Optional[int]:
    if not level_code:
        return None
    # Expect codes like HSK1..HSK6
    try:
        if level_code.upper().startswith("HSK"):
            return int(level_code[3:])
    except Exception:
        return None
    return None


def urgent_words_detailed(db: Session, user: User, lang: str, total: int = 12, new_ratio: float = 0.3) -> List[Dict[str, Any]]:
    # ensure level estimate is fresh enough
    try:
        update_level_if_stale(db, user.id, lang)
    except Exception:
        pass
    prof = _profile_for_lang(db, user, lang)
    if not prof:
        return []
    pid = prof.id
    total = max(1, int(total))
    # Known words scoring
    rows = (
        db.query(UserLexeme, Lexeme, LexemeInfo)
        .join(Lexeme, UserLexeme.lexeme_id == Lexeme.id)
        .outerjoin(LexemeInfo, LexemeInfo.lexeme_id == Lexeme.id)
        .filter(UserLexeme.account_id == user.id, UserLexeme.profile_id == pid)
        .all()
    )
    known_scored: List[Tuple[float, Lexeme, Dict[str, Any]]] = []
    now_ts = 0.0
    for ul, lx, li in rows:
        # decayed Beta metrics
        alpha = float(getattr(ul, "alpha", None) or 1.0)
        beta = float(getattr(ul, "beta", None) or 9.0)
        mu = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1)) if (alpha + beta + 1) > 0 else 0.0
        ucb = mu + math.sqrt(var)
        # retention
        S = max(0.5, float(getattr(ul, "stability", 1.0) or 1.0))
        ref = ul.last_seen_at or ul.first_seen_at or ul.created_at
        dt = 0.0
        if ref:
            try:
                dt = max(0.0, (datetime.utcnow() - ref).total_seconds() / 86400.0)
            except Exception:
                dt = 0.0
        try:
            R_now = math.exp(-dt / S)
        except Exception:
            R_now = 0.5
        imp = float(getattr(ul, "importance", 0.5) or 0.5)
        d_div = 1.0 - math.exp(- float(getattr(ul, "distinct_texts", 0) or 0) / 8.0)
        rec_pen = 0.0
        if ul.last_clicked_at:
            try:
                rec_sec = (datetime.utcnow() - ul.last_clicked_at).total_seconds()
                rec_pen = max(0.0, 1.0 - rec_sec / 600.0)  # 10 min decay
            except Exception:
                rec_pen = 0.0
        base = 0.6 * (1.0 - R_now) + 0.3 * ucb + 0.1 * (0.5 + 0.5 * imp) + 0.05 * d_div - 0.1 * rec_pen
        known_scored.append((base, lx, {"ucb": ucb, "R_now": R_now, "pos": lx.pos}))
    known_scored.sort(key=lambda t: t[0], reverse=True)

    # dynamic new ratio by due backlog (approx via R_now)
    due = sum(1 for _, _lx, meta in known_scored if meta.get("R_now", 1.0) < 0.9)
    tk_default = int(round(total * (1.0 - new_ratio)))
    if due > 2 * tk_default:
        new_ratio = 0.1
    elif due < 0.5 * tk_default:
        new_ratio = 0.4
    target_known = max(0, min(len(known_scored), total - int(round(total * new_ratio))))
    picked: List[Dict[str, Any]] = []
    picked_ids = set()
    def _sim(a: str, b: str) -> float:
        if not a or not b or a == b:
            return 1.0 if a == b else 0.0
        if lang.startswith("zh"):
            def bigrams(s: str):
                return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else {s}
            A, B = bigrams(a), bigrams(b)
            if not A or not B:
                return 0.0
            return len(A & B) / max(1, len(A | B))
        return 0.3 if a.lower() != b.lower() else 1.0
    # MMR select for known
    K = []
    cand_known = []
    for score, lx, meta in known_scored:
        form = _variant_form_for_lang(db, user, lx, lang)
        cand_known.append((score, lx, form, meta))
    while cand_known and len(K) < target_known:
        best = None
        best_s = -1e9
        for score, lx, form, meta in cand_known[:200]:
            if any(p["form"] == form for p in picked):
                continue
            sim = 0.0
            if K:
                sim = max(_sim(form, p["form"]) for p in K)
            s = 0.7 * score - 0.3 * sim
            if s > best_s:
                best_s = s
                best = (score, lx, form, meta)
        if not best:
            break
        _, lx, form, _meta = best
        K.append({"form": form, "lexeme_id": lx.id, "known": True})
        cand_known = [c for c in cand_known if c[1].id != lx.id]
    picked.extend(K)

    # New word candidates (unknown to user)
    need_new = total - len(picked)
    if need_new > 0:
        # Build pool from LexemeInfo near profile level and decent frequency
        user_level = getattr(prof, "level_value", 0.0) or 0.0
        hsk_target = None
        if lang.startswith("zh"):
            # map continuous 0..6 to integer 1..6 buckets
            v = max(0.0, min(6.0, float(user_level)))
            hsk_target = int(min(6, max(1, int(round(v))))) or 1
        subq_known = select(UserLexeme.lexeme_id).where(UserLexeme.account_id == user.id, UserLexeme.profile_id == pid)
        pool_q = (
            db.query(Lexeme, LexemeInfo)
            .join(LexemeInfo, LexemeInfo.lexeme_id == Lexeme.id)
            .filter(Lexeme.lang == ("zh" if lang.startswith("zh") else lang))
            .filter(Lexeme.id.notin_(subq_known))
        )
        pool = pool_q.limit(5000).all()
        scored_new: List[Tuple[float, Lexeme, str]] = []
        for lx, li in pool:
            if not li:
                continue
            # prefer close to target level (HSK delta small) and good frequency (low rank)
            level_num = _hsk_numeric(li.level_code) if lang.startswith("zh") else None
            level_closeness = 0.0
            if hsk_target and level_num:
                delta = abs(level_num - hsk_target)
                level_closeness = 1.0 / (1.0 + delta)
            freq = 0.0
            if li.freq_rank:
                freq = 1.0 / (1.0 + li.freq_rank)
            score = 0.6 * level_closeness + 0.4 * freq
            form = _variant_form_for_lang(db, user, lx, lang)
            scored_new.append((score, lx, form))
        scored_new.sort(key=lambda t: t[0], reverse=True)
        # MMR for new
        N = []
        while scored_new and len(N) < need_new:
            best = None
            best_s = -1e9
            for score, lx, form in scored_new[:200]:
                if any(p["form"] == form for p in picked) or any(p["form"] == form for p in N):
                    continue
                sim = 0.0
                base_set = picked + N
                if base_set:
                    sim = max(_sim(form, p["form"]) for p in base_set)
                s = 0.7 * score - 0.3 * sim
                if s > best_s:
                    best_s = s
                    best = (score, lx, form)
            if not best:
                break
            _, lx, form = best
            N.append({"form": form, "lexeme_id": lx.id, "known": False})
            scored_new = [c for c in scored_new if c[1].id != lx.id]
        picked.extend(N)

    return picked[:total]


def pick_words(db: Session, user: User, lang: str, count: int = 12, *, new_ratio: Optional[float] = None) -> List[str]:
    """Return only forms for prompt inclusion.

    new_ratio: desired fraction of new words among picked (0..1). If None, default from urgent_words_detailed.
    """
    kwargs: Dict[str, Any] = {"total": count}
    if isinstance(new_ratio, (int, float)):
        kwargs["new_ratio"] = float(new_ratio)
    return [it["form"] for it in urgent_words_detailed(db, user, lang, **kwargs)]


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
        .filter(UserLexeme.account_id == user.id, UserLexeme.profile_id == pid)
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
