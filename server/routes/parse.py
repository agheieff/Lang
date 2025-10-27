from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from nlp.tokenize import TOKENIZERS, Token


router = APIRouter(prefix="/api", tags=["parse"])


class ParseRequest(BaseModel):
    lang: str
    text: str


@router.post("/parse")
def parse(req: ParseRequest) -> Dict[str, Any]:
    tok = TOKENIZERS.get(req.lang, TOKENIZERS["default"])
    words: list[Token] = tok.tokenize(req.text)
    tokens_out: list[Dict[str, Any]] = []

    def add_sep(start: int, end: int):
        if end > start:
            tokens_out.append(
                {
                    "text": req.text[start:end],
                    "start": start,
                    "end": end,
                    "is_word": False,
                    "is_mwe": False,
                }
            )

    last = 0
    for w in words:
        if w.start > last:
            add_sep(last, w.start)
        entry: Dict[str, Any] = {
            "text": w.text,
            "start": w.start,
            "end": w.end,
            "is_word": True,
            "is_mwe": w.is_mwe,
        }
        if req.lang.startswith("zh"):
            try:
                from pypinyin import Style, lazy_pinyin  # type: ignore

                p_mark_list = lazy_pinyin(w.text, style=Style.TONE)
                p_num_list = lazy_pinyin(w.text, style=Style.TONE3)
                chars = []
                for i, ch in enumerate(w.text):
                    p_mark = p_mark_list[i] if i < len(p_mark_list) else None
                    p_num = p_num_list[i] if i < len(p_num_list) else None
                    chars.append(
                        {
                            "ch": ch,
                            "start": w.start + i,
                            "end": w.start + i + 1,
                            "pinyin": p_mark,
                            "pinyin_num": p_num,
                        }
                    )
                entry["chars"] = chars
            except Exception:
                pass
        tokens_out.append(entry)
        last = w.end
    add_sep(last, len(req.text))

    return {"tokens": tokens_out}

