from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from ..db import get_db
from ..models import LanguageWordList, Lexeme, UserLexeme, Profile
from ..deps import get_current_user

router = APIRouter()


class WordListCreate(BaseModel):
    lang: str
    list_name: str
    word: str
    pos: Optional[str] = None
    frequency_rank: Optional[int] = None
    frequency_score: Optional[float] = None
    level_code: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class WordListUpdate(BaseModel):
    frequency_rank: Optional[int] = None
    frequency_score: Optional[float] = None
    level_code: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class WordListOut(BaseModel):
    id: int
    lang: str
    list_name: str
    word: str
    pos: Optional[str] = None
    frequency_rank: Optional[int] = None
    frequency_score: Optional[float] = None
    level_code: Optional[str] = None
    category: Optional[str] = None
    tags: Dict[str, Any]
    created_at: datetime
    learned: Optional[bool] = None
    learning_progress: Optional[Dict[str, Any]] = None


@router.post("/wordlists", response_model=WordListOut)
def create_word_list_item(
    item: WordListCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Add a word to a language word list (admin only for now)"""
    existing = db.query(LanguageWordList).filter(
        LanguageWordList.lang == item.lang,
        LanguageWordList.list_name == item.list_name,
        LanguageWordList.word == item.word,
    ).first()
    
    if existing:
        raise HTTPException(400, "Word already exists in this list")
    
    word_list_item = LanguageWordList(
        lang=item.lang,
        list_name=item.list_name,
        word=item.word,
        pos=item.pos,
        frequency_rank=item.frequency_rank,
        frequency_score=item.frequency_score,
        level_code=item.level_code,
        category=item.category,
        tags=item.tags or {},
    )
    
    db.add(word_list_item)
    db.commit()
    db.refresh(word_list_item)
    
    return WordListOut(
        id=word_list_item.id,
        lang=word_list_item.lang,
        list_name=word_list_item.list_name,
        word=word_list_item.word,
        pos=word_list_item.pos,
        frequency_rank=word_list_item.frequency_rank,
        frequency_score=word_list_item.frequency_score,
        level_code=word_list_item.level_code,
        category=word_list_item.category,
        tags=word_list_item.tags,
        created_at=word_list_item.created_at,
        learned=None,
        learning_progress=None,
    )


@router.get("/wordlists", response_model=List[WordListOut])
def list_word_list_items(
    lang: str,
    list_name: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get words from language word lists, optionally filtered by list name"""
    query = db.query(LanguageWordList).filter(LanguageWordList.lang == lang)
    
    if list_name:
        query = query.filter(LanguageWordList.list_name == list_name)
    
    items = query.order_by(
        LanguageWordList.frequency_rank.asc(),
        LanguageWordList.word.asc()
    ).all()
    
    # Check which words the user has learned
    profile = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    user_lexeme_ids = set()
    learning_data = {}
    
    if profile:
        user_lexemes = db.query(UserLexeme).filter(
            UserLexeme.user_id == user.id,
            UserLexeme.profile_id == profile.id
        ).all()
        
        # Get lexemes for these user_lexemes
        lexeme_ids = [ul.lexeme_id for ul in user_lexemes]
        lexemes = db.query(Lexeme).filter(Lexeme.id.in_(lexeme_ids)).all()
        lexeme_to_word = {lex.id: lex.lemma for lex in lexemes}
        
        for ul in user_lexemes:
            word = lexeme_to_word.get(ul.lexeme_id)
            if word:
                user_lexeme_ids.add(word)
                learning_data[word] = {
                    "p_click": ul.a_click / (ul.a_click + ul.b_nonclick) if (ul.a_click + ul.b_nonclick) > 0 else 0.0,
                    "exposures": ul.exposures or 0,
                    "clicks": ul.clicks or 0,
                    "stability": ul.stability or 0.0,
                }
    
    result = []
    for item in items:
        learned = item.word in user_lexeme_ids
        progress = learning_data.get(item.word)
        
        result.append(WordListOut(
            id=item.id,
            lang=item.lang,
            list_name=item.list_name,
            word=item.word,
            pos=item.pos,
            frequency_rank=item.frequency_rank,
            frequency_score=item.frequency_score,
            level_code=item.level_code,
            category=item.category,
            tags=item.tags,
            created_at=item.created_at,
            learned=learned,
            learning_progress=progress,
        ))
    
    return result


@router.get("/wordlists/lists")
def list_available_lists(
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all available word list names for a language"""
    lists = db.query(LanguageWordList.list_name).filter(
        LanguageWordList.lang == lang
    ).distinct().all()
    
    return {"lists": [lst[0] for lst in lists]}


@router.get("/wordlists/stats")
def get_word_list_stats(
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get statistics about word lists for a language"""
    # Get total words in all lists for this language
    total_words = db.query(LanguageWordList).filter(
        LanguageWordList.lang == lang
    ).count()
    
    # Get list names and counts
    list_counts = db.query(
        LanguageWordList.list_name,
        db.func.count(LanguageWordList.id)
    ).filter(
        LanguageWordList.lang == lang
    ).group_by(LanguageWordList.list_name).all()
    
    # Get user's learned words count
    profile = db.query(Profile).filter(Profile.user_id == user.id, Profile.lang == lang).first()
    learned_count = 0
    if profile:
        user_lexemes = db.query(UserLexeme).filter(
            UserLexeme.user_id == user.id,
            UserLexeme.profile_id == profile.id
        ).all()
        
        lexeme_ids = [ul.lexeme_id for ul in user_lexemes]
        learned_words = db.query(Lexeme).filter(
            Lexeme.id.in_(lexeme_ids),
            Lexeme.lang == lang
        ).count()
        learned_count = learned_words
    
    return {
        "total_words": total_words,
        "learned_words": learned_count,
        "progress_percentage": round((learned_count / total_words * 100), 2) if total_words > 0 else 0.0,
        "lists": [{"name": name, "count": count} for name, count in list_counts],
    }


@router.delete("/wordlists/{word_list_id}")
def delete_word_list_item(
    word_list_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a word from a word list (admin only for now)"""
    item = db.query(LanguageWordList).filter(LanguageWordList.id == word_list_id).first()
    if not item:
        raise HTTPException(404, "Word list item not found")
    
    db.delete(item)
    db.commit()
    
    return {"ok": True, "message": "Word list item deleted"}


@router.patch("/wordlists/{word_list_id}", response_model=WordListOut)
def update_word_list_item(
    word_list_id: int,
    update: WordListUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update a word list item (admin only for now)"""
    item = db.query(LanguageWordList).filter(LanguageWordList.id == word_list_id).first()
    if not item:
        raise HTTPException(404, "Word list item not found")
    
    if update.frequency_rank is not None:
        item.frequency_rank = update.frequency_rank
    if update.frequency_score is not None:
        item.frequency_score = update.frequency_score
    if update.level_code is not None:
        item.level_code = update.level_code
    if update.category is not None:
        item.category = update.category
    if update.tags is not None:
        item.tags = update.tags
    
    db.commit()
    db.refresh(item)
    
    return WordListOut(
        id=item.id,
        lang=item.lang,
        list_name=item.list_name,
        word=item.word,
        pos=item.pos,
        frequency_rank=item.frequency_rank,
        frequency_score=item.frequency_score,
        level_code=item.level_code,
        category=item.category,
        tags=item.tags,
        created_at=item.created_at,
        learned=None,
        learning_progress=None,
    )
