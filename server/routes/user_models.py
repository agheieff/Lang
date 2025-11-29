"""
API routes for user model configuration.
"""
from __future__ import annotations

from typing import Optional, List, Literal, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator

from server.auth import Account
from server.deps import get_current_account
from server.account_db import get_db
from server.models import UserModelConfig
from server.services.user_model_service import get_user_model_service, TaskType

router = APIRouter(prefix="/api/models", tags=["models"])


# ==================== Schemas ====================

class ModelConfigCreate(BaseModel):
    display_name: str
    provider: str  # "openrouter", "openai", "anthropic", "local"
    model_id: str  # e.g. "anthropic/claude-3-sonnet"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    capabilities: Optional[List[str]] = None
    extra_params: Optional[dict] = None
    use_for_generation: bool = False
    use_for_word_translation: bool = False
    use_for_sentence_translation: bool = False
    priority: int = 100
    
    @field_validator('base_url', 'api_key', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v == '':
            return None
        return v
    
    @field_validator('max_tokens', mode='before')
    @classmethod
    def empty_str_to_none_int(cls, v: Any) -> Optional[int]:
        if v == '' or v is None:
            return None
        return int(v)


class ModelConfigUpdate(BaseModel):
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[dict] = None
    use_for_generation: Optional[bool] = None
    use_for_word_translation: Optional[bool] = None
    use_for_sentence_translation: Optional[bool] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    
    @field_validator('base_url', 'api_key', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v == '':
            return None
        return v
    
    @field_validator('max_tokens', 'priority', mode='before')
    @classmethod
    def empty_str_to_none_int(cls, v: Any) -> Optional[int]:
        if v == '' or v is None:
            return None
        return int(v)


class ModelConfigResponse(BaseModel):
    id: int
    display_name: str
    provider: str
    model_id: str
    base_url: Optional[str]
    api_key_masked: Optional[str]  # Masked API key for display
    source: str
    is_editable: bool
    is_key_visible: bool
    max_tokens: Optional[int]
    capabilities: List[str]
    use_for_generation: bool
    use_for_word_translation: bool
    use_for_sentence_translation: bool
    priority: int
    is_active: bool
    
    @classmethod
    def from_config(cls, config: UserModelConfig) -> "ModelConfigResponse":
        # Mask API key
        api_key_masked = None
        if config.api_key and config.is_key_visible:
            if len(config.api_key) > 8:
                api_key_masked = config.api_key[:4] + "****" + config.api_key[-4:]
            else:
                api_key_masked = "****"
        elif config.api_key and not config.is_key_visible:
            api_key_masked = "[hidden]"
        
        return cls(
            id=config.id,
            display_name=config.display_name,
            provider=config.provider,
            model_id=config.model_id,
            base_url=config.base_url,
            api_key_masked=api_key_masked,
            source=config.source,
            is_editable=config.is_editable,
            is_key_visible=config.is_key_visible,
            max_tokens=config.max_tokens,
            capabilities=config.capabilities or [],
            use_for_generation=config.use_for_generation,
            use_for_word_translation=config.use_for_word_translation,
            use_for_sentence_translation=config.use_for_sentence_translation,
            priority=config.priority,
            is_active=config.is_active,
        )


class TaskAssignment(BaseModel):
    task: TaskType
    model_config_id: int


# ==================== API Routes ====================

@router.get("", response_model=List[ModelConfigResponse])
def list_models(
    include_inactive: bool = False,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """List all model configurations for the current user."""
    service = get_user_model_service()
    
    # Ensure user has system models
    service.ensure_user_has_models(db, account.id, tier=getattr(account, "tier", "Free"))
    
    models = service.list_models(db, account.id, include_inactive=include_inactive)
    return [ModelConfigResponse.from_config(m) for m in models]


@router.post("", response_model=ModelConfigResponse)
def create_model(
    data: ModelConfigCreate,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Create a new user-defined model configuration."""
    service = get_user_model_service()
    
    config = service.create_model(
        db=db,
        account_id=account.id,
        display_name=data.display_name,
        provider=data.provider,
        model_id=data.model_id,
        base_url=data.base_url,
        api_key=data.api_key,
        max_tokens=data.max_tokens,
        capabilities=data.capabilities,
        extra_params=data.extra_params,
        use_for_generation=data.use_for_generation,
        use_for_word_translation=data.use_for_word_translation,
        use_for_sentence_translation=data.use_for_sentence_translation,
        priority=data.priority,
    )
    
    return ModelConfigResponse.from_config(config)


@router.get("/{model_config_id}", response_model=ModelConfigResponse)
def get_model(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Get a specific model configuration."""
    service = get_user_model_service()
    config = service.get_model(db, account.id, model_config_id)
    
    if not config:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelConfigResponse.from_config(config)


@router.patch("/{model_config_id}", response_model=ModelConfigResponse)
def update_model(
    model_config_id: int,
    data: ModelConfigUpdate,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Update a model configuration."""
    service = get_user_model_service()
    
    updates = data.model_dump(exclude_unset=True)
    config = service.update_model(db, account.id, model_config_id, **updates)
    
    if not config:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelConfigResponse.from_config(config)


@router.delete("/{model_config_id}")
def delete_model(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Delete a user-defined model configuration."""
    service = get_user_model_service()
    
    success = service.delete_model(db, account.id, model_config_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete this model (not found or not user-owned)")
    
    return {"ok": True}


@router.post("/{model_config_id}/activate")
def activate_model(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Activate a model."""
    service = get_user_model_service()
    success = service.set_model_active(db, account.id, model_config_id, True)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"ok": True}


@router.post("/{model_config_id}/deactivate")
def deactivate_model(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Deactivate a model."""
    service = get_user_model_service()
    success = service.set_model_active(db, account.id, model_config_id, False)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"ok": True}


@router.post("/assign")
def assign_model_to_task(
    data: TaskAssignment,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Assign a model to a specific task (generation, word_translation, sentence_translation)."""
    service = get_user_model_service()
    
    success = service.assign_model_to_task(
        db=db,
        account_id=account.id,
        model_config_id=data.model_config_id,
        task=data.task,
        exclusive=True
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"ok": True}


@router.get("/assigned/{task}", response_model=Optional[ModelConfigResponse])
def get_assigned_model(
    task: TaskType,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Get the model assigned to a specific task."""
    service = get_user_model_service()
    config = service.get_assigned_model(db, account.id, task)
    
    if not config:
        return None
    
    return ModelConfigResponse.from_config(config)


# ==================== HTMX HTML Routes ====================

htmx_router = APIRouter(prefix="/settings/models", tags=["settings-models"])


@htmx_router.get("", response_class=HTMLResponse)
def list_models_html(
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Returns the list of models as HTML."""
    service = get_user_model_service()
    tier = account.subscription_tier or "Free"
    service.ensure_user_has_models(db, account.id, tier=tier)
    models = service.list_models(db, account.id, include_inactive=True)
    
    if not models:
        return """
        <div class="text-gray-500 italic p-4 text-center">
            No models configured. System defaults will be used.
        </div>
        """
    
    html_parts = []
    for config in models:
        # Mask API key
        if config.api_key and config.is_key_visible:
            masked_key = "****" + config.api_key[-4:] if len(config.api_key) > 4 else "****"
        elif config.api_key:
            masked_key = "[system key]"
        else:
            masked_key = "Not set"
        
        # Build task badges
        tasks = []
        if config.use_for_generation:
            tasks.append('<span class="bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded">Generation</span>')
        if config.use_for_word_translation:
            tasks.append('<span class="bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">Words</span>')
        if config.use_for_sentence_translation:
            tasks.append('<span class="bg-purple-100 text-purple-800 text-xs px-2 py-0.5 rounded">Sentences</span>')
        tasks_html = " ".join(tasks) if tasks else '<span class="text-gray-400 text-xs">Not assigned</span>'
        
        # Source badge
        source_badge = {
            "user": '<span class="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded">User</span>',
            "system": '<span class="bg-yellow-100 text-yellow-800 text-xs px-2 py-0.5 rounded">System</span>',
            "subscription": '<span class="bg-indigo-100 text-indigo-800 text-xs px-2 py-0.5 rounded">Subscription</span>',
        }.get(config.source, "")
        
        # Active/inactive styling
        inactive_class = "opacity-50" if not config.is_active else ""
        
        # Action buttons
        actions = []
        if config.is_active:
            actions.append(f'''
                <button hx-post="/settings/models/{config.id}/deactivate" 
                        hx-target="#models-list" hx-swap="innerHTML"
                        class="text-gray-500 hover:text-gray-700 text-xs">Disable</button>
            ''')
        else:
            actions.append(f'''
                <button hx-post="/settings/models/{config.id}/activate" 
                        hx-target="#models-list" hx-swap="innerHTML"
                        class="text-blue-600 hover:text-blue-800 text-xs">Enable</button>
            ''')
        
        if config.source == "user":
            actions.append(f'''
                <button hx-delete="/settings/models/{config.id}" 
                        hx-confirm="Delete this model?"
                        hx-target="#models-list" hx-swap="innerHTML"
                        class="text-red-600 hover:text-red-800 text-xs">Delete</button>
            ''')
        
        # Only show key info for user-added models
        key_info = f" | Key: {masked_key}" if config.source == "user" else ""
        
        html = f"""
        <div class="border rounded-lg p-4 mb-3 {inactive_class}" id="model-{config.id}">
            <div class="flex justify-between items-start">
                <div class="flex-1">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="font-semibold">{config.display_name}</span>
                        {source_badge}
                    </div>
                    <div class="text-sm text-gray-600 mb-2">
                        <span class="font-mono text-xs bg-gray-100 px-1 rounded">{config.model_id}</span>
                    </div>
                    <div class="text-xs text-gray-500 mb-2">
                        Provider: {config.provider}{key_info}
                    </div>
                    <div class="flex gap-1">
                        {tasks_html}
                    </div>
                </div>
                <div class="flex flex-col gap-1 items-end">
                    {' '.join(actions)}
                </div>
            </div>
        </div>
        """
        html_parts.append(html)
    
    return "\n".join(html_parts)


@htmx_router.post("/{model_config_id}/activate", response_class=HTMLResponse)
def activate_model_html(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Activate a model and return updated list."""
    service = get_user_model_service()
    service.set_model_active(db, account.id, model_config_id, True)
    return list_models_html(account=account, db=db)


@htmx_router.post("/{model_config_id}/deactivate", response_class=HTMLResponse)
def deactivate_model_html(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Deactivate a model and return updated list."""
    service = get_user_model_service()
    service.set_model_active(db, account.id, model_config_id, False)
    return list_models_html(account=account, db=db)


@htmx_router.delete("/{model_config_id}", response_class=HTMLResponse)
def delete_model_html(
    model_config_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Delete a model and return updated list."""
    service = get_user_model_service()
    service.delete_model(db, account.id, model_config_id)
    return list_models_html(account=account, db=db)


def _get_show_advanced_setting(account: Account) -> bool:
    """Check if user wants advanced model assignment UI."""
    extras = account.extras or {}
    return extras.get("show_advanced_model_ui", False)


def _build_simple_assignments_html(models, current_model_id: Optional[int]) -> str:
    """Simple view: one dropdown for all tasks."""
    first_model_name = models[0].display_name if models else "system default"
    options = [f'<option value="">-- Auto (uses: {first_model_name}) --</option>']
    for m in models:
        selected = "selected" if m.id == current_model_id else ""
        options.append(f'<option value="{m.id}" {selected}>{m.display_name}</option>')
    
    return f"""
    <div class="space-y-4">
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Model for all tasks</label>
            <select name="all_model" 
                    hx-post="/settings/models/assign-all"
                    hx-target="#assignment-status"
                    hx-swap="innerHTML"
                    class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm">
                {''.join(options)}
            </select>
        </div>
        <div class="flex justify-between items-center">
            <div id="assignment-status" class="text-sm"></div>
            <button hx-post="/settings/models/toggle-advanced?show=true"
                    hx-target="#model-assignments"
                    hx-swap="innerHTML"
                    class="text-xs text-blue-600 hover:text-blue-800">
                More options...
            </button>
        </div>
    </div>
    """


def _build_advanced_assignments_html(models, gen_model, word_model, sent_model) -> str:
    """Advanced view: separate dropdown for each task."""
    def build_dropdown(task: str, current_id: Optional[int]) -> str:
        first_model_name = models[0].display_name if models else "system default"
        options = [f'<option value="">-- Auto (uses: {first_model_name}) --</option>']
        for m in models:
            selected = "selected" if m.id == current_id else ""
            options.append(f'<option value="{m.id}" {selected}>{m.display_name}</option>')
        
        return f"""
        <select name="{task}_model" 
                hx-post="/settings/models/assign/{task}"
                hx-target="#assignment-status"
                hx-swap="innerHTML"
                class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm">
            {''.join(options)}
        </select>
        """
    
    return f"""
    <div class="space-y-4">
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Text Generation</label>
            {build_dropdown("generation", gen_model.id if gen_model else None)}
        </div>
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Word Translations</label>
            {build_dropdown("word_translation", word_model.id if word_model else None)}
        </div>
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Sentence Translations</label>
            {build_dropdown("sentence_translation", sent_model.id if sent_model else None)}
        </div>
        <div class="flex justify-between items-center">
            <div id="assignment-status" class="text-sm"></div>
            <button hx-post="/settings/models/toggle-advanced?show=false"
                    hx-target="#model-assignments"
                    hx-swap="innerHTML"
                    class="text-xs text-blue-600 hover:text-blue-800">
                Less options
            </button>
        </div>
    </div>
    """


@htmx_router.get("/assignments", response_class=HTMLResponse)
def get_assignments_html(
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Returns the model assignment dropdowns (simple or advanced based on preference)."""
    service = get_user_model_service()
    models = service.list_models(db, account.id, include_inactive=False)
    
    show_advanced = _get_show_advanced_setting(account)
    
    if show_advanced:
        gen_model = service.get_assigned_model(db, account.id, "generation")
        word_model = service.get_assigned_model(db, account.id, "word_translation")
        sent_model = service.get_assigned_model(db, account.id, "sentence_translation")
        return _build_advanced_assignments_html(models, gen_model, word_model, sent_model)
    else:
        # For simple view, show the model used for generation (or first assigned)
        gen_model = service.get_assigned_model(db, account.id, "generation")
        current_id = gen_model.id if gen_model else None
        return _build_simple_assignments_html(models, current_id)


@htmx_router.post("/assign/{task}", response_class=HTMLResponse)
async def assign_model_html(
    task: TaskType,
    request: Request,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Assign a model to a task from form data."""
    form_data = await request.form()
    model_id_str = form_data.get(f"{task}_model", "")
    
    if not model_id_str:
        # Unassign all from this task
        task_field = f"use_for_{task}"
        db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account.id
        ).update({task_field: False})
        db.commit()
        return '<span class="text-green-600">Assignment cleared</span>'
    
    try:
        model_id = int(model_id_str)
        service = get_user_model_service()
        success = service.assign_model_to_task(db, account.id, model_id, task, exclusive=True)
        if success:
            return '<span class="text-green-600">Saved</span>'
        else:
            return '<span class="text-red-600">Model not found</span>'
    except ValueError:
        return '<span class="text-red-600">Invalid model ID</span>'


@htmx_router.post("/assign-all", response_class=HTMLResponse)
async def assign_all_model_html(
    request: Request,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Assign a single model to all tasks."""
    form_data = await request.form()
    model_id_str = form_data.get("all_model", "")
    
    service = get_user_model_service()
    
    if not model_id_str:
        # Clear all assignments
        for task in ["generation", "word_translation", "sentence_translation"]:
            task_field = f"use_for_{task}"
            db.query(UserModelConfig).filter(
                UserModelConfig.account_id == account.id
            ).update({task_field: False})
        db.commit()
        return '<span class="text-green-600">All assignments cleared</span>'
    
    try:
        model_id = int(model_id_str)
        # Assign to all three tasks
        for task in ["generation", "word_translation", "sentence_translation"]:
            service.assign_model_to_task(db, account.id, model_id, task, exclusive=True)
        return '<span class="text-green-600">Saved for all tasks</span>'
    except ValueError:
        return '<span class="text-red-600">Invalid model ID</span>'


@htmx_router.post("/toggle-advanced", response_class=HTMLResponse)
def toggle_advanced_view(
    show: bool,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """HTMX: Toggle between simple and advanced model assignment views."""
    from server.db import get_global_db
    
    # Update account extras in global DB (where accounts live)
    global_db = next(get_global_db())
    try:
        global_account = global_db.query(Account).filter(Account.id == account.id).first()
        if global_account:
            extras = global_account.extras or {}
            extras["show_advanced_model_ui"] = show
            global_account.extras = extras
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(global_account, "extras")
            global_db.commit()
    finally:
        global_db.close()
    
    # Return the appropriate view
    service = get_user_model_service()
    models = service.list_models(db, account.id, include_inactive=False)
    
    if show:
        gen_model = service.get_assigned_model(db, account.id, "generation")
        word_model = service.get_assigned_model(db, account.id, "word_translation")
        sent_model = service.get_assigned_model(db, account.id, "sentence_translation")
        return _build_advanced_assignments_html(models, gen_model, word_model, sent_model)
    else:
        gen_model = service.get_assigned_model(db, account.id, "generation")
        current_id = gen_model.id if gen_model else None
        return _build_simple_assignments_html(models, current_id)
