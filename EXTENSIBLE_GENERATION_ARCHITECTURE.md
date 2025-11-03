# Extensible Content Generation Architecture

## Problem Statement

The current text generation pipeline is well-designed for its specific purpose: generating individual reading texts with translations. However, the architecture faces challenges when presented with new generative tasks that deviate from the "one text → translations" pattern. 

### Current Architecture Strengths

1. **Excellent Separation of Concerns**
   - `TextGenerationService`: Pure text content creation
   - `TranslationService`: Pure translation generation  
   - `GenerationOrchestrator`: Coordination and state management
   - Each service is independently testable and replaceable

2. **Robust Infrastructure**
   - Comprehensive logging and retry mechanisms
   - Model registry for flexible LLM selection
   - Session management and transaction handling
   - Real-time notifications via SSE

3. **Clean Single Entry Point**
   - `ensure_text_available()` provides simple interface
   - Handles race conditions and duplicate generation prevention

### Architecture Limitations for New Use Cases

#### Problem 1: Rigid Data Model

The `ReadingText` model assumes:
- One content field per text
- One-to-one relationship with translations
- Linear progression through states

```python
# Current model - too rigid for branching
class ReadingText:
    id: int
    account_id: int  
    content: str  # Single content only
    # ... translations are separate but 1:1
```

**How this breaks for branching narratives:**
- Need to generate 2-3 content variants per decision point
- Need to track which variant was "chosen" vs "discarded"
- Need parent-child relationships between story nodes

#### Problem 2: Fixed Pipeline Assumptions

The orchestrator assumes a fixed workflow:
```
create_placeholder → generate_text → generate_translations → notify_ready
```

**How this breaks for other patterns:**
- Branching: generate_multiple_texts_in_parallel → wait_for_choice → generate_branch
- Interactive: generate_partial → wait_for_input → continue_generation  
- Multi-format: generate_text + generate_audio + generate_images

#### Problem 3: State Management Coupling

The `TextState` enum and state management are tightly coupled to the single-text use case:

```python
class TextState:
    NONE → GENERATING → CONTENT_READY → FULLY_READY → OPENED → READ
```

**How this breaks for complex workflows:**
- No concept of "siblings waiting for choice"
- No concept of "partial generation states" 
- No concept of "multi-modal readiness"

## Solution Approach: Generative Pipeline Framework

### Design Principles

1. **Preserve Current Simplicity**: Don't break existing works
2. **Extract Common Patterns**: Identify reusable pipeline components
3. **Enable Composition**: Allow mixing-and-matching pipeline steps
4. **Layered Extensibility**: Simple extensions first, complex later

### Core Abstraction: Generative Job

Instead of assuming "one text", we abstract to a "generative job" that produces arbitrary artifacts:

```python
@dataclass
class GenerativeJob:
    job_type: str  # "single_text", "branching_narrative", "interactive_story"
    account_id: int
    parameters: Dict[str, Any]  # Flexible parameter bag
    artifacts: List[GenerativeArtifact]  # What this job produces
    
class GenerativeArtifact:
    artifact_type: str  # "text_content", "translation", "audio", "choice_point"  
    metadata: Dict[str, Any]
    content: Any
    relationships: List[ArtifactRelationship]  # Links to other artifacts
```

### Pipeline Step Framework

Extract the current pipeline steps into composable components:

```python
class PipelineStep:
    def execute(self, context: PipelineContext) -> StepResult
    
class Context:
    job: GenerativeJob
    db: Session
    services: ServiceRegistry
    
# Current steps become concrete implementations:
class TextContentStep(PipelineStep): pass
class TranslationStep(PipelineStep): pass  
class NotificationStep(PipelineStep): pass
```

### Configurable Workflow Definitions

Instead of hardcoded pipeline, define workflows as configuration:

```python
# Single text workflow (current behavior)
SINGLE_TEXT_WORKFLOW = [
    "create_placeholder",
    "generate_text_content", 
    "generate_translations",
    "notify_ready"
]

# Branching narrative workflow  
BRANCHING_WORKFLOW = [
    "create_story_node",
    "generate_text_variants",  # Parallel step
    "generate_all_translations",  # Parallel step 
    "wait_for_user_choice",
    "selected_only_cleanup",
    "notify_ready"
]
```

### Flexible Data Model Evolution

Evolve the current models gradually:

#### Phase 1: Extend Existing Models
```python
# Add to ReadingText (minimal impact)
class ReadingText:
    # ... existing fields ...
    parent_text_id: Optional[int] = None  # For branching
    variant_index: Optional[int] = None   # A, B, C choices
    is_chosen: bool = False               # User selection tracking
```

#### Phase 2: Introduce Generative Artifacts
```python
class GenerativeArtifact:
    id: int
    job_id: str  # Links multiple artifacts
    artifact_type: str  # "text", "translation", "audio"
    parent_artifact_id: Optional[int]
    metadata: JSON  # Flexible per-artifact data
```

### Service Evolution Strategy

#### Current Services Stay Intact
- `TextGenerationService`: Still handles single text generation
- `TranslationService`: Still handles translation generation
- `GenerationOrchestrator`: Still handles single text orchestration

#### New Services Extend Capabilities
```python
class VariantGenerationService:
    """Extends TextGenerationService for multiple variants"""
    def generate_variants(self, job_id: str, specs: List[PromptSpec]) -> List[TextGenerationResult]
    
class StoryOrchestrator:
    """Extends GenerationOrchestrator for branching narratives"""
    def ensure_node_available(self, db, account_id, lang, parent_node_id: Optional[int])
    
class WorkflowEngine:
    """Configurable pipeline execution engine"""
    def execute_workflow(self, workflow: WorkflowDefinition, context: PipelineContext)
```

## Implementation Sketches

### Sketch 1: Minimal Branching Extension (Low Risk)

Add branching capability with minimal changes:

```python
# 1. Extend ReadingText model (add columns)
ALTER TABLE reading_texts ADD COLUMN parent_text_id INT NULL;
ALTER TABLE reading_texts ADD COLUMN variant_index INT NULL;
ALTER TABLE reading_texts ADD COLUMN is_chosen BOOLEAN DEFAULT FALSE;

# 2. Simple variant generation
class VariantGenerationService:
    def generate_branch_variants(self, db, account_id, lang, parent_id: int, count: int = 3):
        specs = [
            build_branch_prompt_spec(db, account_id, lang, parent_id, i) 
            for i in range(count)
        ]
        # Generate in parallel using existing TextGenerationService
        return [
            self.text_gen_service.generate_text_content(db, account_id, lang, text_id, job_dir, spec.messages)
            for spec in specs
        ]

# 3. API extensions  
@router.get("/reading/branch/{parent_id}")
def get_branch_options(parent_id: int, db):
    variants = db.query(ReadingText).filter(ReadingText.parent_text_id == parent_id).all()
    return {"variants": variants}

@router.post("/reading/choose/{variant_id}")
def choose_variant(variant_id: int, db):
    # Mark chosen, discard others
    chosen = db.get(ReadingText, variant_id)
    chosen.is_chosen = True
    # Discard siblings
    db.query(ReadingText).filter(
        ReadingText.parent_text_id == chosen.parent_text_id,
        ReadingText.id != chosen.id
    ).delete()
    db.commit()
```

Benefits: Small changes, uses existing infrastructure, quick to implement
Limitations: Still somewhat rigid, no proper story node concept

### Sketch 2: Workflow Framework Extension (Medium Risk)

Introduce the configurable workflow framework:

```python
# 1. Workflow definitions
WORKFLOWS = {
    "single_text": WorkflowDefinition(
        steps=[
            CreatePlaceholderStep(),
            GenerateTextStep(),
            GenerateTranslationsStep(), 
            NotifyReadyStep()
        ]
    ),
    "branching_story": WorkflowDefinition(
        steps=[
            CreateStoryNodeStep(),
            GenerateVariantsStep(parallel=True),  # Run variants in parallel
            GenerateAllTranslationsStep(parallel=True),
            WaitForChoiceStep(),
            CleanupUnchosenStep(),
            NotifyReadyStep()
        ]
    )
}

# 2. Enhanced orchestrator
class EnhancedGenerationOrchestrator:
    def ensure_content_available(self, db, account_id, job_type: str, **kwargs):
        workflow = WORKFLOWS[job_type]
        context = PipelineContext(
            job=GenerativeJob(job_type, account_id, kwargs),
            db=db,
            services=self.service_registry
        )
        return self.workflow_engine.execute(workflow, context)

# 3. Generic API
@router.post("/generation/start")
def start_generation(job_type: str, parameters: Dict):
    job_id = orchestrator.ensure_content_available(db, account.id, job_type, **parameters)
    return {"job_id": job_id}

@router.get("/generation/{job_id}/status")  
def get_generation_status(job_id: int):
    return workflow_engine.get_job_status(job_id)

@router.post("/generation/{job_id}/interact")
def interact_with_job(job_id: int, interaction: Dict):
    return workflow_engine.handle_interaction(job_id, interaction)
```

Benefits: Very flexible, supports many use cases, clean separation
Limitations: More complex, requires more testing and migration

### Sketch 3: Plugin Architecture (High Risk, High Reward)

Full plugin-based architecture for maximum extensibility:

```python
# 1. Plugin interfaces
class ContentGenerator(Protocol):
    def generate(self, context: GenerationContext) -> GenerationResult

class Processor(Protocol):
    def process(self, input: GenerationResult, context: GenerationContext) -> ProcessResult

class Notifier(Protocol):
    def notify(self, event: GenerationEvent, context: GenerationContext) -> None

# 2. Plugin registry
class PluginRegistry:
    def register_generator(self, name: str, generator: ContentGenerator): pass
    def register_processor(self, name: str, processor: Processor): pass
    def register_notifier(self, name: str, notifier: Notifier): pass

# 3. Configuration-driven pipelines
pipeline_config = {
    "branching_story": {
        "generators": ["text_variant_generator"],
        "processors": ["translation_processor", "choice_processor"],
        "notifiers": ["sse_notifier"]
    }
}

# 4. Runtime execution
class PipelineExecutor:
    def execute(self, pipeline_name: str, context: GenerationContext):
        config = self.pipelines[pipeline_name]
        
        # Generate phase
        for generator_name in config["generators"]:
            generator = self.registry.get_generator(generator_name)
            result = generator.generate(context)
            context.add_result(result)
        
        # Process phase
        for processor_name in config["processors"]:
            processor = self.registry.get_processor(processor_name)
            context = processor.process(context.results, context)
        
        # Notify phase
        for notifier_name in config["notifiers"]:
            notifier = self.registry.get_notifier(notifier_name)
            notifier.notify(context.events, context)
```

Benefits: Extremely flexible, supports any generative use case
Limitations: Very complex, high learning curve, overkill for current needs

## Migration Path Recommendation

### Phase 0: Keep Current System Intact
- Don't break existing functionality
- Document current architecture patterns

### Phase 1: Minimal Extensions (2-4 weeks)
- Add `parent_text_id` and `variant_index` to `ReadingText`
- Implement `VariantGenerationService` using existing services
- Add simple branching endpoints
- Test with pilot branching use case

### Phase 2: Workflow Framework (6-8 weeks after Phase 1)
- Design and implement workflow engine
- Migrate single text pipeline to workflow format
- Implement branching narrative workflow
- Add generic orchestration endpoints
- Gradual migration of existing code

### Phase 3: Plugin Architecture (Optional, Future)
- Based on real needs from Phases 1-2
- Consider if we have more diverse generative requirements
- Evaluate cost/benefit of full plugin system

## Conclusion

The current architecture is well-designed for its purpose but needs strategic evolution for branching narratives and other complex generative patterns. The key is to:

1. **Preserve current simplicity** - don't break what works
2. **Extract common patterns** - workflow engine as foundation  
3. **Enable gradual extension** - start with minimal changes
4. **Design for composition** - mix-and-match pipeline steps

This approach allows the system to remain maintainable for its current use case while becoming capable of handling more complex generative scenarios like the branching narrative RPG example.

The modular service architecture we have today provides an excellent foundation for this evolution - we just need to add the orchestration and data model layers to support more complex workflows.
