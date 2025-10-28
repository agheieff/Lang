def generate_text(language):
    GENERATING=1
    prompt = compose_text_prompt(language)
    result = call_llm(prompt)
    GENERATING=0
    return result

def generate_translations(language, text_id):
    prompt = compose_translation_prompt(text_id, language)
    result = call_llm(prompt)
    return result

def generation_flow(something):
    generate_text
    put text into db
    generate translations and
    generate words
    await then put translations into db
    await then put words into db

ensure_text_exists():
    if GENERATING=0 and in the db all texts have opened_at filled:
        generation_flow()

# some logs writing stuff
