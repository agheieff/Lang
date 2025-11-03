- use uv for everything python
- always prefer ts over js
- make everything as modular as possible, so that it's easy to make adjustments/swap parts
- comments only where they actually have some value, explain why, not what

# Principles
- DRY - extremely
- Always aim to create reusable components, functions, or modules that can be shared
- Always prefer to rework the existing solutions over writing new duplicate code
- Do not worry about breaking things, this is a pre-production codebase
- No backcompatibility - feel free to overwrite anything, we can drop the db at any time
- Feel free to always go back and consolidate after making changes
- Good architecture
- Multiple smaller files, no monoliths
- IMPORTANT - don't make mistakes

# How the flow is supposed to work
- The user signs up
- GET /
- triggers GET /reading/current
- System sees no texts available - runs the generation once (the text
- The texts finishes generating - is sent to the server
- Server sends an SSE to the client - a text is ready - it's shown to the user
- Ensure text available function fires
- sees that the only generated text is opened, need to generate a new one
- sends request to llm for generating the second text
- at the same time, as soon as the first text finishes generating, sends the requests for words (parallel ones) and sentences translation
- As soon as the word and translations for the text 1 arrives they are put in the db
- the server sends SSE to the client - the text is fully ready, it updates with the translations (clickable words, sentences translations)
- all the words are recorded in local storage - schema for that largely in plnace
- Same for the 2nd text - when it finishes generating, the parallel requests are sent
- When the requests arrive, they are combined together and the translations are put in the db.
- The second text is fully ready - sse for that, next text button activates
- The user clicks next text - the db sets read_at for the first text, and opened_at for the second
- at the same time all the local storage info about the first text (all the words, and user's interactions with them) is sent to the server, and a process function (idk what it's called) is run on those from the server side
- wait for the process function to finish, then ...
- ensure text available fires - sees no unopened text, sends llm request for third text generation, then for translations, when those arrive and are put in the db, next text button activates
- etc

In the future we will enable:
- different sentence splitting
- multiple pregenerated texts
- more dynamic prompts based on the data from the db
