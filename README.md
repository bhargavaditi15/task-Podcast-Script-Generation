# Podcast Script Generator

Generates a natural, two-speaker (Host/Guest) podcast script grounded in
user-uploaded documents, for a chosen set of topics and a target duration.
Three ways to run it: Streamlit UI (primary), a terminal CLI, and a FastAPI
service with Swagger docs -- all three call the exact same backend.

## Contents

```
app.py                 Streamlit UI (primary interface)
cli.py                 Terminal fallback
api.py                 FastAPI + Swagger fallback
src/
  config.py            Shared constants (durations, speed range, provider/model lists)
  llm/                  Provider-agnostic LLM abstraction
    base.py             LLMClient interface + LLMError
    providers.py        OpenAI / Anthropic / Gemini / Groq / Custom-endpoint / Mock adapters
    factory.py          get_llm_client(provider, api_key, model, base_url)
  documents/
    parser.py           PDF/DOCX/TXT text extraction with edge-case handling
    chunker.py           Paragraph-aware chunking for large documents
  topics/
    extractor.py        Topic extraction (map-reduce over chunks) + manual-topic grounding check
  script/
    planner.py          Duration + speaking speed -> deterministic per-section word budgets
    retrieval.py        Keyword-overlap excerpt retrieval per topic
    generator.py        Section-by-section generation and full-script modification
    models.py           Speaker / Section / ScriptResult dataclasses
  utils/
    validation.py       Consolidated pre-flow validation
    jsonutil.py          Best-effort JSON parsing from LLM output
sample_docs/            Two sample source documents used to produce the samples below
sample_outputs/         Pre-generated sample scripts for quick review
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional -- only needed to pre-fill API keys
```

Nothing in `.env` is required at import time. You can also skip it entirely
and paste an API key directly into the Streamlit sidebar or pass it as a CLI
flag / API request field each time.

For a full step-by-step install and environment guide, see `SETUP_GUIDE.md`.

## Running the Streamlit UI (primary)

```bash
streamlit run app.py
```

1. **Sidebar -- LLM configuration**: pick a provider, paste an API key (not
   needed for Mock or a keyless local endpoint), pick a model, click **Test
   connection**. This is the "any LLM, configurable" requirement -- swap
   providers/models without touching code.
2. **Step 1 -- Setup**: speaker names/genders/speaking speeds, upload one or
   more PDF/DOC/DOCX/TXT files, pick a target duration, click **Start**. If
   anything mandatory is missing, one consolidated error lists every problem
   at once.
3. **Step 2 -- Topics**: review extracted topics, check/uncheck them, optionally
   type extra topics to check against the documents (shown as *Included* vs
   *Ignored*), then **Approve and continue** or **Restart flow**.
4. **Step 3 -- Review & modify**: read the generated script, download it, or
   type a modification instruction and **Regenerate** -- every regeneration
   returns a brand-new complete script (never a partial patch), and you can do
   this as many times as you like. **Restart flow** resets everything.

## Running the CLI fallback

```bash
python cli.py \
  --docs sample_docs/electric_vehicles.txt \
  --host-name Asha --guest-name Rahul \
  --host-gender female --guest-gender male \
  --host-speed 100 --guest-speed 90 \
  --duration 10 \
  --provider "Mock (offline/dev)" --model mock-1 \
  --output sample_outputs/my_script.txt
```

- Omit `--topics` to auto-select every extracted topic.
- `--extra-topics "A, B"` runs the same included/ignored grounding check as
  the UI and prints both lists.
- `--list-topics-only` extracts and prints topics, then exits (useful for
  reviewing before committing to a full generation run).
- Without `--non-interactive`, after the first script is generated you're
  dropped into an interactive loop: type a modification instruction (or press
  Enter to finish) and it regenerates the full script each time, mirroring
  the UI's review/modify loop.
- Run `python cli.py --help` for the full flag list. Swap `--provider` /
  `--model` / `--api-key` / `--base-url` to point at any supported LLM.

## Running the FastAPI fallback (Swagger)

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger docs. The API is
stateless: `/parse-documents` returns parsed text, which you pass directly in
the body of `/extract-topics`, `/classify-topics`, `/generate-script`; the
sections returned by `/generate-script` are passed back into
`/modify-script` along with a new instruction to run the modification loop
over HTTP. `GET /providers` lists supported providers/models/durations.

## Supported LLMs (configurable, no code changes needed)

| Provider | Env var (optional pre-fill) | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | |
| Google Gemini | `GOOGLE_API_KEY` | |
| Groq | `GROQ_API_KEY` | |
| Custom (OpenAI-compatible) | `CUSTOM_LLM_API_KEY` + `CUSTOM_LLM_BASE_URL` | Point at a local Ollama/vLLM/LM Studio server; API key optional |
| Mock (offline/dev) | none | Deterministic, no network calls -- used to produce `sample_outputs/` and to let you exercise the whole app with zero API cost |

No secrets are committed -- `.env` is git-ignored, `.env.example` only has
placeholders, and the UI/CLI/API all accept a key at runtime instead of a
hardcoded one.

## Design notes / how edge cases are handled

- **Consolidated validation**: `src/utils/validation.py` returns every
  problem at once (missing name, no document, no duration, no API key, etc.)
  instead of a fix-one-at-a-time loop.
- **Length approximation is deterministic, not LLM-guessed**:
  `src/script/planner.py` converts `(duration, host speed, guest speed)`
  into a target word count and allocates it across opening / each topic /
  closing *before* any LLM call. Each section is then generated against its
  own word budget and stitched together section-by-section (carrying the
  previous section's last lines forward for a smooth transition). This is
  what keeps a 45-60 minute script (thousands of words) from truncating or
  drifting wildly off-length, which a single "write the whole script" prompt
  would risk.
- **Grounding**: topics come from the documents (map-reduce extraction over
  chunks); each topic section is generated against a keyword-retrieved
  excerpt from the source documents, and the system prompt explicitly
  forbids inventing facts not in the excerpt.
- **Included vs ignored**: manually-typed topics are checked against the
  documents with an LLM grounding call (topics already found by extraction
  are auto-included without an extra call).
- **Thin document**: if topic extraction returns nothing, the user is
  notified and the flow restarts (per spec) rather than proceeding with an
  empty topic list.
- **Restart**: resets every input and selection (speakers, documents, topics,
  duration, script, and the file-uploader widget itself).
- **Modification loop**: unlimited cycles; every regeneration re-runs the
  full section-by-section pipeline with the instruction applied, so the
  output is always a complete script, never a diff.
- **Manual topic grounding**: user-entered extra topics are checked against the
  source documents and classified as included or ignored before the final script is generated.
- **LLM failures**: invalid key / unknown model / rate limits / timeouts are
  caught, retried a couple of times with backoff where transient, and
  surfaced as an actionable message rather than a stack trace. "Test
  connection" catches bad credentials before the real flow starts.
- Other handled cases: corrupted/empty/password-protected files, legacy
  `.doc` (asks for `.docx`/PDF instead), non-UTF8 `.txt`, duplicate topics,
  zero topics selected, oversized documents (chunked), too many topics for a
  short duration (coverage shrinks per topic with a warning instead of
  failing).

## Sample outputs

`sample_outputs/` contains two full scripts generated from
`sample_docs/electric_vehicles.txt` and `sample_docs/remote_work.txt` using
the **Mock (offline/dev)** provider, since this environment has no real LLM
API key configured. The Mock provider proves the full pipeline end-to-end
(parsing -> extraction -> grounded per-section generation -> assembly) with
templated text and no API cost, but it does **not** demonstrate the
conversational quality or precise length-matching a real model produces --
those depend on the actual LLM you configure. Point the same code at OpenAI,
Anthropic, Gemini, or Groq with a real key to get natural dialogue that
actually hits the requested word budget per section.

# task-Podcast-Script-Generation