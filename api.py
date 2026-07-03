"""FastAPI fallback for the podcast script generator.

Exposes the same backend pipeline as app.py / cli.py over HTTP with an
auto-generated Swagger UI at /docs. The API is stateless: each call after
/parse-documents takes the previously-parsed document text (and, for
/modify-script, the previous script's sections) directly in the request body
instead of relying on server-side session state.

Run with: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import DURATION_OPTIONS_MIN, LLM_PROVIDERS, PROVIDER_MODELS
from src.documents.parser import DocumentParseError, extract_text
from src.llm.base import LLMError
from src.llm.factory import get_llm_client
from src.script.generator import generate_script, modify_script
from src.script.models import ScriptResult, Section, Speaker
from src.topics.extractor import classify_manual_topics, extract_topics
from src.utils.validation import validate_setup_inputs

app = FastAPI(title="Podcast Script Generator API", version="1.0.0")

# The FastAPI application exposes the same backend pipeline as the UI and CLI.
# It is intentionally stateless: all document text and script state is passed in
# request bodies rather than stored on the server.

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LLMConfigIn(BaseModel):
    provider: str = Field(..., description=f"One of: {', '.join(LLM_PROVIDERS)}")
    api_key: str = ""
    model: str
    base_url: str = ""


class SpeakerIn(BaseModel):
    name: str
    gender: str
    speed: int = 100


class ParsedDocument(BaseModel):
    filename: str
    status: str
    detail: str


class ParseDocumentsResponse(BaseModel):
    documents: dict[str, str]
    results: list[ParsedDocument]


class ExtractTopicsRequest(BaseModel):
    documents: dict[str, str]
    llm: LLMConfigIn


class ExtractTopicsResponse(BaseModel):
    topics: list[str]
    thin_document: bool


class ClassifyTopicsRequest(BaseModel):
    documents: dict[str, str]
    extracted_topics: list[str] = []
    manual_topics: list[str]
    llm: LLMConfigIn


class ClassifyTopicsResponse(BaseModel):
    included: list[str]
    ignored: list[str]


class GenerateScriptRequest(BaseModel):
    documents: dict[str, str]
    topics: list[str]
    host: SpeakerIn
    guest: SpeakerIn
    duration_minutes: int
    llm: LLMConfigIn


class SectionOut(BaseModel):
    kind: str
    topic: Optional[str]
    word_budget: int
    text: str


class ScriptOut(BaseModel):
    sections: list[SectionOut]
    full_text: str
    target_words: int
    actual_words: int
    warnings: list[str]


class ModifyScriptRequest(BaseModel):
    documents: dict[str, str]
    existing_sections: list[SectionOut]
    target_words: int
    warnings: list[str] = []
    modification_instruction: str
    host: SpeakerIn
    guest: SpeakerIn
    llm: LLMConfigIn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_llm(cfg: LLMConfigIn):
    # Construct an LLM client from the incoming API payload and convert any
    # provider-specific errors into a FastAPI HTTPException.
    try:
        return get_llm_client(provider=cfg.provider, api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _to_script_out(result: ScriptResult) -> ScriptOut:
    # Convert internal result dataclass into the API response schema.
    return ScriptOut(
        sections=[SectionOut(kind=s.kind, topic=s.topic, word_budget=s.word_budget, text=s.text) for s in result.sections],
        full_text=result.full_text,
        target_words=result.target_words,
        actual_words=result.actual_words,
        warnings=result.warnings,
    )


def _from_sections(sections_in: list[SectionOut], target_words: int, warnings: list[str]) -> ScriptResult:
    # Reconstruct the internal ScriptResult object from incoming API payloads
    # when clients want to modify an existing script.
    result = ScriptResult(target_words=target_words, warnings=list(warnings))
    result.sections = [Section(kind=s.kind, topic=s.topic, word_budget=s.word_budget, text=s.text) for s in sections_in]
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Each endpoint forwards requests to the shared backend pipeline and converts
# internal exceptions into HTTP errors for clients.


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/providers")
def providers():
    return {"providers": LLM_PROVIDERS, "models": PROVIDER_MODELS, "durations_minutes": DURATION_OPTIONS_MIN}


@app.post("/parse-documents", response_model=ParseDocumentsResponse)
async def parse_documents(files: list[UploadFile] = File(...)):
    # Accept uploaded files, parse text from each, and report per-file status.
    documents = {}
    results = []
    for f in files:
        content = await f.read()
        try:
            text = extract_text(content, f.filename)
            documents[f.filename] = text
            results.append(ParsedDocument(filename=f.filename, status="ok", detail=f"{len(text.split())} words extracted"))
        except DocumentParseError as exc:
            results.append(ParsedDocument(filename=f.filename, status="error", detail=str(exc)))

    if not documents:
        raise HTTPException(status_code=422, detail="No document could be parsed successfully. " + "; ".join(r.detail for r in results if r.status == "error"))

    return ParseDocumentsResponse(documents=documents, results=results)


@app.post("/extract-topics", response_model=ExtractTopicsResponse)
def extract_topics_endpoint(req: ExtractTopicsRequest):
    if not req.documents:
        raise HTTPException(status_code=422, detail="At least one parsed document is required.")
    llm = _get_llm(req.llm)
    try:
        topics = extract_topics(req.documents, llm)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ExtractTopicsResponse(topics=topics, thin_document=not topics)


@app.post("/classify-topics", response_model=ClassifyTopicsResponse)
def classify_topics_endpoint(req: ClassifyTopicsRequest):
    llm = _get_llm(req.llm)
    try:
        included, ignored = classify_manual_topics(req.manual_topics, req.extracted_topics, req.documents, llm)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ClassifyTopicsResponse(included=included, ignored=ignored)


@app.post("/generate-script", response_model=ScriptOut)
def generate_script_endpoint(req: GenerateScriptRequest):
    errors = validate_setup_inputs(
        host_name=req.host.name,
        guest_name=req.guest.name,
        host_gender=req.host.gender,
        guest_gender=req.guest.gender,
        host_speed=req.host.speed,
        guest_speed=req.guest.speed,
        doc_texts=req.documents,
        duration_minutes=req.duration_minutes,
        llm_provider=req.llm.provider,
        llm_model=req.llm.model,
        llm_api_key=req.llm.api_key,
    )
    if not req.topics:
        errors.append("At least one topic must be selected.")
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    llm = _get_llm(req.llm)
    host = Speaker(name=req.host.name, gender=req.host.gender, speed=req.host.speed)
    guest = Speaker(name=req.guest.name, gender=req.guest.gender, speed=req.guest.speed)

    try:
        result = generate_script(
            topics=req.topics,
            host=host,
            guest=guest,
            duration_minutes=req.duration_minutes,
            doc_texts=req.documents,
            llm=llm,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _to_script_out(result)


@app.post("/modify-script", response_model=ScriptOut)
def modify_script_endpoint(req: ModifyScriptRequest):
    if not req.modification_instruction.strip():
        raise HTTPException(status_code=422, detail="modification_instruction must not be empty.")

    llm = _get_llm(req.llm)
    host = Speaker(name=req.host.name, gender=req.host.gender, speed=req.host.speed)
    guest = Speaker(name=req.guest.name, gender=req.guest.gender, speed=req.guest.speed)
    existing = _from_sections(req.existing_sections, req.target_words, req.warnings)

    try:
        result = modify_script(existing, req.modification_instruction, host, guest, req.documents, llm)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _to_script_out(result)
