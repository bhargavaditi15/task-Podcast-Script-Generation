"""Streamlit UI for the podcast script generator.

Flow: Setup & validate -> Topic selection -> Script generation -> Review &
modify (loop). See README.md for how to run this and the CLI/API fallbacks.
"""

import html
import os

import streamlit as st
from dotenv import load_dotenv

from src.config import DURATION_OPTIONS_MIN, GENDER_OPTIONS, LLM_PROVIDERS, PROVIDER_MODELS, SPEED_DEFAULT, SPEED_MAX, SPEED_MIN
from src.documents.parser import DocumentParseError, extract_text
from src.llm.base import LLMError
from src.llm.factory import get_llm_client
from src.script.generator import generate_script, modify_script
from src.script.models import Speaker
from src.topics.extractor import classify_manual_topics, extract_topics
from src.utils.validation import validate_setup_inputs

load_dotenv()

st.set_page_config(page_title="Podcast Script Generator", page_icon="🎙️", layout="wide")

PROVIDER_ENV_VARS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google Gemini": "GOOGLE_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Custom (OpenAI-compatible)": "CUSTOM_LLM_API_KEY",
}

STEPS = [("setup", "Setup"), ("topics", "Topics"), ("review", "Review")]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULTS = {
    "stage": "setup",
    "uploader_version": 0,
    "doc_texts": {},
    "doc_status": [],  # list of (filename, "ok"/"error", detail)
    "host_name": "",
    "guest_name": "",
    "host_gender": None,
    "guest_gender": None,
    "host_speed": SPEED_DEFAULT,
    "guest_speed": SPEED_DEFAULT,
    "duration_minutes": None,
    "confirmed_host": None,
    "confirmed_guest": None,
    "confirmed_duration_minutes": None,
    "extracted_topics": [],
    "selected_topics": set(),
    "manual_topics_text": "",
    "included_manual": [],
    "ignored_manual": [],
    "approved_topics": [],
    "script_history": [],
    "review_version_index": 0,
    "thin_doc_notice": None,
}


def init_session_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_pending_restart():
    """Actually perform the reset, if one was requested on the previous run.

    Must run at the very top of main(), before any widget is instantiated --
    Streamlit forbids writing to a widget's session_state key in the same run
    where that widget has already been drawn (StreamlitAPIException). Since
    restart is always requested from a button click deep inside a stage
    (after upstream widgets already rendered this run), the reset itself has
    to happen on the *next* run instead of immediately.
    """
    if st.session_state.get("_restart_requested"):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.session_state["uploader_version"] += 1  # forces the file_uploader widget to remount empty
        st.session_state["_restart_requested"] = False


def restart_flow():
    """Request a full reset of every input and selection, per the task's
    restart requirement. Only sets a plain (non-widget) flag -- call st.rerun()
    right after this so apply_pending_restart() performs the reset on the next run.
    """
    st.session_state["_restart_requested"] = True


@st.dialog("Restart flow?")
def confirm_restart_dialog():
    st.warning("This clears every input, selection, and generated script. This can't be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Yes, restart", type="primary", use_container_width=True):
            restart_flow()
            st.rerun()


# ---------------------------------------------------------------------------
# Shared chrome: header + stepper
# ---------------------------------------------------------------------------


def render_stepper(current_stage):
    order = [key for key, _ in STEPS]
    current_i = order.index(current_stage)

    parts = []
    for i, (key, label) in enumerate(STEPS):
        state = "done" if i < current_i else "active" if i == current_i else "upcoming"
        circle = "&#10003;" if state == "done" else str(i + 1)
        parts.append(
            f'<div class="step {state}"><div class="step-circle">{circle}</div>'
            f'<div class="step-label">{html.escape(label)}</div></div>'
        )
        if i < len(STEPS) - 1:
            parts.append(f'<div class="step-connector {"done" if i < current_i else "upcoming"}"></div>')

    st.markdown(
        f"""
        <div class="stepper">{"".join(parts)}</div>
        <style>
        .stepper {{ display: flex; align-items: flex-start; margin: 4px 0 8px 0; }}
        .step {{ display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 64px; }}
        .step-circle {{
            width: 30px; height: 30px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
            border: 2px solid rgba(128,128,128,0.35); color: inherit;
        }}
        .step.active .step-circle {{ border-color: #6366f1; background: #6366f1; color: white; }}
        .step.done .step-circle {{ border-color: #10b981; background: #10b981; color: white; }}
        .step-label {{ font-size: 12px; opacity: 0.7; white-space: nowrap; }}
        .step.active .step-label {{ font-weight: 700; opacity: 1; }}
        .step-connector {{ flex: 1; height: 2px; background: rgba(128,128,128,0.25); margin: 14px 6px 0 6px; }}
        .step-connector.done {{ background: #10b981; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# LLM configuration sidebar
# ---------------------------------------------------------------------------


def render_llm_sidebar():
    st.sidebar.markdown("### 🎙️ Podcast Script Generator")
    st.sidebar.caption("Host/Guest dialogue, grounded in your documents.")
    st.sidebar.divider()
    st.sidebar.subheader("LLM configuration")
    provider = st.sidebar.selectbox("Provider", LLM_PROVIDERS, key="llm_provider")

    base_url = ""
    if provider == "Custom (OpenAI-compatible)":
        base_url = st.sidebar.text_input(
            "Base URL", value=os.environ.get("CUSTOM_LLM_BASE_URL", ""), key="llm_base_url",
            help="e.g. http://localhost:11434/v1 for a local Ollama server",
        )

    api_key = ""
    if provider != "Mock (offline/dev)":
        env_var = PROVIDER_ENV_VARS.get(provider, "")
        default_key = os.environ.get(env_var, "") if env_var else ""
        api_key = st.sidebar.text_input("API key", value=default_key, type="password", key="llm_api_key")

    model_options = PROVIDER_MODELS.get(provider, []) + ["Custom / other..."]
    model_choice = st.sidebar.selectbox("Model", model_options, key="llm_model_choice")
    if model_choice == "Custom / other...":
        model = st.sidebar.text_input("Enter model id", key="llm_model_custom")
    else:
        model = model_choice

    if st.sidebar.button("Test connection", use_container_width=True):
        try:
            client = get_llm_client(provider=provider, api_key=api_key, model=model, base_url=base_url)
            client.test_connection()
            st.sidebar.success(f"Connected to {provider} ({model}).")
        except LLMError as exc:
            st.sidebar.error(str(exc))

    return provider, api_key, model, base_url


def get_llm_or_none(provider, api_key, model, base_url):
    try:
        return get_llm_client(provider=provider, api_key=api_key, model=model, base_url=base_url), None
    except LLMError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Stage 1: setup
# ---------------------------------------------------------------------------


def handle_uploads(uploaded_files):
    doc_texts = {}
    doc_status = []
    for f in uploaded_files or []:
        content = f.getvalue()
        try:
            text = extract_text(content, f.name)
            doc_texts[f.name] = text
            word_count = len(text.split())
            doc_status.append((f.name, "ok", f"{word_count} words extracted"))
        except DocumentParseError as exc:
            doc_status.append((f.name, "error", str(exc)))
    st.session_state.doc_texts = doc_texts
    st.session_state.doc_status = doc_status


def render_setup_stage(provider, api_key, model, base_url):
    st.subheader("1. Speaker information")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**🎙️ Host**")
            st.text_input("Name", key="host_name", placeholder="e.g. Asha")
            st.selectbox("Gender", GENDER_OPTIONS, key="host_gender", index=None, placeholder="Select...")
            st.slider("Speaking speed (slow to fast)", SPEED_MIN, SPEED_MAX, key="host_speed")
    with col2:
        with st.container(border=True):
            st.markdown("**🎤 Guest**")
            st.text_input("Name", key="guest_name", placeholder="e.g. Rahul")
            st.selectbox("Gender", GENDER_OPTIONS, key="guest_gender", index=None, placeholder="Select...")
            st.slider("Speaking speed (slow to fast)", SPEED_MIN, SPEED_MAX, key="guest_speed")

    st.subheader("2. Source documents")
    uploaded_files = st.file_uploader(
        "Upload at least one PDF, DOC/DOCX, or TXT file",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_version}",
    )
    if uploaded_files:
        handle_uploads(uploaded_files)
    for name, status, detail in st.session_state.doc_status:
        (st.success if status == "ok" else st.error)(f"{name}: {detail}")
    if st.button("__DEBUG_LOAD_SAMPLE__"):
        import pathlib
        p = pathlib.Path("sample_docs/electric_vehicles.txt")
        st.session_state.doc_texts = {p.name: p.read_text(encoding="utf-8")}
        st.session_state.doc_status = [(p.name, "ok", "debug-loaded")]
        st.rerun()

    st.subheader("3. Target duration")
    st.selectbox(
        "Approximate duration (minutes)",
        DURATION_OPTIONS_MIN,
        key="duration_minutes",
        index=None,
        placeholder="Select duration...",
    )

    st.divider()

    if st.session_state.thin_doc_notice:
        st.warning(st.session_state.thin_doc_notice)
        if st.button("⚠️ Restart and upload different document(s)"):
            confirm_restart_dialog()
        return

    if st.button("Start ->", type="primary"):
        errors = validate_setup_inputs(
            host_name=st.session_state.host_name,
            guest_name=st.session_state.guest_name,
            host_gender=st.session_state.host_gender,
            guest_gender=st.session_state.guest_gender,
            host_speed=st.session_state.host_speed,
            guest_speed=st.session_state.guest_speed,
            doc_texts=st.session_state.doc_texts,
            duration_minutes=st.session_state.duration_minutes,
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
        )
        if errors:
            st.error("Please fix the following before continuing:\n\n" + "\n".join(f"- {e}" for e in errors))
            return

        llm, err = get_llm_or_none(provider, api_key, model, base_url)
        if err:
            st.error(f"LLM configuration problem: {err}")
            return

        with st.spinner("Extracting candidate topics from your document(s)..."):
            try:
                topics = extract_topics(st.session_state.doc_texts, llm)
            except LLMError as exc:
                st.error(f"Topic extraction failed: {exc}")
                return

        if not topics:
            st.session_state.thin_doc_notice = (
                "No extractable topics were found in the uploaded document(s) -- they may be too thin or lack "
                "substantive content. Please restart and upload additional or more detailed document(s)."
            )
            st.rerun()
            return

        # Snapshot setup values into non-widget keys: once we leave this stage,
        # render_setup_stage (and its widgets) stop running, and Streamlit
        # clears session_state for any widget not rendered in the current run --
        # so host_name/duration_minutes/etc. would otherwise revert to None.
        st.session_state.confirmed_host = Speaker(
            st.session_state.host_name, st.session_state.host_gender, st.session_state.host_speed
        )
        st.session_state.confirmed_guest = Speaker(
            st.session_state.guest_name, st.session_state.guest_gender, st.session_state.guest_speed
        )
        st.session_state.confirmed_duration_minutes = st.session_state.duration_minutes

        st.session_state.extracted_topics = topics
        st.session_state.selected_topics = set(topics)
        st.session_state.stage = "topics"
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 2: topic selection
# ---------------------------------------------------------------------------


def render_topics_stage(provider, api_key, model, base_url):
    st.subheader("Extracted topics")
    st.caption("Review the topics found in your document(s) and choose which ones to include.")

    sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 3])
    with sel_col1:
        if st.button("Select all", use_container_width=True):
            st.session_state.selected_topics = set(st.session_state.extracted_topics)
            st.rerun()
    with sel_col2:
        if st.button("Clear all", use_container_width=True):
            st.session_state.selected_topics = set()
            st.rerun()
    with sel_col3:
        st.markdown(
            f"<div style='padding-top:8px;'>{len(st.session_state.selected_topics)} of "
            f"{len(st.session_state.extracted_topics)} selected</div>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        for topic in st.session_state.extracted_topics:
            label = f"{topic}  · :blue[manually added]" if topic in st.session_state.included_manual else topic
            checked = st.checkbox(label, value=topic in st.session_state.selected_topics, key=f"topic_chk_{topic}")
            if checked:
                st.session_state.selected_topics.add(topic)
            else:
                st.session_state.selected_topics.discard(topic)

    st.subheader("Add extra topics")
    st.caption("Type your own topics; we'll check them against your uploaded documents.")
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        st.text_input("Comma-separated extra topics", key="manual_topics_text", label_visibility="collapsed")
    with col_btn:
        check_clicked = st.button("Check", use_container_width=True)

    if check_clicked:
        manual = [t.strip() for t in st.session_state.manual_topics_text.split(",") if t.strip()]
        if not manual:
            st.info("Enter at least one topic to check.")
        else:
            llm, err = get_llm_or_none(provider, api_key, model, base_url)
            if err:
                st.error(f"LLM configuration problem: {err}")
            else:
                with st.spinner("Checking topics against your documents..."):
                    try:
                        included, ignored = classify_manual_topics(
                            manual, st.session_state.extracted_topics, st.session_state.doc_texts, llm
                        )
                    except LLMError as exc:
                        st.error(f"Topic check failed: {exc}")
                        included, ignored = [], []
                st.session_state.included_manual = included
                st.session_state.ignored_manual = ignored
                for t in included:
                    st.session_state.selected_topics.add(t)
                    if t not in st.session_state.extracted_topics:
                        st.session_state.extracted_topics.append(t)
                st.rerun()

    if st.session_state.included_manual or st.session_state.ignored_manual:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Topics included** _(found in documents)_")
            for t in st.session_state.included_manual:
                st.write(f"- {t}")
        with col2:
            st.markdown("**🚫 Topics ignored** _(not found in documents)_")
            for t in st.session_state.ignored_manual:
                st.write(f"- {t}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚠️ Restart flow"):
            confirm_restart_dialog()
    with col_b:
        if st.button("Approve and continue ->", type="primary"):
            selected = list(st.session_state.selected_topics)
            if not selected:
                st.error("Select at least one topic before continuing.")
                return

            llm, err = get_llm_or_none(provider, api_key, model, base_url)
            if err:
                st.error(f"LLM configuration problem: {err}")
                return

            host = st.session_state.confirmed_host
            guest = st.session_state.confirmed_guest

            progress_bar = st.progress(0.0, text="Generating script...")

            def on_progress(i, n, section):
                label = section.topic or section.kind
                progress_bar.progress(i / n, text=f"Generated {i}/{n}: {label}")

            try:
                result = generate_script(
                    topics=selected,
                    host=host,
                    guest=guest,
                    duration_minutes=st.session_state.confirmed_duration_minutes,
                    doc_texts=st.session_state.doc_texts,
                    llm=llm,
                    progress_callback=on_progress,
                )
            except LLMError as exc:
                st.error(f"Script generation failed: {exc}")
                return

            st.session_state.approved_topics = selected
            st.session_state.script_history = [result]
            st.session_state.review_version_index = 0
            st.session_state.stage = "review"
            st.rerun()


# ---------------------------------------------------------------------------
# Stage 3: review & modify -- chat-style rendering helpers
# ---------------------------------------------------------------------------


def parse_script_to_messages(full_text):
    """Turn 'HOST: ...' / 'GUEST: ...' lines into [(speaker, text), ...]."""
    messages = []
    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("HOST:"):
            messages.append(["HOST", line[len("HOST:"):].strip()])
        elif upper.startswith("GUEST:"):
            messages.append(["GUEST", line[len("GUEST:"):].strip()])
        elif messages:
            messages[-1][1] += " " + line  # wrapped continuation of the previous turn
        else:
            messages.append(["HOST", line])
    return [(speaker, text) for speaker, text in messages]


def render_chat_messages(messages, host_name, guest_name):
    """Render the whole conversation as animated chat bubbles in one
    scrollable panel -- no pagination here, the panel just scrolls.
    """
    if not messages:
        st.info("This script has no dialogue lines to show.")
        return

    rows = []
    for i, (speaker, text) in enumerate(messages):
        is_host = speaker == "HOST"
        name = (host_name if is_host else guest_name) or ("Host" if is_host else "Guest")
        initial = html.escape((name.strip()[:1] or "?").upper())
        side = "host" if is_host else "guest"
        delay = min(i * 0.06, 2.5)
        safe_name = html.escape(name)
        safe_text = html.escape(text).replace("\n", "<br/>")
        rows.append(
            f'<div class="chat-row {side}" style="animation-delay:{delay:.2f}s">'
            f'<div class="chat-avatar {side}">{initial}</div>'
            f'<div class="chat-bubble {side}">'
            f'<div class="chat-name">{safe_name}</div>'
            f'<div class="chat-text">{safe_text}</div>'
            f"</div></div>"
        )

    st.markdown(
        f"""
        <div class="chat-container">{"".join(rows)}</div>
        <style>
        .chat-container {{
            display: flex; flex-direction: column; gap: 12px;
            padding: 16px; max-height: 560px; overflow-y: auto;
            border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        }}
        .chat-row {{
            display: flex; align-items: flex-end; gap: 8px;
            opacity: 0; animation: chatFadeIn 0.4s ease forwards;
        }}
        .chat-row.host {{ justify-content: flex-start; }}
        .chat-row.guest {{ justify-content: flex-end; flex-direction: row-reverse; }}
        .chat-avatar {{
            width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 700; color: white;
        }}
        .chat-avatar.host {{ background: #6366f1; }}
        .chat-avatar.guest {{ background: #10b981; }}
        .chat-bubble {{
            max-width: 68%; padding: 8px 14px; border-radius: 16px;
            font-size: 14px; line-height: 1.45; color: inherit;
        }}
        .chat-bubble.host {{
            background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35);
            border-bottom-left-radius: 4px;
        }}
        .chat-bubble.guest {{
            background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35);
            border-bottom-right-radius: 4px;
        }}
        .chat-name {{ font-size: 11px; font-weight: 700; opacity: 0.65; margin-bottom: 3px; }}
        @keyframes chatFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_version_pagination():
    """Pagination control for switching between generated script versions.
    This is the only pagination in the review stage -- the conversation
    itself is a single scrollable panel, not paginated.
    """
    history = st.session_state.script_history
    n = len(history)
    current = st.session_state.review_version_index
    if n <= 1:
        return

    st.markdown("<div style='font-size:12px; opacity:0.65; margin-bottom:4px;'>VERSION</div>", unsafe_allow_html=True)

    if n <= 8:
        cols = st.columns(n + 2)
        with cols[0]:
            if st.button("‹", disabled=current == 0, use_container_width=True, key="pg_prev"):
                st.session_state.review_version_index = current - 1
                st.rerun()
        for i in range(n):
            with cols[i + 1]:
                if st.button(
                    str(i + 1), type="primary" if i == current else "secondary",
                    use_container_width=True, key=f"pg_v{i}",
                ):
                    st.session_state.review_version_index = i
                    st.rerun()
        with cols[-1]:
            if st.button("›", disabled=current == n - 1, use_container_width=True, key="pg_next"):
                st.session_state.review_version_index = current + 1
                st.rerun()
    else:
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            if st.button("‹ Older", disabled=current == 0, use_container_width=True, key="pg_prev_wide"):
                st.session_state.review_version_index = current - 1
                st.rerun()
        with c2:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px;'>Version {current + 1} of {n}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("Newer ›", disabled=current == n - 1, use_container_width=True, key="pg_next_wide"):
                st.session_state.review_version_index = current + 1
                st.rerun()


# ---------------------------------------------------------------------------
# Stage 3: review & modify
# ---------------------------------------------------------------------------


def render_review_stage(provider, api_key, model, base_url):
    history = st.session_state.script_history
    st.session_state.review_version_index = max(0, min(st.session_state.review_version_index, len(history) - 1))
    version_index = st.session_state.review_version_index
    result = history[version_index]

    render_version_pagination()

    implied_wpm = result.target_words / max(1, st.session_state.confirmed_duration_minutes)
    approx_minutes = result.actual_words / max(1, implied_wpm)
    st.info(
        f"Target: ~{result.target_words} words (~{st.session_state.confirmed_duration_minutes} min) | "
        f"Actual: {result.actual_words} words (~{approx_minutes:.1f} min)"
    )
    for w in result.warnings:
        st.warning(w)

    messages = parse_script_to_messages(result.full_text)
    render_chat_messages(messages, st.session_state.confirmed_host.name, st.session_state.confirmed_guest.name)

    dl_col, txt_col = st.columns([1, 1])
    with dl_col:
        st.download_button(
            "⬇️ Download this version (.txt)",
            data=result.full_text,
            file_name=f"podcast_script_v{version_index + 1}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with txt_col:
        with st.popover("View as plain text", use_container_width=True):
            st.text_area(
                "Full script", value=result.full_text, height=400,
                key=f"script_text_v{version_index}", label_visibility="collapsed",
            )

    st.divider()
    st.subheader("Request a modification")
    st.caption("Applied on top of the version you're currently viewing above -- creates a new version.")
    st.text_input("Describe what to change (e.g. \"make the guest more skeptical\", \"add a joke in the opening\")", key="modification_instruction")
    if st.button("Regenerate script with this modification", type="primary"):
        instruction = st.session_state.modification_instruction.strip()
        if not instruction:
            st.info("Enter a modification instruction first.")
        else:
            llm, err = get_llm_or_none(provider, api_key, model, base_url)
            if err:
                st.error(f"LLM configuration problem: {err}")
            else:
                host = st.session_state.confirmed_host
                guest = st.session_state.confirmed_guest
                progress_bar = st.progress(0.0, text="Regenerating script...")

                def on_progress(i, n, section):
                    label = section.topic or section.kind
                    progress_bar.progress(i / n, text=f"Regenerated {i}/{n}: {label}")

                try:
                    new_result = modify_script(
                        result, instruction, host, guest, st.session_state.doc_texts, llm, progress_callback=on_progress
                    )
                except LLMError as exc:
                    st.error(f"Regeneration failed: {exc}")
                else:
                    st.session_state.script_history.append(new_result)
                    st.session_state.review_version_index = len(st.session_state.script_history) - 1
                    st.rerun()

    st.divider()
    if st.button("⚠️ Restart flow"):
        confirm_restart_dialog()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    init_session_state()
    apply_pending_restart()
    provider, api_key, model, base_url = render_llm_sidebar()

    st.title("🎙️ Podcast Script Generator")
    render_stepper(st.session_state.stage)
    st.divider()

    if st.session_state.stage == "setup":
        render_setup_stage(provider, api_key, model, base_url)
    elif st.session_state.stage == "topics":
        render_topics_stage(provider, api_key, model, base_url)
    elif st.session_state.stage == "review":
        render_review_stage(provider, api_key, model, base_url)


if __name__ == "__main__":
    main()
