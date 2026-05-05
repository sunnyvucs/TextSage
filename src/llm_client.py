"""
llm_client.py
Unified LLM client — routes to Groq or Ollama based on LLM_BACKEND config.
When LLM_BACKEND=groq, Groq is tried first; on rate-limit errors (429/413)
it automatically falls back to Ollama so the pipeline never stalls.
All AI calls in this project go through chat() / chat_json() / chat_json_multiturn().
"""

import json
import logging
import re
import time

from src.config import LLM_BACKEND, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL

log = logging.getLogger(__name__)

_RATE_LIMIT_CODES = (429, 413)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return any(str(code) in msg for code in _RATE_LIMIT_CODES)


# ── Groq backend ──────────────────────────────────────────────────────────────

def _groq_chat(messages: list[dict], retries: int = 3, delay: int = 3) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
            )
            elapsed = time.perf_counter() - t0
            log.debug("  [Groq] Response in %.2fs (attempt %d)", elapsed, attempt)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("  [Groq] Rate limit hit — falling back to Ollama.")
                raise  # let caller handle fallback immediately, no point retrying
            log.warning("  [Groq] Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise RuntimeError(f"Groq call failed after {retries} attempts: {exc}") from exc
            time.sleep(delay)


# ── Ollama backend ────────────────────────────────────────────────────────────

def _ollama_chat(messages: list[dict], retries: int = 3, delay: int = 5) -> str:
    import ollama
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"num_predict": -1},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            log.warning("  [Ollama] Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise RuntimeError(f"Ollama call failed after {retries} attempts: {exc}") from exc
            time.sleep(delay)


# ── Public interface ──────────────────────────────────────────────────────────

def chat(prompt: str, retries: int = 3) -> str:
    """Send a single-turn prompt and return response text.
    Groq is tried first when LLM_BACKEND=groq; falls back to Ollama on rate limits.
    """
    messages = [{"role": "user", "content": prompt}]
    if LLM_BACKEND == "groq":
        try:
            return _groq_chat(messages, retries=retries)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("  [LLM] Groq rate limit — falling back to Ollama/Gemma.")
                return _ollama_chat(messages, retries=retries)
            raise
    return _ollama_chat(messages, retries=retries)


def chat_multiturn(messages: list[dict], retries: int = 3) -> str:
    """Send a multi-turn conversation and return final response text.
    Groq is tried first when LLM_BACKEND=groq; falls back to Ollama on rate limits.
    """
    if LLM_BACKEND == "groq":
        try:
            return _groq_chat(messages, retries=retries)
        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning("  [LLM] Groq rate limit — falling back to Ollama/Gemma.")
                return _ollama_chat(messages, retries=retries)
            raise
    return _ollama_chat(messages, retries=retries)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        lines = raw.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[1:end])
    return raw.strip()


def _salvage_truncated_json(raw: str) -> dict | None:
    matches = re.findall(
        r'\{\s*"chapter_number"\s*:.*?"topics"\s*:\s*\[.*?\]\s*\}',
        raw, re.DOTALL,
    )
    if not matches:
        return None
    chapters = []
    for m in matches:
        try:
            chapters.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return {"chapters": chapters, "_truncated": True} if chapters else None


def chat_json(prompt: str, retries: int = 3) -> dict | list:
    """Single-turn prompt, returns parsed JSON."""
    raw = _strip_fences(chat(prompt, retries=retries))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_truncated_json(raw)
    if salvaged:
        return salvaged
    raise ValueError(f"Response not valid JSON:\n{raw[:500]}")


def chat_json_multiturn(messages: list[dict], retries: int = 3) -> dict | list:
    """Multi-turn conversation, returns parsed JSON."""
    raw = _strip_fences(chat_multiturn(messages, retries=retries))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_truncated_json(raw)
    if salvaged:
        return salvaged
    raise ValueError(f"Response not valid JSON:\n{raw[:500]}")
