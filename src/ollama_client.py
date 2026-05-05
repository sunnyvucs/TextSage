"""
ollama_client.py
Thin wrapper around the Ollama Python client.
Ollama manages its own context size and GPU settings — no options are forced.
"""

import json
import logging
import re
import time
import ollama
from src.config import OLLAMA_MODEL

log = logging.getLogger(__name__)


def chat(prompt: str, retries: int = 3, delay: int = 5) -> str:
    """Send a prompt to Ollama and return the response text."""
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Ollama call failed after {retries} attempts: {exc}") from exc
            log.warning("Ollama attempt %d/%d failed: %s — retrying in %ds", attempt, retries, exc, delay)
            time.sleep(delay)


def chat_multiturn(messages: list[dict], retries: int = 3, delay: int = 5) -> str:
    """Send a multi-turn conversation to Ollama and return the final response text."""
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Ollama multi-turn call failed after {retries} attempts: {exc}") from exc
            log.warning("Ollama attempt %d/%d failed: %s — retrying in %ds", attempt, retries, exc, delay)
            time.sleep(delay)


def _strip_fences(raw: str) -> str:
    """Remove markdown ```json ... ``` or ``` ... ``` code fences if present."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[1:end])
    return raw.strip()


def _salvage_truncated_json(raw: str) -> dict | list | None:
    """
    If the model truncated mid-JSON, try to recover whatever chapters were
    fully formed before the truncation by extracting complete chapter objects.
    """
    matches = re.findall(
        r'\{\s*"chapter_number"\s*:.*?"topics"\s*:\s*\[.*?\]\s*\}',
        raw,
        re.DOTALL,
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


def chat_json(prompt: str, retries: int = 3, delay: int = 5) -> dict | list:
    """Same as chat() but parses and returns the response as JSON."""
    raw = chat(prompt, retries=retries, delay=delay)
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_truncated_json(raw)
    if salvaged:
        return salvaged
    raise ValueError(f"Model response was not valid JSON and could not be salvaged:\n{raw[:500]}")


def chat_json_multiturn(messages: list[dict], retries: int = 3, delay: int = 5) -> dict | list:
    """Same as chat_multiturn() but parses and returns the response as JSON."""
    raw = chat_multiturn(messages, retries=retries, delay=delay)
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_truncated_json(raw)
    if salvaged:
        return salvaged
    raise ValueError(f"Model response was not valid JSON and could not be salvaged:\n{raw[:500]}")
