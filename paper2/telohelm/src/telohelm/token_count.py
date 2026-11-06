"""Counting a reasoning-model request's input tokens before it is sent (Experiment 2 protocol, section 5: at most
4,096 input tokens per request; Codex's MVP119.INPUT).

The count is the pinned model's own: Qwen3.8-27B's chat template, at the revision Study 1 pinned, rendered as the
server renders it (a sandboxed Jinja environment with trim_blocks and lstrip_blocks, the generation prompt added),
then its tokenizer, with no extra special tokens. Both files are copied into model_files/ from the cluster's model
cache, and the tokenizer's SHA-256 is checked against its Hugging Face blob name before use. The count therefore
includes the template's own markup and, on a repair, every earlier message. It works offline and never contacts a
server. That it equals the server's count is checked against vLLM's /tokenize on the cluster, not assumed here.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

INPUT_TOKEN_CAP = 4_096
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_FILES = Path(__file__).resolve().parent / "model_files" / f"Qwen3.8-27B-{REVISION}"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any]:
    import jinja2
    import jinja2.ext
    import jinja2.sandbox
    import tokenizers
    raw = (MODEL_FILES / "tokenizer.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer.json is not the pinned file")
    tokenizer = tokenizers.Tokenizer.from_str(raw.decode("utf-8"))

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string((MODEL_FILES / "chat_template.jinja").read_text(encoding="utf-8"))
    return tokenizer, template


def render(messages: list, chat_template_kwargs: dict | None = None) -> str:
    _, template = _load()
    return template.render(messages=[{"role": role, "content": text} for role, text in messages],
                           add_generation_prompt=True, **(chat_template_kwargs or {}))


def count(messages: list, chat_template_kwargs: dict | None = None) -> int:
    tokenizer, _ = _load()
    return len(tokenizer.encode(render(messages, chat_template_kwargs), add_special_tokens=False).ids)
