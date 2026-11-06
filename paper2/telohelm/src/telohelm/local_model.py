"""A real local model behind the gateway for development runs (MVP step 2, paper2/plan/MVP-PLAN.md).

The model is served on this machine by vLLM, as in Study 1 (for example mlx-community/Qwen3-4B-Instruct-2507-8bit on
http://127.0.0.1:8000). The provider is LocalChatProvider with the loopback transport and no exchange store, so it is
unreportable: these runs are development integration, not evidence. The prompt and parser here are a development
draft; the frozen prompts belong to backlog item B7.
"""
from __future__ import annotations

import json
from typing import Any

from .runtime import slots as slots
from .runtime.models import LocalChatProvider
from .runtime.transport import LoopbackTransport
from .taskpacks.tanker.public import TANKER

RECENT_REPORTS = 5

# The in-flight models' prompts come from the task's public contract (study2/packs/tanker.py).
SYSTEM = TANKER.flight_system
ASK = dict(TANKER.flight_ask)


def compact(ctx: dict) -> dict:
    """The context without its full radar history: scalar fields and the newest few reports."""
    out = {k: v for k, v in ctx.items() if k != "reports"}
    reports = ctx.get("reports")
    if isinstance(reports, list):
        out["recent_reports"] = reports[-RECENT_REPORTS:]
    return out


def messages_for(decision: str, ctx: dict, reason: str) -> list:
    return [("system", SYSTEM),
            ("user", ASK.get(decision, "Reply {}.") + "\nContext:\n" + json.dumps(compact(ctx), default=str,
                                                                                  sort_keys=True))]


REPORT_CONTROL_INSTRUCTION = (
    "Choose an action for the report decision only. You may make a report now, set or clear the existing report "
    "trigger, or return {} to wait. Do not make a report merely because a reply is requested. A trigger acts on "
    "observed range, which need not be true range when its measurement is old. reply_delay_s is the modelled delay "
    "before this reply can act; the observations can change before delivery. last_action_refusal, when present, "
    "records an earlier proposal and the public evidence that contradicted it. Reply with one JSON action object.")


def report_control_messages(decision: str, ctx: dict, reason: str, *, public=TANKER) -> list:
    """Opt-in: show the selected public goal/actions, including existing deferred actions; no protected inputs."""
    return [("system", public.flight_system + "\n" + public.learner_goal + public.learner_actions
             + REPORT_CONTROL_INSTRUCTION),
            ("user", json.dumps({"decision": decision, "reason": reason, "context": compact(ctx)},
                                default=str, sort_keys=True))]


def first_object(text: str) -> dict:
    """The first complete JSON object in the text, ignoring anything before or after it (Codex's MVP116.PARSE).
    Reasoning a model writes before a closing </think> is skipped first. Raises ValueError if there is none."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object in the reply")


def parse(decision: str, text: str) -> dict | None:
    """The first JSON object in the reply. A reply without one is unparseable (ValueError)."""
    return first_object(text) or None


def parse_with_wait(decision: str, text: str) -> dict:
    """Keep an explicit {} separate from no answer. Delivery decides which decisions permit waiting."""
    return first_object(text)


# Study 1's model settings (study1/config.cluster.json, model_settings), which the plan carries into Experiment 2
# (Appendix A, row 9: the reasoning model at medium effort). They are each model's published generation defaults.
# Until 13 October these runs sent temperature 0 and no effort, so the reasoning model thought at its default,
# 'xhigh'; Study 1 had found that 'xhigh' made its thinking run past 8,192 tokens.
SETTINGS = {
    "small_model": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "reasoning_model": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}},
}


# The reasoning efforts the pinned chat template accepts (model_files/Qwen3.8-27B-1d4bf0f2.../chat_template.jinja); any
# other value raises in the template, so every request would fail its token count (job 4051059 asked for "high").
EFFORTS = ("low", "medium", "xhigh")


def provider(base_url: str, model: str, timeout_s: float = 60.0, max_response_bytes: int = 1 << 20,
             thinking: bool = True, role: str = "small_model", effort: str | None = None) -> Any:
    """An unreportable local chat provider for both roles, on one loopback server, with Study 1's settings for the
    given role. thinking=False asks the server's chat template to skip the model's thinking (a development setting,
    recorded with each run)."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"reasoning effort {effort!r} is not one of {', '.join(EFFORTS)}")
    # A job that shares this server among several conditions limits the requests in flight (runtime/slots.py).
    transport = slots.wrap(LoopbackTransport(base_url, max_response_bytes), base_url)
    settings = json.loads(json.dumps(SETTINGS[role]))
    temperature = settings.pop("temperature")
    if not thinking:
        settings["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort is not None:
        # A development setting (Simon, 15 October): another reasoning effort than Study 1's medium, recorded with
        # each answer. The protocol's effort is medium.
        settings["chat_template_kwargs"] = {**settings.get("chat_template_kwargs", {}), "reasoning_effort": effort}
    return LocalChatProvider(base_url, {"small_model": model, "reasoning_model": model}, transport,
                             {"small_model": timeout_s, "reasoning_model": timeout_s}, temperature=temperature,
                             extra_body=settings)
