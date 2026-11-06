"""Explicit development-server setup for the demo; no automatic model discovery or fallback.

The bundled tokenizer supports one pinned model family. A matching server count
checks this request's formatting, not its weights, hardware or declared capacity.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from . import local_model, token_count
from .learning import GatewayProposer
from .ladder import ReasoningAllowance
from .runtime.transport import LoopbackTransport

MODEL = 'Qwen/Qwen3.8-27B'


@dataclass(frozen=True)
class ServerConfig:
    url: str
    context_window: int
    model: str = MODEL
    max_output_tokens: int = 131072
    input_cap: int = 16384
    timeout_s: float = 14400.0
    effort: str = 'xhigh'

    def __post_init__(self):
        LoopbackTransport(self.url, 1 << 20)  # validates numeric loopback origin without a socket
        if self.model != MODEL:
            raise ValueError('This example supports only the bundled Qwen/Qwen3.8-27B tokenizer/model family')
        for key, ceiling in (('context_window', 262144), ('max_output_tokens', 131072), ('input_cap', 16384)):
            value = getattr(self, key)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f'{key} must be an integer in 1..{ceiling}')
        if self.max_output_tokens >= self.context_window:
            raise ValueError('Declared context must leave room beyond the output allowance')
        if (isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 14400):
            raise ValueError('timeout_s must be finite, positive and at most 14400')
        if self.effort not in local_model.EFFORTS:
            raise ValueError('Unsupported reasoning effort')

    def record(self):
        return {**asdict(self), 'local_tokenizer_revision': token_count.REVISION,
                'local_tokenizer_sha256': token_count.TOKENIZER_SHA256,
                'local_template_sha256': hashlib.sha256(
                    (token_count.MODEL_FILES / 'chat_template.jinja').read_bytes()).hexdigest(),
                'context_capacity_source': 'caller-declared; native serving validation remains required',
                'remote_weights_verified': False, 'reportable': False}


def proposer(work, config: ServerConfig):
    """Count and probe each exact request before the ordinary gateway may send chat."""
    work = Path(work)
    transport = LoopbackTransport(config.url, 1 << 20)
    kwargs = {**local_model.SETTINGS['reasoning_model']['chat_template_kwargs'],
              'reasoning_effort': config.effort}

    def count(messages):
        body = {'model': config.model, 'messages': [{'role': r, 'content': c} for r, c in messages],
                'add_generation_prompt': True, 'chat_template_kwargs': kwargs}
        row = {'request': body, 'local_count': None, 'server_count': None, 'chat_allowed': False,
               'context_window_declared': config.context_window, 'max_output_tokens': config.max_output_tokens}
        try:
            local = token_count.count(messages, kwargs)
            row['local_count'] = local
            if local > config.input_cap or local + config.max_output_tokens > config.context_window:
                raise ValueError('Request exceeds the input allowance or declared context capacity')
            status, reply = transport(transport.origin + '/tokenize', body, min(config.timeout_s, 60))
            row.update(http_status=status, reply=reply)
            server = reply.get('count') if isinstance(reply, dict) else None
            row['server_count'] = server
            if status != 200 or type(server) is not int or server != local:
                raise ValueError('Server token count is absent or differs from the pinned local count')
            row['chat_allowed'] = True
            return local
        except Exception as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            with (work / 'token-count-gates.jsonl').open('a') as f:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

    gateway = GatewayProposer(config.url, config.model, work, 'cartpole-demo', condition='server-development',
                              max_output_tokens=config.max_output_tokens, timeout_s=config.timeout_s,
                              effort=config.effort, input_cap=config.input_cap, count_tokens=count,
                              allowance=ReasoningAllowance(work / 'reasoning-allowance.jsonl', limit=1))
    gateway.fit_tokens = lambda messages: token_count.count(messages, kwargs)
    return gateway
