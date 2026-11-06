"""Trusted host connection to a simulator and its trial rules. Never part of a model request."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..taskpacks.tanker.public import TANKER
from ..contracts.tools import Tool


@dataclass(frozen=True)
class Environment:
    """The host supplies the task's simulator, trial judgement, step bound and permitted tools.

    A simulator implements reset(), step(action) and outcome(), plus t and record.actions. Each action record has
    actions and accepted. Bundles feed the selected context builder. Outcomes use the runtime's common goal,
    settlement and ending fields; a task's trial callback interprets its additional outcome fields.
    """
    task_id: str
    schema_version: int
    create: Callable[[Any], Any]
    trial: Callable[[str | None, str | None, dict, list], dict]
    max_steps: int
    tools: Mapping[str, Tool] | None = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("environment needs a task identity")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("environment schema version must be positive")
        if not callable(self.create) or not callable(self.trial):
            raise ValueError("environment simulator and trial rules must be callable")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("environment step bound must be a positive integer")
        if self.tools is not None:
            registry = dict(self.tools)
            if any(not isinstance(k, str) or not k or not isinstance(v, Tool) for k, v in registry.items()):
                raise ValueError("environment tools must be named trusted Tool entries")
            object.__setattr__(self, "tools", MappingProxyType(registry))

    def record(self) -> dict:
        """Inspection identity; reproducing callbacks also needs the pinned source and host configuration."""
        def name(fn):
            return (getattr(fn, "__module__", type(fn).__module__) + "."
                    + getattr(fn, "__qualname__", type(fn).__qualname__))
        return {"task_id": self.task_id, "schema_version": self.schema_version,
                "create": name(self.create), "trial": name(self.trial), "max_steps": self.max_steps,
                "tools": sorted(self.tools) if self.tools is not None else None}


def environment_for(pack) -> Environment:
    """Keep the historical tanker default; require other tasks to supply their own trusted connection."""
    environment = getattr(pack, "environment", None)
    legacy = (pack.public.pack_id, pack.public.schema_version) == (TANKER.pack_id, TANKER.schema_version)
    if environment is None:
        if not legacy:
            raise ValueError("the selected task must supply an environment")
        from ..taskpacks.tanker.world.tanker_environment import TANKER_ENVIRONMENT
        environment = TANKER_ENVIRONMENT
    if not isinstance(environment, Environment):
        raise ValueError("the task environment must be an Environment")
    if (environment.task_id, environment.schema_version) != (pack.public.pack_id, pack.public.schema_version):
        raise ValueError("environment must match the selected task identity and version")
    if environment.tools is None and not legacy:
        raise ValueError("another task must supply an explicit tool table")
    return environment
