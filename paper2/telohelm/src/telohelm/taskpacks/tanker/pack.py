"""The tanker join task as one pack (study2/packs/__init__.py, TaskPack): every task-specific part the core uses.

Assembled from the slices: the public contract (tanker.py), the goal's parts, the decisions' roles, the help rule, the
action value rules and the briefing check's evidence view (tanker_execution.py), and the context builder
(tanker_context.py). This module imports nothing from the world, the monitor, the reference pilot or the prior.
"""
from __future__ import annotations

from dataclasses import replace

from ...contracts import TaskPack
from .public import TANKER
from .examples import TANKER_EXAMPLES
from .context import ContextBuilder
from .execution import TANKER_EVIDENCE; from .execution import TANKER_GOAL; from .execution import TANKER_HELP; from .execution import TANKER_ROLES; from .execution import tanker_part_error; from .execution import tanker_model_precondition

TANKER_PACK = TaskPack(public=TANKER, goal=TANKER_GOAL, roles=TANKER_ROLES, help=TANKER_HELP,
                       part_error=tanker_part_error, context_builder=ContextBuilder, evidence=TANKER_EVIDENCE,
                       example_policy=TANKER_EXAMPLES, model_precondition=tanker_model_precondition)

# What the context builder calculates from the observations rather than receives: the wind (from heading, track and
# speeds), the range to the goal tanker's newest report and its rate of change, and that report's age. The
# reduced-feature option hides them from Telohelm's procedures, checks and models, and from the field list the learner
# is told, so its own code must derive what it needs; the runtime's own rules still read them.
TANKER_DERIVED = frozenset({"wind_x_nm_s", "wind_y_nm_s", "report_age_s", "observed_range_nm",
                            "observed_range_rate_nm_s"})
TANKER_REDUCED_PACK = replace(TANKER_PACK,
                              public=replace(TANKER, context_fields=TANKER.context_fields - TANKER_DERIVED),
                              hidden=TANKER_DERIVED)
