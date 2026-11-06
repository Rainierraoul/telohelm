"""Decision D42 (Simon, 21 October 2026): what Telohelm is told is frozen.

The learner's rules, the in-flight models' prompt, the context and output fields, the runtime rules (context building
and the method ladder's escalation and help) and the trusted guards are fixed as of 5bfdc29. This test fails if any of
them changes. To change one: add an entry to paper2/notes/DESIGNER-KNOWLEDGE-LEDGER.md saying what changed, why, and
whether it followed a failure, and update the fingerprint here in the same commit. Until the protocol freeze the
other agent reviews it afterwards (record, don't block); from the freeze, no change without a protocol amendment.
"""
import hashlib
import json
from pathlib import Path

import telohelm.examples as X
import telohelm.learning as G
import telohelm.local_model as L
import paper2.experiment2.run_stream as RS
import telohelm.runtime.procedures as P

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())

TEXTS = {
    "learning.PROPOSER_RULES": "7446fd91dfa0fde2b491c91e4f282c96c66d539172fc0fd12ef13912c01dcd27",
    "run_stream.POC_RULES": "5aecc665d4016114364fa34ba40a8476ec0ed239aef81cadbe60bfe6936a8bd9",
    "run_stream.POC_RULES_BOTH": "1d5ebfe65803135a40db6f06b1c3f1d85b83abb85808b8a280525c693c3b56ed",
    "run_stream.POC_RULES_C": "709a95b8332a903919f0c7d846a78fe41695a688db4260b7071e5db0be7348d9",
    "examples.RULES": "da0ba465061000cc99cc5370e19d234b893e95058f0ba87814d81f821ab8a804",
    "local_model.SYSTEM": "11b1cb06dfb2702024fe8160333723ae0364cf4ceea44fe6a104f259fa04ba1b",
    "procedures.CONTEXT_FIELDS": "635973fddc125fb8ff1f30d627bc83277bc159e8236bd705499b2b1db12d0d23",
    "procedures.OUTPUT_FIELDS": "30516e436387cfe5aee6614c922ff1eac93b1e65a0091fbaf3ccded34df89730",
}
FILES = {
    'paper2/telohelm/src/telohelm/learning.py': "859f7e14045086b562154952e68fbb78b2091e6f6ee339c7da2d88fe1a44c373",
    'paper2/telohelm/src/telohelm/runtime/model_actions.py': "6ebe75bd6268a0978f0094ec01af0fd43dcc700afedf01985b480a8d9448e3e3",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/pack.py': "d1e18dc03143a24ba325c0f2241489c7139c4b5b825c1444d59d69ceae203659",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/execution.py': "379f8a5a93dbb835b51725479e9480baf89628c4e8dc04a128c5d5aa13de1eb2",
    # 26 October (TURN242, M1 timing rehearsal): opt-in host options only (layout, learning off with flight reasoning,
    # library copy and snapshot, timing spans); POC_RULES texts and what any condition is told are unchanged.
    # TURN244: spans gain start rows and --record-inputs writes scenario fingerprints; still host-side only.
    # TURN260: an opt-in pause_authority hook lets the study runner judge the pause across a condition's streams.
    # 27 October: an opt-in small_call_limit, the per-episode small-model allowance for C-full (amendment 7).
    'paper2/experiment2/run_stream.py': "a750beec33be5b52e73bbc42b40c52af04ca0de9b24956e71f067a2cbc001d84",
    'paper2/telohelm/src/telohelm/local_model.py': "89c00441c7a3114db57180f277c2eed95361c1877c89a0acbed7ca1566a33063",
    'paper2/telohelm/src/telohelm/runtime/delivery.py': "390d1a1e3c4c252a1317dc62c43d971b50c08f4dbb0e1ed2bec483bf49d51507",
    'paper2/telohelm/src/telohelm/contracts/__init__.py': "d2d577112e21cc25021d6bcbbdb392f912778866c95aaf0118c4f45ee4e26430",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/world/tanker_environment.py': "829258d188437c34b65cdadca667e6db1f9abf76b0b08e13ce8ebfcbf3ee200f",
    'paper2/telohelm/src/telohelm/runtime/environment.py': "5c23d493f38535fea4682d4b525fa6b74ad5bda9161d7b7c11d8809e2ba67b38",
    # Selected task storage preserves the implicit legacy tanker contract.
    'paper2/telohelm/src/telohelm/runtime/library.py': "d191ac8178a714c1a3d00c4d8dea9f573c4caadc9c912327ea93ea9aa3784c5e",
    # Candidate schema selection retains the legacy tanker rules.
    'paper2/telohelm/src/telohelm/runtime/procedures.py': "07aa352d0d1550e17035ab9aef744c66fe8f038c1f76454268dfaa65d70dfdc9",
    'paper2/telohelm/src/telohelm/contracts/example_view.py': "64788aff3f30d8488d3bff054db1040a0307556d366f8172b6c3f40d01365dbe",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/examples.py': "36e585c603842560cbb3b2841e780ac0208a124a8b4dfce2e38e800ef9c57cde",
    # Selected-builder reconstruction keeps the default tanker examples unchanged.
    'paper2/telohelm/src/telohelm/examples.py': "d33d0980a8ec5725c4cf6af072cd04911a79e43e07d665b96f1e2ce24f38dcd6",
    # 24 October: task-defined once-per-goal completion; same tanker behavior.
    'paper2/telohelm/src/telohelm/contracts/decisions.py': "de8d179e5a0f89f9a303a5c3a247b8047a9d5d839a621c06c68240dbde6b3875",
    'paper2/telohelm/src/telohelm/runtime/episode_loop.py': "8120321341b1b0933ac89684241b7f5770e6d466b40f8729c72dbe69625c47c5",
    # Trusted tool descriptors and the unchanged tanker implementation, moved on 23 October 2026.
    'paper2/telohelm/src/telohelm/contracts/tools.py': "0cc19728f137fdd89ffd37e20ae0abf0acba6f54c11389396d1a56ece5174b17",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/tools.py': "73fc8495436194e1e2e08e353d328ecb7306d451901a37f96edd3650b2bae464",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/prediction.py': "2b59e0464058acf8891d4342f18b56d4000d279e6b92215b5f6eda5a2df63551",
    # Moved from study2/runtime/context.py (68a5d6cf...) on 23 October 2026 with its code unchanged; that file now
    # only re-exports it.
    'paper2/telohelm/src/telohelm/taskpacks/tanker/context.py': "ffea2dc972490f8ef7d79bc087c80014a80e50f3f1f2dad68db6b65ea68966ff",
    # 23 October 2026: its stale-evidence rule and field value rules moved, unchanged, to packs/tanker_execution.py
    # (TANKER_HELP, tanker_part_error; fingerprinted in test_the_moved_help_and_action_rules_are_unchanged).
    'paper2/telohelm/src/telohelm/runtime/escalation.py': "caac0060b329d2eec6aa7e64431c44f32d8df265fc009c90922c73c40471587b",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-heading-guard.py': "3bde2cf1bb491e9ba0572693348e1c8246863a4cb52fc8215087c9a75d64eaed",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-report-guard.py': "b0c59c8fe08cc507d53030a743b1a52384ac93698fb24229d9a309917817f80a",
}
MESSAGE = ("{} changed. Decision D42 freezes what Telohelm is told: add a ledger entry "
           "(paper2/notes/DESIGNER-KNOWLEDGE-LEDGER.md) and update this fingerprint in the same commit.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_what_the_learner_and_the_flight_models_are_told_is_unchanged():
    current = {"learning.PROPOSER_RULES": G.PROPOSER_RULES, "run_stream.POC_RULES": RS.POC_RULES,
               "run_stream.POC_RULES_BOTH": RS.POC_RULES_BOTH, "run_stream.POC_RULES_C": RS.POC_RULES_C,
               "examples.RULES": X.RULES, "local_model.SYSTEM": L.SYSTEM,
               "procedures.CONTEXT_FIELDS": json.dumps(sorted(P.CONTEXT_FIELDS)),
               "procedures.OUTPUT_FIELDS": json.dumps(sorted(P.OUTPUT_FIELDS))}
    for name, expected in TEXTS.items():
        assert _sha(current[name]) == expected, MESSAGE.format(name)


def test_the_runtime_rules_and_trusted_guards_are_unchanged():
    for name, expected in FILES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, MESSAGE.format(name)


# The opt-in disagreement watch and own reports (23 October 2026; Codex's TURN231 item 4). With the options off none of
# this reaches Telohelm; with them on, the meaning sentence and the gap go to the flight models, the learner's feedback
# carries the gap, and these files decide when a procedure is withheld. A change to any of them needs a ledger entry.
WATCH_TEXTS = {
    "tanker_execution.TANKER_EVIDENCE.meaning": "07022abf0e6dfbe9438cfceb9b11e73421dea92125edf71244790e72ce6a52ff",
    "tanker_execution.TANKER_EVIDENCE.policy": "5f4bff921fc337744e1eace23601859cc8dd92d34c53cae93e8753656e8c93a8",
    "tanker_execution.briefing_gap": "e4d2fa9611d12e945fd955c78130b4b6bef918329611b16e6adcc2d457c149c3",
    "learning.briefing_disagreement": "b5b3a61d31f995c8b5e6fc0e39cec2a8dca9aa72a2d4c8f72cc2493f0072b5b1",
}
WATCH_FILES = {
    'paper2/telohelm/src/telohelm/contracts/__init__.py': "d2d577112e21cc25021d6bcbbdb392f912778866c95aaf0118c4f45ee4e26430",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/world/tanker_environment.py': "829258d188437c34b65cdadca667e6db1f9abf76b0b08e13ce8ebfcbf3ee200f",
    'paper2/telohelm/src/telohelm/runtime/environment.py': "5c23d493f38535fea4682d4b525fa6b74ad5bda9161d7b7c11d8809e2ba67b38",
    # Selected task storage preserves the implicit legacy tanker contract.
    'paper2/telohelm/src/telohelm/runtime/library.py': "d191ac8178a714c1a3d00c4d8dea9f573c4caadc9c912327ea93ea9aa3784c5e",
    # Candidate schema selection retains the legacy tanker rules.
    'paper2/telohelm/src/telohelm/runtime/procedures.py': "07aa352d0d1550e17035ab9aef744c66fe8f038c1f76454268dfaa65d70dfdc9",
    'paper2/telohelm/src/telohelm/contracts/example_view.py': "64788aff3f30d8488d3bff054db1040a0307556d366f8172b6c3f40d01365dbe",
    'paper2/telohelm/src/telohelm/taskpacks/tanker/examples.py': "36e585c603842560cbb3b2841e780ac0208a124a8b4dfce2e38e800ef9c57cde",
    # Selected-builder reconstruction keeps the default tanker examples unchanged.
    'paper2/telohelm/src/telohelm/examples.py': "d33d0980a8ec5725c4cf6af072cd04911a79e43e07d665b96f1e2ce24f38dcd6",
    # 24 October: task-defined once-per-goal completion; same tanker behavior.
    'paper2/telohelm/src/telohelm/contracts/decisions.py': "de8d179e5a0f89f9a303a5c3a247b8047a9d5d839a621c06c68240dbde6b3875",
    'paper2/telohelm/src/telohelm/runtime/episode_loop.py': "8120321341b1b0933ac89684241b7f5770e6d466b40f8729c72dbe69625c47c5",
    'paper2/telohelm/src/telohelm/runtime/disagreement.py': "e88a181063fa625ac7333656294272c32c5024e9483df05bcdd9032354222813",
    'paper2/telohelm/src/telohelm/runtime/own_reports.py': "7be2e0cccdfc3a45fbe89cd8c99dff50cf27d80be017c94dfac202bb3d5eb58d",
    'paper2/telohelm/src/telohelm/contracts/evidence.py': "5340828aea94582afd99ce2f143917025b47856861ed5b1579a4aec2e04385a3",
}


def test_what_the_opt_in_watch_tells_and_decides_is_unchanged():
    import inspect
    import telohelm.taskpacks.tanker.execution as TX
    v = TX.TANKER_EVIDENCE
    current = {
        "tanker_execution.TANKER_EVIDENCE.meaning": v.meaning,
        "tanker_execution.TANKER_EVIDENCE.policy": json.dumps(
            {"tolerance": v.tolerance, "unit": v.unit, "affects": sorted(v.affects),
             "withdraw": {k: dict(w) for k, w in v.withdraw.items()}}, sort_keys=True),
        "tanker_execution.briefing_gap": inspect.getsource(TX.briefing_gap) + inspect.getsource(TX.briefed_position),
        "learning.briefing_disagreement": inspect.getsource(G.briefing_disagreement),
    }
    for name, expected in WATCH_TEXTS.items():
        assert _sha(current[name]) == expected, MESSAGE.format(name)
    for name, expected in WATCH_FILES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, MESSAGE.format(name)


def test_the_moved_help_and_action_rules_are_unchanged():
    import inspect
    import telohelm.taskpacks.tanker.execution as TX
    h = TX.TANKER_HELP
    text = json.dumps({"age_field": h.age_field, "stale_after_s": h.stale_after_s, "text": h.text}, sort_keys=True)
    assert _sha(text + inspect.getsource(TX.tanker_part_error)) == "4d9f5a662a4ef52e367cddf9287556fbdf284e19746a2e837c901ac6e22e8a02", MESSAGE.format("tanker_execution.TANKER_HELP")


def test_the_reduced_feature_option_shows_what_it_showed():
    # Opt-in (--reduced-features, 23 October 2026): the field list the learner is told and the fields hidden from
    # procedures, checks and models. Recorded in the ledger.
    from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK as R
    shown = json.dumps({"fields": sorted(R.public.context_fields), "hidden": sorted(R.hidden)})
    assert _sha(shown) == "8224d949767053f39b87dca5db7e5c3ccfb22f63b57ae6ceb0e0b8d8ef55e817", MESSAGE.format("tanker_pack.TANKER_REDUCED_PACK")


def test_the_tankers_once_per_goal_completion_fields_are_unchanged():
    from telohelm.taskpacks.tanker.execution import TANKER_ROLES
    body = json.dumps({k: sorted(v) for k, v in TANKER_ROLES.completion_fields.items()}, sort_keys=True)
    assert _sha(body) == "ed76cbf06d74fff316b566901e373d5a7964999df7e556db4a3b9c2f8585ab28", MESSAGE.format("TANKER_ROLES.completion_fields")
