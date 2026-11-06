"""Fixed controls for exact HELP167.ATOMIC repair; no generated episode."""
from telohelm.runtime.escalation import FlightLadder
from paper2.experiment2.tests.test_flight_ladder import _Gateway


def test_ladder_proposal_then_public_refusal_then_acceptance_uses_one_request():
    ladder=FlightLadder(_Gateway());ladder.at(1)
    ctx={"t":100,"report_age_s":12,"deadline":None}
    proposed=ladder.help(ctx,{},False)
    assert proposed and not ladder.help_asked
    ladder.help_outcome(100,proposed,"ladder",None,False)
    assert not ladder.help_asked and ladder.help(ctx,{},False)==proposed
    assert ladder.events[-1]["kind"]=="help_refused_with_action"
    ladder.help_outcome(101,proposed,"ladder",None,True)
    assert ladder.help_asked and ladder.help(ctx,{},False) is None
    assert [e["kind"] for e in ladder.events].count("help_asked")==1
