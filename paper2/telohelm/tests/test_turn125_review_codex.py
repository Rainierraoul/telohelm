"""Independent exact enumeration of D22's sampling rationale. No scenarios or outcome data."""
from collections import Counter
from itertools import combinations, product
from math import comb


def test_uniform_five_of_twenty_can_omit_a_phase3_block():
    covered = sum(len({i // 5 for i in subset}) == 4 for subset in combinations(range(20), 5))
    assert comb(20, 5) == 15504
    assert covered == 5000 and comb(20, 5) - covered == 10504


def test_four_block_draws_and_one_from_remainder_are_uniform_over_covered_subsets():
    ordered_outcomes = Counter()
    for first in product(*(range(5*b, 5*b+5) for b in range(4))):
        remaining = set(range(20)) - set(first)
        assert len(remaining) == 16
        for extra in remaining:
            selected = tuple(sorted((*first, extra)))
            assert len(set(selected)) == 5 and len({i // 5 for i in selected}) == 4
            ordered_outcomes[selected] += 1
    assert sum(ordered_outcomes.values()) == 5**4 * 16 == 10000
    assert len(ordered_outcomes) == 5000 and set(ordered_outcomes.values()) == {2}
    # Every episode index has the same inclusion probability 2500/10000 = 1/4.
    marginals = Counter()
    for subset, multiplicity in ordered_outcomes.items():
        for index in subset:
            marginals[index] += multiplicity
    assert set(marginals.values()) == {2500}
