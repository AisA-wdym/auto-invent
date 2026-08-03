"""The challenge factory: architecture.md 7.

The pack is upstream of every score, so a defect here cannot be separated from signal anywhere
downstream. These tests concentrate on the properties that make a pack usable at all: the exact
family split, the independence of domain and family assignment, and the fact that every filter
fails toward rejection.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import pytest

from protocol.canonical import digest_object
from protocol.fixedpoint import PPM
from validator.challenge_factory.dedup import (
    cosine_ppm,
    fingerprint,
    is_duplicate,
    jaccard_ppm,
)
from validator.challenge_factory.discriminator import ProbeOutcome, assess, instability_ppm
from validator.challenge_factory.generator import GeneratorConfig
from validator.challenge_factory.linter import Requirement, lint
from validator.challenge_factory.pipeline import pack_hash
from validator.challenge_factory.safety import screen
from validator.challenge_factory.store import (
    InMemoryStore,
    StoredPack,
    StoreError,
    assert_not_sandbox_reachable,
)
from validator.challenge_factory.taxonomy import (
    EXCLUDED_DOMAINS,
    Slot,
    Taxonomy,
    TaxonomyError,
    plan,
)

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
GENERATORS = SEASON["challenge_generation"]["generators"]
TAXONOMY = Taxonomy.from_season(SEASON)
CONFIG = GeneratorConfig.from_season(SEASON)
SEED = bytes(range(32))


def valid_candidate(**over) -> dict:
    body = {
        "title": "Bounding tail latency in a fan-out read path",
        "domain": "software_architecture",
        "problem_statement": (
            "A read request fans out to twelve shards and returns when all twelve reply, so the "
            "latency of the whole request is the latency of its slowest shard. At the 99th "
            "percentile this makes the request roughly as slow as the worst shard on its worst "
            "day, and adding shards makes it worse rather than better. Practitioners hedge by "
            "issuing duplicate requests, which doubles load precisely when the system is already "
            "struggling. What is needed is a mechanism that bounds the tail without a "
            "proportional increase in offered load."
        ),
        "research_objective": (
            "Produce mechanisms that bound 99th-percentile fan-out latency without more than a "
            "10% increase in total requests issued."
        ),
        "current_baseline": (
            "Hedged requests after a fixed delay, and tied requests with cancellation."
        ),
        "known_attempts": ["Hedging after the 95th percentile", "Tied requests"],
        "constraints": [
            "The mechanism must add at most 10% to total requests issued.",
            "It cannot assume shards report their own load honestly.",
        ],
        "forbidden_shortcuts": ["Simply reducing the shard count is not a solution."],
        "required_output": {
            "portfolio_size": 5,
            "ranked": True,
            "mechanism_required": True,
            "prior_art_comparison_required": True,
            "falsification_plan_required": True,
            "simulation_or_calculation_required": True,
        },
        "resource_limits": {
            "maximum_wall_time_seconds": 1_800,
            "maximum_rcc": 400,
            "maximum_search_calls": 100,
        },
    }
    body.update(over)
    return body


# --------------------------------------------------------------------------
# 7.2: the plan, and the independence that makes 7.2.1's signal mean anything
# --------------------------------------------------------------------------


def test_the_pack_has_exactly_ten_slots_per_family():
    """Exactly ten each, not ten on average.

    Drawing a family per slot independently would give ten and ten in expectation; a day that
    came out fourteen-six would silently break the balance the two-generator design rests on.
    """
    slots = plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS)
    counts = Counter(slot.generator_family for slot in slots)
    assert counts == {"gpt": 10, "claude": 10}


def test_the_split_is_exact_for_every_seed():
    """Checked across many seeds, because "exact" is the claim and one seed cannot show it."""
    for index in range(50):
        seed = digest_object({"n": index}).encode()[:32]
        counts = Counter(
            slot.generator_family for slot in plan(seed, taxonomy=TAXONOMY, generators=GENERATORS)
        )
        assert counts == {"gpt": 10, "claude": 10}, f"seed {index} dealt {counts}"


def test_the_domain_stratification_is_exactly_as_declared():
    counts = Counter(slot.domain for slot in plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS))
    assert counts == Counter(TAXONOMY.stratification)


def test_domain_and_family_assignment_are_independent():
    """The property that makes 7.2.1's overfit signal measure what it claims.

    If GPT always drew the algorithm slots, a laboratory good at algorithms would score better on
    GPT's half for reasons unrelated to generator style — and "scores far better on one family's
    problems has learned a generator" would be measuring domain skill instead. Measured over many
    seeds: each domain must go to each family sometimes.
    """
    pairs: set[tuple[str, str]] = set()
    for index in range(60):
        seed = digest_object({"i": index}).encode()[:32]
        for slot in plan(seed, taxonomy=TAXONOMY, generators=GENERATORS):
            pairs.add((slot.domain, slot.generator_family))

    for domain in TAXONOMY.stratification:
        families = {family for stored_domain, family in pairs if stored_domain == domain}
        assert families == {"gpt", "claude"}, (
            f"{domain} only ever went to {families}: domain and family are correlated, which "
            "makes the generator-overfit signal measure domain skill"
        )


def test_the_same_seed_plans_the_same_pack():
    assert plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS) == plan(
        SEED, taxonomy=TAXONOMY, generators=GENERATORS
    )


def test_a_different_seed_plans_a_different_pack():
    other = digest_object({"other": True}).encode()[:32]
    assert plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS) != plan(
        other, taxonomy=TAXONOMY, generators=GENERATORS
    )


def test_the_generator_order_in_config_does_not_change_the_plan():
    """Two validators with the same seed and the same generators listed differently must agree.

    The pack hash is committed, so they would be unable to explain a difference.
    """
    assert plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS) == plan(
        SEED, taxonomy=TAXONOMY, generators=list(reversed(GENERATORS))
    )


def test_every_slot_has_a_critic():
    """A slot with no reviewer would put an unreviewed problem into a committed pack."""
    for slot in plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS):
        assert slot.critic_family


def test_the_critic_family_comes_from_the_season_config():
    """Owner decision: the critique does not reach across model families, so each family reviews
    its own output. Asserted against the config rather than hard-coded, so changing the config
    changes the behaviour and this test follows it."""
    declared = {
        str(entry["family"]): str(entry["critic_family"]) for entry in GENERATORS
    }
    for slot in plan(SEED, taxonomy=TAXONOMY, generators=GENERATORS):
        assert slot.critic_family == declared[slot.generator_family]


def test_a_slot_may_critique_its_own_family():
    """Permitted since the cross-family requirement was dropped. Previously refused in the type."""
    slot = Slot(index=0, domain="algorithms", generator_family="gpt", critic_family="gpt")
    assert slot.critic_family == "gpt"


def test_a_slot_with_no_critic_cannot_be_constructed():
    with pytest.raises(TaxonomyError, match="no critic family"):
        Slot(index=0, domain="algorithms", generator_family="gpt", critic_family="")


# --------------------------------------------------------------------------
# Taxonomy validation, before a day depends on it
# --------------------------------------------------------------------------


def test_the_example_season_taxonomy_validates():
    TAXONOMY.validate()


def test_a_stratification_that_does_not_sum_to_the_pack_size_is_refused():
    """Scaling to fit would silently overrule the declared emphasis of 7.2."""
    with pytest.raises(TaxonomyError, match="silently overrule"):
        Taxonomy(
            domains=("algorithms", "software_architecture"),
            challenges_per_day=20,
            stratification={"algorithms": 5, "software_architecture": 5},
            excluded_domains=frozenset(),
        ).validate()


def test_a_taxonomy_containing_an_excluded_domain_is_refused():
    """2 excludes these from scoring; a season that listed one would enable it."""
    with pytest.raises(TaxonomyError, match="excluded from V1 scoring"):
        Taxonomy(
            domains=("algorithms", "weapons_malware_or_exploits"),
            challenges_per_day=2,
            stratification={"algorithms": 1, "weapons_malware_or_exploits": 1},
            excluded_domains=frozenset(),
        ).validate()


def test_a_domain_stratified_at_zero_is_refused():
    with pytest.raises(TaxonomyError, match="removed from the stratification"):
        Taxonomy(
            domains=("algorithms", "memory_and_retrieval"),
            challenges_per_day=1,
            stratification={"algorithms": 1, "memory_and_retrieval": 0},
            excluded_domains=frozenset(),
        ).validate()


def test_a_stratification_naming_an_unknown_domain_is_refused():
    with pytest.raises(TaxonomyError, match="not in the taxonomy"):
        Taxonomy(
            domains=("algorithms",),
            challenges_per_day=2,
            stratification={"algorithms": 1, "telepathy": 1},
            excluded_domains=frozenset(),
        ).validate()


def test_a_generator_slot_total_disagreeing_with_the_taxonomy_is_refused():
    """Two definitions of the pack size mean a slot is unassigned or generated twice."""
    with pytest.raises(TaxonomyError, match="has one definition"):
        plan(
            SEED,
            taxonomy=TAXONOMY,
            generators=[
                {"family": "gpt", "slots": 5, "critic_family": "claude"},
                {"family": "claude", "slots": 5, "critic_family": "gpt"},
            ],
        )


# --------------------------------------------------------------------------
# 7.4 step 2: the linter
# --------------------------------------------------------------------------


def test_a_well_formed_candidate_passes():
    result = lint(valid_candidate())
    assert result.accepted, result.failures


def test_the_linter_reports_every_failure_rather_than_the_first():
    """A generator told one problem at a time makes one fix per candidate."""
    result = lint({"title": "x"})
    assert len(result.failures) >= 5


def test_a_short_problem_statement_fails():
    result = lint(valid_candidate(problem_statement="Make it faster."))
    assert any(Requirement.PROBLEM in failure for failure in result.failures)


def test_an_objective_that_repeats_the_statement_fails():
    """The problem stated twice and the goal never — which reads as complete."""
    statement = valid_candidate()["problem_statement"]
    result = lint(valid_candidate(research_objective=statement))
    assert any("repeats problem_statement" in failure for failure in result.failures)


def test_a_candidate_with_no_checkable_constraint_fails():
    """"Should be efficient" cannot be scored for fit."""
    result = lint(
        valid_candidate(constraints=["Should be efficient", "Ought to be maintainable"])
    )
    assert any("no constraint is checkable" in failure for failure in result.failures)


def test_a_candidate_with_no_forbidden_shortcuts_fails():
    """Without them, restating the baseline is a valid submission."""
    result = lint(valid_candidate(forbidden_shortcuts=[]))
    assert any("forbidden_shortcuts" in failure for failure in result.failures)


def test_a_retrieval_shaped_problem_fails():
    result = lint(
        valid_candidate(title="What is the time complexity of quicksort in the average case?")
    )
    assert any(Requirement.INVENTION in failure for failure in result.failures)


def test_a_problem_requiring_a_physical_prototype_fails():
    result = lint(
        valid_candidate(
            constraints=[
                "The physical prototype must weigh under 2kg.",
                "It must not exceed 40 degrees.",
            ]
        )
    )
    assert any(Requirement.SCOPE in failure for failure in result.failures)


def test_a_problem_requiring_private_data_fails():
    result = lint(
        valid_candidate(
            research_objective=(
                "Use our internal dataset of request traces to derive a bound on tail latency."
            )
        )
    )
    assert any(Requirement.SCOPE in failure for failure in result.failures)


def test_an_unbounded_budget_fails():
    """An unbounded ceiling means two laboratories were not asked the same question."""
    result = lint(
        valid_candidate(
            resource_limits={
                "maximum_wall_time_seconds": 1_800,
                "maximum_rcc": 0,
                "maximum_search_calls": 100,
            }
        )
    )
    assert any(Requirement.BUDGET in failure for failure in result.failures)


def test_a_string_constraints_field_does_not_count_as_many_constraints():
    """`constraints: "must be fast"` would otherwise iterate character by character and report
    sixteen single-letter constraints, passing the count check."""
    result = lint(valid_candidate(constraints="must be fast within 10ms"))
    assert any(Requirement.CONSTRAINTS in failure for failure in result.failures)


def test_the_linter_never_raises_on_a_malformed_candidate():
    """A generator returning something that is not a challenge is the ordinary case; an exception
    would abort a day's generation over one bad completion."""
    for candidate in ({}, {"title": None}, {"constraints": 5}, {"required_output": "yes"}):
        assert lint(candidate).accepted is False


def test_the_linter_is_deterministic():
    candidate = valid_candidate()
    assert lint(candidate) == lint(candidate)


def test_a_candidate_in_an_excluded_domain_fails_the_linter():
    result = lint(
        valid_candidate(domain="legal_or_policy"), excluded_domains=EXCLUDED_DOMAINS
    )
    assert any("excluded from V1 scoring" in failure for failure in result.failures)


# --------------------------------------------------------------------------
# The safety filter
# --------------------------------------------------------------------------


def test_an_ordinary_candidate_is_safe():
    assert screen(valid_candidate()).safe


def test_exploit_alone_is_not_flagged():
    """"Exploit locality of reference" is ordinary in this domain, and a filter that flagged it
    would reject a large fraction of legitimate architecture problems."""
    assert screen(
        valid_candidate(
            research_objective="Exploit locality of reference to reduce cache pressure by 30%."
        )
    ).safe


def test_exploit_with_vulnerability_context_is_flagged():
    verdict = screen(
        valid_candidate(
            problem_statement=(
                "Design a mechanism to exploit a buffer overflow vulnerability and deliver a "
                "payload to the target process without detection. " * 3
            )
        )
    )
    assert not verdict.safe
    assert verdict.excluded_domain == "weapons_malware_or_exploits"


def test_defensive_malware_work_is_permitted():
    """The field's legitimate work is overwhelmingly defensive; flagging it would exclude it."""
    assert screen(
        valid_candidate(
            research_objective=(
                "Improve malware detection by classifying behavioural traces in a sandbox."
            )
        )
    ).safe


def test_offensive_malware_work_is_flagged():
    verdict = screen(
        valid_candidate(
            problem_statement="Design ransomware that spreads across a network. " * 8
        )
    )
    assert not verdict.safe


def test_a_clinical_candidate_is_flagged():
    verdict = screen(
        valid_candidate(
            problem_statement="Design a system to recommend chemotherapy dosing. " * 8
        )
    )
    assert verdict.excluded_domain == "clinical_or_medical_treatment"


def test_an_excluded_term_hidden_in_a_constraint_is_still_found():
    """A generator that put the excluded requirement in a constraint rather than the statement
    would otherwise pass — and the constraint is what the laboratory is held to."""
    verdict = screen(valid_candidate(constraints=["Must be validated in a clinical trial.", "x"]))
    assert not verdict.safe


def test_a_declared_excluded_domain_is_refused_without_keyword_matching():
    verdict = screen(valid_candidate(domain="legal_or_policy"), excluded_domains=EXCLUDED_DOMAINS)
    assert not verdict.safe
    assert verdict.excluded_domain == "legal_or_policy"


def test_the_season_cannot_shrink_the_built_in_exclusions():
    """Unioned rather than replaced: a season that could shrink them could re-enable the domains
    where a published wrong answer causes harm outside the subnet."""
    verdict = screen(
        valid_candidate(problem_statement="Design a targeting system for a warhead. " * 8),
        excluded_domains=frozenset(),
    )
    assert not verdict.safe


# --------------------------------------------------------------------------
# 7.4 step 4: dedup
# --------------------------------------------------------------------------


def test_a_candidate_is_a_duplicate_of_itself():
    candidate = valid_candidate()
    history = [("c1", fingerprint(candidate))]
    assert is_duplicate(candidate, history=history, threshold_ppm=850_000).is_duplicate


def test_an_unrelated_candidate_is_not_a_duplicate():
    other = valid_candidate(
        title="Choosing a replication factor under a fixed storage budget",
        problem_statement=(
            "Storage is fixed and durability requirements differ per dataset. Uniform replication "
            "wastes capacity on data nobody would miss and under-protects data that matters. "
            "Practitioners pick a single factor and live with both errors. " * 3
        ),
        research_objective="Allocate replication per dataset under one fixed total capacity.",
    )
    history = [("c1", fingerprint(valid_candidate()))]
    verdict = is_duplicate(other, history=history, threshold_ppm=850_000)
    assert not verdict.is_duplicate


def test_a_paraphrase_is_caught_by_the_fingerprint():
    """The case fingerprints exist for: the same problem in different words."""
    original = valid_candidate()
    paraphrase = valid_candidate(
        title="Bounding tail latency in a fan-out read path (revised)",
        problem_statement=original["problem_statement"] + " This restates the same situation.",
    )
    verdict = is_duplicate(
        paraphrase, history=[("c1", fingerprint(original))], threshold_ppm=850_000
    )
    assert verdict.is_duplicate
    assert verdict.detected_by == "fingerprint"


def test_cross_domain_candidates_are_never_duplicates():
    """Two problems in different domains are not duplicates however much vocabulary they share."""
    same_text = valid_candidate(domain="algorithms")
    history = [("c1", fingerprint(valid_candidate(domain="software_architecture")))]
    assert not is_duplicate(same_text, history=history, threshold_ppm=850_000).is_duplicate


def test_two_empty_fingerprints_are_not_similar():
    """Calling two contentless problems identical would reject the second for resembling the
    first, when both should fail the linter instead."""
    assert jaccard_ppm(frozenset(), frozenset()) == 0


def test_jaccard_is_a_ppm_integer():
    left, right = frozenset({1, 2, 3}), frozenset({2, 3, 4})
    similarity = jaccard_ppm(left, right)
    assert isinstance(similarity, int)
    assert similarity == 2 * PPM // 4


def test_an_embedding_duplicate_is_caught_when_the_fingerprint_misses():
    """Semantically identical, structurally different vocabulary — the case fingerprints miss."""
    verdict = is_duplicate(
        valid_candidate(),
        history=[],
        threshold_ppm=850_000,
        candidate_embedding=[1.0, 0.0, 0.0],
        embedding_history=[("c9", [0.99, 0.01, 0.0])],
    )
    assert verdict.is_duplicate
    assert verdict.detected_by == "embedding"


def test_a_negative_cosine_clamps_to_zero():
    """"Semantically opposite" is not a degree of duplication."""
    assert cosine_ppm([1.0, 0.0], [-1.0, 0.0]) == 0


def test_mismatched_embedding_dimensions_are_refused():
    """Two different embedding models produce vectors that are not commensurable."""
    with pytest.raises(ValueError, match="not commensurable"):
        cosine_ppm([1.0, 0.0], [1.0, 0.0, 0.0])


def test_a_stale_embedding_dimension_does_not_abort_dedup():
    """One entry from an old model must not stop the day's dedup."""
    verdict = is_duplicate(
        valid_candidate(),
        history=[],
        threshold_ppm=850_000,
        candidate_embedding=[1.0, 0.0],
        embedding_history=[("old", [1.0, 0.0, 0.0]), ("new", [1.0, 0.0])],
    )
    assert verdict.is_duplicate


def test_a_very_short_document_still_fingerprints():
    """An empty fingerprint is similar to nothing, so the shortest problems would never be
    detected as duplicates."""
    tiny = {"domain": "algorithms", "title": "Sort faster"}
    assert fingerprint(tiny).shingles


def test_the_fingerprint_ignores_resource_limits():
    """Every problem in a season has the same budget, so including it would give every pair a
    similarity floor and blunt the threshold."""
    a = fingerprint(valid_candidate())
    b = fingerprint(
        valid_candidate(
            resource_limits={
                "maximum_wall_time_seconds": 900,
                "maximum_rcc": 200,
                "maximum_search_calls": 50,
            }
        )
    )
    assert a == b


# --------------------------------------------------------------------------
# 7.4 step 5: the discrimination probe
# --------------------------------------------------------------------------


#: 7.4 step 5's other two thresholds, from the season config rather than from code defaults.
THRESHOLDS = {
    "minimum_degradation_gap_ppm": SEASON["challenge_generation"][
        "minimum_degradation_gap_ppm"
    ],
    "maximum_instability_ppm": SEASON["challenge_generation"]["maximum_judge_instability_ppm"],
}


def outcome(**over) -> ProbeOutcome:
    fields = dict(
        reference_scores={"a": 400_000, "b": 700_000, "c": 550_000, "d": 620_000},
        degraded_scores={"a": 200_000, "b": 400_000, "c": 300_000, "d": 350_000},
        with_mechanism=4,
        answered_by_retrieval=False,
        judge_instability_ppm=40_000,
    )
    fields.update(over)
    return ProbeOutcome(**fields)


def test_a_discriminating_problem_passes():
    verdict = assess(outcome(), minimum_spread_ppm=120_000, **THRESHOLDS)
    assert verdict.discriminates, verdict.failures


def test_a_problem_every_reference_answers_identically_is_rejected():
    """Condition 1: no spread means no information, and the slot measures nothing."""
    verdict = assess(
        outcome(reference_scores={"a": 600_000, "b": 605_000, "c": 602_000, "d": 601_000}),
        minimum_spread_ppm=120_000, **THRESHOLDS
    )
    assert not verdict.discriminates
    assert "essentially the same answer" in verdict.reason()


def test_a_problem_answered_by_retrieval_is_rejected():
    verdict = assess(outcome(answered_by_retrieval=True), minimum_spread_ppm=120_000, **THRESHOLDS)
    assert not verdict.discriminates


def test_a_problem_the_judges_cannot_score_consistently_is_rejected():
    """Condition 3: the panel disagreeing with itself means its scores are noise."""
    verdict = assess(
        outcome(judge_instability_ppm=400_000), minimum_spread_ppm=120_000, **THRESHOLDS
    )
    assert not verdict.discriminates
    assert "disagrees with itself" in verdict.reason()


def test_a_problem_where_no_reference_states_a_mechanism_is_rejected():
    verdict = assess(outcome(with_mechanism=0), minimum_spread_ppm=120_000, **THRESHOLDS)
    assert not verdict.discriminates


def test_a_problem_where_degraded_answers_score_as_well_is_rejected():
    """Condition 5, the sharpest of the five: a panel that cannot tell a damaged portfolio from a
    whole one was never measuring quality."""
    verdict = assess(
        outcome(degraded_scores={"a": 395_000, "b": 690_000, "c": 545_000, "d": 615_000}),
        minimum_spread_ppm=120_000, **THRESHOLDS
    )
    assert not verdict.discriminates
    assert "cannot tell a damaged portfolio" in verdict.reason()


def test_one_reference_is_not_enough_to_measure_spread():
    """A missing measurement is not a passing one."""
    verdict = assess(
        outcome(reference_scores={"a": 500_000}, degraded_scores={"a": 100_000}),
        minimum_spread_ppm=120_000, **THRESHOLDS
    )
    assert not verdict.discriminates
    assert "at least two" in verdict.reason()


def test_a_missing_degraded_score_is_skipped_rather_than_read_as_a_collapse():
    """Scoring an absent probe as zero would manufacture a large gap out of an outage."""
    verdict = assess(
        outcome(degraded_scores={"a": 200_000, "b": 400_000}),
        minimum_spread_ppm=120_000,
        **THRESHOLDS,
    )
    assert verdict.degradation_gap_ppm == ((400_000 - 200_000) + (700_000 - 400_000)) // 2


def test_instability_is_zero_for_a_single_judging():
    assert instability_ppm([500_000]) == 0


def test_instability_is_the_range_across_repeats():
    assert instability_ppm([500_000, 520_000, 900_000]) == 400_000


# --------------------------------------------------------------------------
# 7.5: the store, the commitment ordering, and sandbox reachability
# --------------------------------------------------------------------------


def stored_pack(date: str = "2026-08-03") -> StoredPack:
    challenges = (valid_candidate(challenge_id="sha256:" + "a" * 64),)
    return StoredPack(
        date=date,
        pack_hash=pack_hash(
            date=date, challenges=challenges, generation_protocol_version="CPG-1.0"
        ),
        challenges=challenges,
        generation_protocol_version="CPG-1.0",
        challenges_per_generator={"gpt": 1},
    )


def test_a_pack_can_be_written_and_read_back():
    store = InMemoryStore()
    pack = stored_pack()
    store.write_pack(pack, committed_hash=pack.pack_hash, ttl_days=90)
    assert store.read_pack("2026-08-03") == pack


def test_a_pack_with_no_committed_hash_is_refused():
    """7.5: the chain commitment comes first, because a store that can be edited between
    generation and commitment makes the commitment meaningless."""
    store = InMemoryStore()
    with pytest.raises(StoreError, match="chain commitment comes first"):
        store.write_pack(stored_pack(), committed_hash="", ttl_days=90)


def test_a_pack_whose_hash_differs_from_the_commitment_is_refused():
    store = InMemoryStore()
    with pytest.raises(StoreError, match="commitment does not cover"):
        store.write_pack(stored_pack(), committed_hash="sha256:" + "f" * 64, ttl_days=90)


def test_a_pack_edited_after_storage_is_detected_on_read():
    """Verified on the way *out*, which is what makes a mid-round restart safe: it distinguishes
    "recovered the committed pack" from "recovered something"."""
    store = InMemoryStore()
    pack = stored_pack()
    store.write_pack(pack, committed_hash=pack.pack_hash, ttl_days=90)
    store.packs["2026-08-03"]["challenges"][0]["title"] = "something else entirely"
    with pytest.raises(StoreError, match="not the committed pack"):
        store.read_pack("2026-08-03")


def test_a_missing_pack_reads_as_none_rather_than_raising():
    """A validator that has not generated yet is a normal state, not an error."""
    assert InMemoryStore().read_pack("2026-08-03") is None


def test_a_store_declared_sandbox_reachable_refuses_to_start():
    """A laboratory that reaches the store reads every problem in the pack, including ones it has
    not been given."""
    with pytest.raises(StoreError, match="read the entire pack"):
        assert_not_sandbox_reachable("redis://127.0.0.1:6379/0", sandbox_reachable=True)


def test_a_store_bound_to_all_interfaces_refuses_to_start():
    """The declaration is not the fact: a Redis on 0.0.0.0 is reachable whatever the config says."""
    with pytest.raises(StoreError, match="reachable from every network"):
        assert_not_sandbox_reachable("redis://0.0.0.0:6379/0", sandbox_reachable=False)


def test_a_loopback_store_starts_cleanly():
    assert_not_sandbox_reachable("redis://127.0.0.1:6379/0", sandbox_reachable=False)


def test_a_non_loopback_store_warns_but_starts(caplog):
    """A private host the validator controls can be correct; it is the configuration that goes
    wrong silently, so it is logged loudly."""
    with caplog.at_level("WARNING"):
        assert_not_sandbox_reachable("redis://10.0.1.5:6379/0", sandbox_reachable=False)
    assert "no route to it" in caplog.text


def test_run_bindings_record_which_challenge_a_run_received():
    store = InMemoryStore()
    store.bind_run("run-1", challenge_id="sha256:" + "a" * 64, miner_hotkey="5F")
    assert store.run_binding("run-1") == {
        "challenge_id": "sha256:" + "a" * 64,
        "miner_hotkey": "5F",
    }


# --------------------------------------------------------------------------
# 7.4 step 6: what the pack hash covers
# --------------------------------------------------------------------------


def test_the_pack_hash_changes_when_a_challenge_changes():
    challenges = (valid_candidate(),)
    edited = (valid_candidate(title="different"),)
    assert pack_hash(
        date="2026-08-03", challenges=challenges, generation_protocol_version="CPG-1.0"
    ) != pack_hash(date="2026-08-03", challenges=edited, generation_protocol_version="CPG-1.0")


def test_the_pack_hash_changes_with_the_date():
    challenges = (valid_candidate(),)
    assert pack_hash(
        date="2026-08-03", challenges=challenges, generation_protocol_version="CPG-1.0"
    ) != pack_hash(
        date="2026-08-04", challenges=challenges, generation_protocol_version="CPG-1.0"
    )


def test_the_pack_hash_is_stable_across_calls():
    challenges = (valid_candidate(),)
    first = pack_hash(
        date="2026-08-03", challenges=challenges, generation_protocol_version="CPG-1.0"
    )
    second = pack_hash(
        date="2026-08-03", challenges=challenges, generation_protocol_version="CPG-1.0"
    )
    assert first == second


def test_the_generator_config_refuses_more_candidates_than_it_has_variations():
    """Asking one generator the same question more often than there are variation prompts yields
    paraphrases rather than candidates, and the slot effectively had one."""
    season = json.loads(json.dumps(SEASON))
    season["challenge_generation"]["candidates_per_slot"] = 99
    with pytest.raises(ValueError, match="paraphrases rather than candidates"):
        GeneratorConfig.from_season(season)


def test_the_example_season_generator_config_parses():
    assert CONFIG.candidates_per_slot == 4
    assert set(CONFIG.generators) == {"gpt", "claude"}
    assert CONFIG.protocol_version == "CPG-1.0"


# --------------------------------------------------------------------------
# The discrimination probe cannot be skipped silently
# --------------------------------------------------------------------------
#
# `build_pack` took `probe=None` as a *default* and skipped 7.4 step 5 entirely, and the validator's
# composition passed no probe — so the strongest filter in the pipeline never ran and the resulting
# PackResult was indistinguishable from a probed one. The parameter is now required, the absence is
# recorded, and commitment refuses it unless the season says otherwise.


def test_the_probe_parameter_is_required_rather_than_defaulted():
    """Running without the discrimination check must be a decision the caller writes down."""
    import inspect

    from validator.challenge_factory.pipeline import build_pack

    assert inspect.signature(build_pack).parameters["probe"].default is inspect.Parameter.empty


def test_an_unprobed_pack_is_refused_at_commitment_by_default():
    """A committed pack is scored against every laboratory in the cohort. One that may contain
    problems on which they are all equal produces a day's ranking made of noise — which is worse
    than a day with no ranking, because it is indistinguishable from a real result."""
    from validator.challenge_factory.pipeline import PackResult, PipelineError, commit_and_store

    result = PackResult(
        date="2026-08-03",
        challenges=(valid_candidate(challenge_id="sha256:" + "a" * 64),),
        generation_protocol_version="CPG-1.0",
        challenges_per_generator={"gpt": 1},
        rejections=(),
        rcc=0,
        discrimination_probed=False,
    )
    with pytest.raises(PipelineError, match="without the discrimination probe"):
        commit_and_store(
            result,
            publish=lambda _payload: 1,
            store=InMemoryStore(),
            salt_commitment="a" * 64,
            ttl_days=90,
        )


def test_an_unprobed_pack_may_be_committed_when_the_season_permits_it():
    """A testnet standing up reference laboratories legitimately needs this. It is a declared
    degradation rather than an oversight, which is the whole difference."""
    from validator.challenge_factory.pipeline import PackResult, commit_and_store

    published: list[str] = []
    result = PackResult(
        date="2026-08-03",
        challenges=(valid_candidate(challenge_id="sha256:" + "a" * 64),),
        generation_protocol_version="CPG-1.0",
        challenges_per_generator={"gpt": 1},
        rejections=(),
        rcc=0,
        discrimination_probed=False,
    )
    digest = commit_and_store(
        result,
        publish=lambda payload: (published.append(payload), 1)[1],
        store=InMemoryStore(),
        salt_commitment="a" * 64,
        ttl_days=90,
        allow_unprobed=True,
    )
    assert digest.startswith("sha256:")
    assert published


def test_a_probed_pack_commits_without_the_flag():
    from validator.challenge_factory.pipeline import PackResult, commit_and_store

    result = PackResult(
        date="2026-08-03",
        challenges=(valid_candidate(challenge_id="sha256:" + "a" * 64),),
        generation_protocol_version="CPG-1.0",
        challenges_per_generator={"gpt": 1},
        rejections=(),
        rcc=0,
        discrimination_probed=True,
    )
    assert commit_and_store(
        result,
        publish=lambda _payload: 1,
        store=InMemoryStore(),
        salt_commitment="a" * 64,
        ttl_days=90,
    ).startswith("sha256:")


def test_the_season_declares_whether_unprobed_packs_are_allowed():
    """A threshold that decides what may be committed belongs in the hashed config, not in a
    function default where two validators could disagree."""
    assert SEASON["challenge_generation"]["allow_unprobed_packs"] is False
    assert CONFIG.allow_unprobed_packs is False
