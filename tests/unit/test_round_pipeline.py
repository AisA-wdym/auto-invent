"""The round pipeline: submissions in, standings out. architecture.md 6.1, 9, 13, 14, 17, 18.

Three modules that only exist as a composition — `submissions`, `execution`, `rounds` — and
composition is where the expensive mistakes live. Each piece they call is individually correct and
unit-tested elsewhere; what these tests hold is the joins.

The joins that matter, and what each would cost:

* A refused submission takes the round down → one miner's packaging mistake becomes everybody's
  burn.
* An execution that never ran scores as valid → a laboratory that produced nothing outranks one that
  produced something.
* An unmeasured criterion becomes zero → a judge outage costs a miner a quarter of a day's score.
* A laboratory outside the cohort gets `pairwise=0` rather than `None` → exclusion from a
  sampling decision becomes a penalty.

Every one of those is a silent failure: the round completes, the numbers look plausible, and the
ranking means something other than what it claims.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chain.client import Neuron, RegisteredCommitment, SubnetView
from protocol.commitments import SubmissionCommitment
from validator.execution import Execution, RoundExecution, as_document, execute_round, from_document
from validator.rounds import FunnelConfig, _criterion_inputs, _family_gap, _select_cohort
from validator.sandbox.container import Limits
from validator.sandbox.runner import LabResult, RunnerError
from validator.scoring.daily import ScoreHistory
from validator.scoring.gates import GateReport, GateResult
from validator.submissions import Prepared, prepare_all, submissions_for

pytestmark = pytest.mark.determinism

DIGEST = "sha256:" + "ab" * 32
OTHER = "sha256:" + "cd" * 32


def commitment(round_id="2026-08-03", digest=DIGEST, url="https://example.test/b.tar.gz"):
    return SubmissionCommitment(
        round_id=round_id, bundle_digest=digest, capsule_digest=OTHER, artifact_url=url
    )


def view(entries):
    """A snapshot with one commitment per uid."""
    return SubnetView(
        netuid=1,
        mechid=0,
        block=1_000,
        neurons=tuple(
            Neuron(uid=uid, hotkey=f"5F{uid}", coldkey="c", stake_tao=1.0,
                   validator_permit=False, active=True)
            for uid, _ in entries
        ),
        commitments=tuple(
            RegisteredCommitment(uid=uid, hotkey=f"5F{uid}", raw=item.encode(), block=900)
            for uid, item in entries
        ),
    )


# --------------------------------------------------------------------------
# submissions: which commitments are this round's
# --------------------------------------------------------------------------


def test_only_this_rounds_commitments_are_run():
    """The pallet keeps one slot per hotkey and overwrites it, so a miner who submitted yesterday
    and not today still has yesterday's commitment on chain. Running it would score a laboratory
    against a pack it was never submitted for."""
    snapshot = view(
        [(0, commitment(round_id="2026-08-03")), (1, commitment(round_id="2026-08-02"))]
    )
    found = submissions_for(snapshot, round_id="2026-08-03")
    assert [uid for uid, _hotkey, _c in found] == [0]


def test_a_commitment_from_another_protocol_is_skipped_rather_than_failing():
    """Other subnets and other protocol versions share this channel."""
    snapshot = SubnetView(
        netuid=1,
        mechid=0,
        block=1_000,
        neurons=(Neuron(0, "5F0", "c", 1.0, False, True),),
        commitments=(RegisteredCommitment(0, "5F0", "someone-elses-format", 900),),
    )
    assert submissions_for(snapshot, round_id="2026-08-03") == []


def test_identical_bundles_from_two_hotkeys_run_once(monkeypatch, tmp_path):
    """The cheapest sybil there is: register twice, submit the same laboratory, take two shares of
    one result. The lowest uid keeps its place because uid order is registration order — and the
    same handling covers an honest fork, which is indistinguishable from here."""
    monkeypatch.setattr(
        "validator.submissions._prepare_one",
        lambda **kwargs: Prepared(
            uid=kwargs["uid"], hotkey=kwargs["hotkey"], bundle_digest=DIGEST,
            image_digest=DIGEST, manifest={}, api_key="k", declared_spend_cap_usd=0,
            root=tmp_path,
        ),
    )
    snapshot = view([(3, commitment()), (1, commitment())])
    prepared = prepare_all(
        snapshot, round_id="2026-08-03", chain=object(), workspace=tmp_path
    )
    assert [lab.uid for lab in prepared.ready] == [1]
    assert [item.uid for item in prepared.refused] == [3]
    assert "identical bundle to uid 1" in prepared.refused[0].reason


def test_one_submission_failing_refuses_only_that_submission(monkeypatch, tmp_path):
    """The rule the whole module is arranged around. A raise out of `prepare_all` would take every
    other miner's day with it."""

    def prepare(**kwargs):
        if kwargs["uid"] == 1:
            raise OSError("its artifact host is down")
        return Prepared(
            uid=kwargs["uid"], hotkey=kwargs["hotkey"], bundle_digest=OTHER,
            image_digest=DIGEST, manifest={}, api_key="k", declared_spend_cap_usd=0,
            root=tmp_path,
        )

    monkeypatch.setattr("validator.submissions._prepare_one", prepare)
    snapshot = view([(1, commitment(digest=DIGEST)), (2, commitment(digest=OTHER))])
    prepared = prepare_all(
        snapshot, round_id="2026-08-03", chain=object(), workspace=tmp_path
    )
    assert [lab.uid for lab in prepared.ready] == [2]
    assert prepared.refused[0].uid == 1
    assert "host is down" in prepared.refused[0].reason


def test_a_prepared_laboratory_never_prints_its_credential():
    """A dataclass repr in a traceback is how a key reaches a log file, and logs from a round are
    published (6.3, 22)."""
    prepared = Prepared(
        uid=1, hotkey="5F1", bundle_digest=DIGEST, image_digest=DIGEST, manifest={},
        api_key="sk-or-v1-verysecret",
        declared_spend_cap_usd=0,
        root=None,  # type: ignore[arg-type]
    )
    assert "verysecret" not in repr(prepared)
    assert "<redacted>" in repr(prepared)


# --------------------------------------------------------------------------
# execution: one failure is one failed execution
# --------------------------------------------------------------------------


@dataclass
class FakeRunner:
    """Returns a portfolio, or raises, per (uid, challenge)."""

    raise_for: set[tuple[int, str]] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def execute(self, **kwargs):
        challenge_id = str(kwargs["challenge"]["challenge_id"])
        self.calls.append((kwargs["miner_hotkey"], challenge_id))
        uid = int(kwargs["miner_hotkey"].removeprefix("5F"))
        if (uid, challenge_id) in self.raise_for:
            raise RunnerError("no workspace")
        return LabResult(
            run_id=kwargs["run_id"],
            miner_hotkey=kwargs["miner_hotkey"],
            challenge_id=challenge_id,
            portfolio={"ideas": []},
            measured_usage={"rcc": 100, "search_calls": 0},
            receipt_calls=({"kind": "llm"},),
            wall_seconds=1.0,
        )


def lab(uid: int, tmp_path) -> Prepared:
    return Prepared(
        uid=uid, hotkey=f"5F{uid}", bundle_digest=DIGEST, image_digest=DIGEST,
        manifest={"model_manifest": {"portfolio": "anthropic/claude-sonnet-5"}},
        api_key="k", declared_spend_cap_usd=0, root=tmp_path,
    )


def challenge(index: int) -> dict[str, Any]:
    return {
        "challenge_id": f"sha256:{index:064x}",
        "domain": "systems",
        "resource_limits": {"maximum_rcc": 1_000_000, "maximum_search_calls": 100},
    }


def run_round(*, laboratories, challenges, runner, block=1_000, deadline=10_000, concurrency=2):
    return asyncio.run(
        execute_round(
            laboratories=laboratories,
            challenges=challenges,
            runner=runner,  # type: ignore[arg-type]
            validator_hotkey="5Gv",
            round_id="2026-08-03",
            limits=Limits(1, 1, 1, 1, 60),
            allowed_models=["anthropic/claude-sonnet-5"],
            excluded_domains=frozenset(),
            deadline_block=deadline,
            current_block=lambda: block,
            episode_seconds=60,
            concurrency=concurrency,
        )
    )


def test_every_laboratory_meets_every_challenge(tmp_path):
    runner = FakeRunner()
    executed = run_round(
        laboratories=[lab(1, tmp_path), lab(2, tmp_path)],
        challenges=[challenge(1), challenge(2)],
        runner=runner,
    )
    assert len(executed.executions) == 4
    assert len(runner.calls) == 4
    assert set(executed.by_uid) == {1, 2}


def test_one_execution_raising_does_not_end_the_round(tmp_path):
    runner = FakeRunner(raise_for={(1, f"sha256:{1:064x}")})
    executed = run_round(
        laboratories=[lab(1, tmp_path), lab(2, tmp_path)],
        challenges=[challenge(1), challenge(2)],
        runner=runner,
    )
    assert len(executed.executions) == 4
    broken = [item for item in executed.executions if item.not_attempted]
    assert len(broken) == 1
    assert broken[0].uid == 1


def test_an_execution_that_never_ran_is_not_valid(tmp_path):
    """The most expensive possible direction for this mistake. `GateReport.valid` is "every result
    passed", so an *empty* report passes vacuously — a laboratory that never ran would score as one
    that passed all thirteen gates."""
    runner = FakeRunner(raise_for={(1, f"sha256:{1:064x}")})
    executed = run_round(
        laboratories=[lab(1, tmp_path)], challenges=[challenge(1)], runner=runner
    )
    assert executed.executions[0].valid is False
    assert executed.executions[0].gates.results, "an empty gate report would pass vacuously"


def test_the_deadline_stops_the_round_and_records_what_did_not_run(tmp_path):
    """Recorded, not silently absent: a laboratory with six results out of twenty scored as though
    it had twenty is penalised for the validator running late."""
    runner = FakeRunner()
    executed = run_round(
        laboratories=[lab(1, tmp_path)],
        challenges=[challenge(1), challenge(2)],
        runner=runner,
        block=20_000,
        deadline=10_000,
    )
    assert runner.calls == []
    assert executed.stopped_at_deadline
    assert all(item.not_attempted for item in executed.executions)
    assert all(not item.valid for item in executed.executions)


def test_nothing_to_execute_is_not_an_error(tmp_path):
    empty = run_round(laboratories=[], challenges=[challenge(1)], runner=FakeRunner())
    assert empty.executions == ()


# --------------------------------------------------------------------------
# execution: surviving the gap between two scheduler steps
# --------------------------------------------------------------------------


def test_executions_round_trip_through_their_document(tmp_path):
    """Scoring is a separate step after the execution-close block, so what the containers produced
    has to survive the gap — and it cost real money, so losing it to a restart loses the round."""
    runner = FakeRunner(raise_for={(2, f"sha256:{2:064x}")})
    executed = run_round(
        laboratories=[lab(1, tmp_path), lab(2, tmp_path)],
        challenges=[challenge(1), challenge(2)],
        runner=runner,
    )
    document = json.loads(json.dumps(as_document(executed)))
    restored = from_document(document)
    assert len(restored.executions) == len(executed.executions)
    assert set(restored.by_uid) == set(executed.by_uid)
    assert [item.valid for item in restored.executions] == [
        item.valid for item in executed.executions
    ]


def test_a_document_missing_its_gate_report_raises_rather_than_passing_vacuously():
    """The same trap as above, one layer down. Defaulting `gates` to empty on read would make every
    restored execution valid."""
    with pytest.raises(KeyError):
        from_document({"executions": [{"uid": 1, "hotkey": "5F1", "challenge_id": "c",
                                       "result": None}]})


# --------------------------------------------------------------------------
# rounds: the joins that decide a score
# --------------------------------------------------------------------------


def test_an_unmeasured_criterion_is_omitted_rather_than_zeroed():
    """With originality at 25%, one judge outage written as zero is a quarter of a day's score
    removed for something that was never the miner's fault."""
    inputs = _criterion_inputs({"value": 800_000}, {})
    assert set(inputs) == {"value"}
    assert inputs["value"].pointwise_ppm == 800_000
    assert inputs["value"].pairwise_ppm is None


def test_a_laboratory_outside_the_cohort_keeps_its_screen_rather_than_scoring_zero():
    """Exclusion from the cohort is a sampling decision, not a penalty. `pairwise=0` would make it
    one."""
    inputs = _criterion_inputs({"mechanism": 600_000}, {})
    assert inputs["mechanism"].pairwise_ppm is None
    assert inputs["mechanism"].scoreable


def test_a_criterion_measured_by_neither_is_absent_entirely():
    assert _criterion_inputs({}, {}) == {}


def test_the_family_gap_needs_two_families():
    """Reporting one side's mean as a gap would flag every laboratory that failed half a pack."""
    assert _family_gap({"gpt": [500_000]}) == 0
    assert _family_gap({"gpt": [500_000], "claude": [700_000]}) == 200_000


def test_the_cohort_takes_the_top_screeners():
    screens = {
        (uid, "c1"): {"value": value}
        for uid, value in ((1, 900_000), (2, 800_000), (3, 100_000))
    }
    cohort = _select_cohort(
        screens=screens,
        funnel=FunnelConfig(2, 0, 0, 1, 0),
        seed=b"seed",
        history={},
        execution=RoundExecution(by_uid={1: (), 2: (), 3: ()}),
    )
    assert cohort == {1, 2}


def test_a_laboratory_with_no_history_joins_the_cohort_regardless_of_its_screen():
    """17.2's anti-lock-in provision. A first-day laboratory cannot screen well against incumbents
    it has never been compared with, and without this it never would be."""
    screens = {
        (uid, "c1"): {"value": value}
        for uid, value in ((1, 900_000), (2, 800_000), (9, 10_000))
    }
    cohort = _select_cohort(
        screens=screens,
        funnel=FunnelConfig(1, 0, 1, 1, 0),
        seed=b"seed",
        history={1: ScoreHistory(["d"], [500_000]), 2: ScoreHistory(["d"], [500_000])},
        execution=RoundExecution(by_uid={1: (), 2: (), 9: ()}),
    )
    assert 9 in cohort


def test_the_random_draw_is_seeded_so_two_validators_choose_the_same_cohort():
    """17.5's replication compares tournaments. Two different cohorts are two different tournaments,
    which would read as the validators disagreeing about the miners."""
    screens = {(uid, "c1"): {"value": 500_000 - uid} for uid in range(10)}
    kwargs = dict(
        screens=screens,
        funnel=FunnelConfig(2, 3, 0, 1, 0),
        history={},
        execution=RoundExecution(by_uid={uid: () for uid in range(10)}),
    )
    first = _select_cohort(seed=b"the-day's-seed", **kwargs)
    second = _select_cohort(seed=b"the-day's-seed", **kwargs)
    third = _select_cohort(seed=b"a-different-day", **kwargs)
    assert first == second
    assert first != third, "a draw that ignored the seed would be the same every day"


def test_a_round_where_nothing_passed_still_publishes_every_laboratory():
    """A round that published an empty table would be indistinguishable from a round with no
    entrants, and the miners who ran and failed would have nothing to fix."""
    from validator.rounds import _empty_labs

    failed = Execution(
        uid=1, hotkey="5F1", challenge_id="c1", result=None,
        gates=GateReport((GateResult("13.6", False, "budget exceeded"),)),
    )
    labs = _empty_labs(RoundExecution(executions=(failed,), by_uid={1: (failed,)}))
    assert len(labs) == 1
    assert labs[0].failed_gates == ("13.6 budget exceeded",)
    assert labs[0].rolling_ppm == 0
