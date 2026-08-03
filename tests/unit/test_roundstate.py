"""Round state and the disclosure gate: architecture.md 6.2, 6.3, 22.

This file is what actually holds the rule that the day's problems are not published before execution
closes. The reachability gate cannot: `RoundState.public_view` is only reported reachable through a
name conflation — `handle.read()` on a file object is indistinguishable from `RedisRoundStore.read`
— so a pin there would report success without checking anything. See the note in
`tools/reachability.py`.

The property matters more than most: a laboratory that could read a problem it has not been given
yet does not degrade the measurement, it ends it. And the dashboard lives in a separate repository,
so the guarantee has to hold at the *writer*, here, rather than depending on a renderer nobody
reviews.
"""

from __future__ import annotations

import json

import pytest

from validator.cycle import Phase
from validator.roundstate import (
    InMemoryRoundStore,
    LabStatus,
    RoundState,
    StandingEntry,
    summarise,
)

pytestmark = pytest.mark.determinism

SECRET = "the coordinator must bound its fan-out tail without hedging"

#: Every phase before execution closes. Parameterised over the *whole* enum rather than a hand-
#: picked subset, so a phase added to `validator.cycle` without a decision here shows up as a
#: failure instead of silently defaulting to disclosed.
SEALED = tuple(
    phase
    for phase in Phase
    if phase not in {Phase.SCORING, Phase.AWAITING_WEIGHTS, Phase.DONE}
)
DISCLOSED = (Phase.SCORING, Phase.AWAITING_WEIGHTS, Phase.DONE)


def challenges(count: int = 20) -> tuple[dict, ...]:
    return tuple(
        {
            "challenge_id": f"sha256:{index:064x}",
            "title": f"Problem {index}",
            "problem_statement": SECRET,
        }
        for index in range(count)
    )


def state(**over) -> RoundState:
    fields = dict(
        date="2026-08-03",
        validator_hotkey="5Gvalidator",
        phase=Phase.EXECUTING.name,
        block=7_300,
        pack_hash="sha256:" + "ab" * 32,
        challenge_count=20,
        challenges_per_generator={"gpt": 10, "claude": 10},
        floor_ppm=500_000,
        labs=(
            LabStatus(1, "5Fa", "running", 12, 20, rcc_spent=240),
            LabStatus(2, "5Fb", "complete", 20, 20, ("13.6 budget exceeded",), 400),
        ),
        standings=(
            StandingEntry(1, "5Fa", 712_000, 690_000, 20, 45_000, True, 175_000),
            StandingEntry(2, "5Fb", 480_000, 455_000, 18, 220_000, False, 0),
        ),
        challenges=challenges(),
        generation_rejections={"linter": 14, "dedup": 3},
    )
    fields.update(over)
    return RoundState(**fields)


# --------------------------------------------------------------------------
# 6.2: the problems are absent, not filtered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", SEALED, ids=lambda phase: phase.name)
def test_the_public_view_omits_the_problems_before_execution_closes(phase):
    """Checked on the serialised bytes as well as the field set.

    A future change that nested the problems under another key, or renamed them, would still be
    caught — which a `"challenges" not in view` assertion alone would not do.
    """
    view = state(phase=phase.name).public_view()
    assert "challenges" not in view
    assert SECRET not in json.dumps(view), f"{phase.name} leaked the problem text"


@pytest.mark.parametrize("phase", DISCLOSED, ids=lambda phase: phase.name)
def test_the_public_view_publishes_the_problems_once_execution_has_closed(phase):
    """6.3. Withholding forever would make the subnet unauditable, which is the opposite failure."""
    view = state(phase=phase.name).public_view()
    assert len(view["challenges"]) == 20
    assert view["challenges"][0]["problem_statement"] == SECRET


def test_the_sealed_view_says_why_rather_than_returning_an_empty_list():
    """An empty list reads as "there are no problems", which mid-generation is alarming and
    wrong."""
    view = state(phase=Phase.GENERATING.name).public_view()
    assert "sealed until execution closes" in view["challenges_withheld"]


def test_the_disclosed_view_drops_the_withheld_notice():
    """Both fields present at once would be contradictory, and a renderer would have to guess."""
    view = state(phase=Phase.DONE.name).public_view()
    assert "challenges_withheld" not in view


def test_every_phase_in_the_enum_is_classified():
    """The disclosure set is membership-tested, so a phase added without a decision would default to
    *sealed* — the safe direction, but silently. This makes the omission visible."""
    assert set(SEALED) | set(DISCLOSED) == set(Phase)
    assert not set(SEALED) & set(DISCLOSED)


def test_an_unrecognised_phase_is_an_error_rather_than_a_guess():
    """A state written by a different release must not render as `DONE` — that would claim a round
    had finished when nobody knows what it did."""
    with pytest.raises(ValueError, match="does not recognise"):
        state(phase="SOMETHING_NEW").public_view()


# --------------------------------------------------------------------------
# What is public from the start, and what is never public
# --------------------------------------------------------------------------


def test_the_pack_hash_is_public_before_reveal():
    """A commitment nobody can see commits to nothing."""
    view = state(phase=Phase.AWAITING_REVEAL.name).public_view()
    assert view["pack_hash"].startswith("sha256:")
    assert view["challenge_count"] == 20


def test_the_per_family_counts_are_public_but_not_the_per_slot_split():
    """7.4 step 6: the commitment names the counts and not which slot came from which family —
    that would tell a laboratory which half of the pack to expect from whom."""
    view = state().public_view()
    assert view["challenges_per_generator"] == {"gpt": 10, "claude": 10}
    assert not any("slot" in key for key in view)


def test_the_floor_is_public():
    """20.1's score to beat. A floor nobody can see is a floor nobody can aim at."""
    assert state().public_view()["floor_ppm"] == 500_000


def test_the_precommitted_salt_never_reaches_the_public_view():
    """It is in the recovery document and not in the published one. The salt becomes public later,
    when the pack commitment carries it forward, so this is not a secrecy requirement so much as a
    narrower surface for free — and `as_document` adds it *after* `public_view` returns, so the
    exclusion is by construction rather than by a filter somebody could remove."""
    secret = "9f" * 32
    full = state(phase=Phase.DONE.name, salt_hex=secret, salt_commitment="cd" * 32)
    view = full.public_view()
    assert "salt_hex" not in view and "salt_commitment" not in view
    assert secret not in json.dumps(view)
    assert full.as_document()["salt_hex"] == secret
    assert RoundState.from_document(full.as_document()).salt_hex == secret


def test_the_public_view_carries_no_credential_material():
    """6.3 publishes source and portfolios; credentials are excluded by construction — and this is
    the document a public web page is built from."""
    serialised = json.dumps(state(phase=Phase.DONE.name).public_view())
    for shape in ("sk-or", "sk-ant", "api_key", "key_capsule", "secret"):
        assert shape not in serialised


# --------------------------------------------------------------------------
# Two documents: the writer applies the gate exactly once
# --------------------------------------------------------------------------


def test_the_store_keeps_the_problems_while_the_public_document_withholds_them():
    """Disclosure is a property of what is *read out to the world*, not of what is written down. A
    store that withheld would leave the validator unable to recover its own round after a restart —
    and it cannot regenerate, because the seed's randomness has passed."""
    store = InMemoryRoundStore()
    store.write(state(phase=Phase.EXECUTING.name))

    recovered = store.read("2026-08-03")
    assert recovered is not None
    assert len(recovered.challenges) == 20

    public = store.read_public("2026-08-03")
    assert public is not None
    assert "challenges" not in public
    assert SECRET not in json.dumps(public)


def test_the_public_document_is_written_by_the_writer_rather_than_derived_on_read():
    """The reason the split is safe across repositories.

    The dashboard reads a document that never contained the problems, so there is no gate in the
    dashboard to get wrong and no field for a rendering change to start showing. If `read_public`
    derived the view lazily, a caller could reach past it.
    """
    store = InMemoryRoundStore()
    store.write(state(phase=Phase.EXECUTING.name))
    # The stored public document is a plain dict, already gated — not a RoundState to re-render.
    assert isinstance(store.public["2026-08-03"], dict)
    assert "challenges" not in store.public["2026-08-03"]


def test_a_phase_transition_rewrites_both_documents():
    """Round state is *meant* to change, unlike a challenge pack, which is committed and must not
    move. The public document must move with it, or the dashboard would show a sealed round after
    disclosure — or, worse, the reverse."""
    store = InMemoryRoundStore()
    store.write(state(phase=Phase.EXECUTING.name))
    store.write(state(phase=Phase.SCORING.name))

    assert store.read("2026-08-03").phase == Phase.SCORING.name
    public = store.read_public("2026-08-03")
    assert public is not None
    assert len(public["challenges"]) == 20


def test_a_round_that_was_never_written_reads_as_none():
    """A validator on its first day is a normal state, not an error."""
    store = InMemoryRoundStore()
    assert store.read("2026-08-03") is None
    assert store.read_public("2026-08-03") is None


def test_round_state_survives_a_document_round_trip():
    """The full document is the validator's recovery path, so it has to reconstruct exactly."""
    original = state()
    assert RoundState.from_document(original.as_document()) == original


# --------------------------------------------------------------------------
# Derived figures
# --------------------------------------------------------------------------


def test_progress_is_reported_per_laboratory():
    labs = {lab["uid"]: lab for lab in state().public_view()["labs"]}
    assert labs[1]["progress_ppm"] == 600_000
    assert labs[2]["progress_ppm"] == 1_000_000


def test_progress_on_a_round_with_no_challenges_is_zero_rather_than_a_division_error():
    assert LabStatus(1, "5Fa", "pending", 0, 0).progress_ppm() == 0


def test_failed_gates_are_published():
    """22 publishes hard-gate outcomes, and a miner who cannot see which gate they failed cannot
    fix it."""
    labs = {lab["uid"]: lab for lab in state().public_view()["labs"]}
    assert labs[2]["failed_gates"] == ["13.6 budget exceeded"]


def test_the_family_gap_is_published():
    """7.2.1's overfit signal. Shown rather than kept internal, because a miner who can see it can
    fix it — and one who cannot is being penalised for something invisible."""
    standings = state().public_view()["standings"]
    assert standings[1]["family_gap_ppm"] == 220_000


def test_history_is_chronological_regardless_of_write_order():
    store = InMemoryRoundStore()
    for day in (3, 1, 2):
        store.write(state(date=f"2026-08-{day:02d}"))
    summary = summarise(store.recent(10))
    assert [entry["date"] for entry in summary["history"]] == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
    ]


def test_recent_returns_newest_first():
    store = InMemoryRoundStore()
    for day in (1, 3, 2):
        store.write(state(date=f"2026-08-{day:02d}"))
    assert [round_.date for round_ in store.recent(3)] == [
        "2026-08-03",
        "2026-08-02",
        "2026-08-01",
    ]


def test_the_top_score_comes_from_the_standings_rather_than_being_recomputed():
    """A figure computed a second way would eventually disagree with the emission it claims to
    explain."""
    store = InMemoryRoundStore()
    store.write(state())
    assert summarise(store.recent(1))["history"][0]["top_rolling_ppm"] == 712_000


def test_burned_days_are_counted():
    store = InMemoryRoundStore()
    store.write(state(date="2026-08-01", burned=True))
    store.write(state(date="2026-08-02"))
    assert summarise(store.recent(10))["days_burned"] == 1


def test_an_empty_history_is_not_an_error():
    assert summarise([]) == {"rounds_recorded": 0, "history": [], "days_burned": 0}
