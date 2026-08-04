"""The observable state of a round: what the portal may show, and when.

architecture.md 6.2, 6.3 and 22. A round is public in stages, and the stages are the point:

| Phase | Public | Withheld |
|---|---|---|
| before reveal | pack **hash**, challenge count, per-family split | the problems |
| executing | which labs are running, and how far through | problems, portfolios |
| after execution close | everything: problems, portfolios, judges, gates, scores | credentials |

## Why the disclosure gate lives here rather than in the portal

A portal is a rendering layer, and rendering layers acquire "just show this one extra field"
changes. If the gate lived there, one such change would publish the day's problems mid-round — and a
laboratory that can read the problems it has not been given yet is the end of the measurement, not a
degradation of it.

So `RoundState.public_view()` is the only way to get a renderable object, it takes the phase, and it
does not carry challenge text at all until `Phase.SCORING`. The portal cannot show what it was never
given. `tests/unit/test_roundstate.py` asserts that directly, because this is the one property a
future contributor is most likely to break with the best intentions.

## Two documents, because the dashboard lives in another repository

The dashboard is a separate project (`auto-invent-dashboard`) that reads this store. That split
moves the disclosure decision to the worse place if the dashboard reads the *full* state and filters
it itself: the gate would then live in the least-reviewed repository, maintained by whoever last
touched the page.

So the validator writes **two** keys per round:

    round:{date}          the complete state, including the problems. Validator-only, for recovery.
    round:public:{date}   `public_view()` output — already gated, problems absent before disclosure.

The dashboard reads only `round:public:*`. It cannot leak the day's problems because it never
receives them, which is a stronger guarantee than "it is careful not to show them" and survives a
contributor who has never read section 6.2.

`write` stores both. `read` returns the full state — the validator's own recovery path.
`read_public` returns the gated document, and is what a dashboard's reader calls.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from validator.cycle import Phase


class StoreError(RuntimeError):
    """A round document that must not be written where it was about to be written."""

__all__ = [
    "FanoutRoundStore",
    "PublicOnlyRedisStore",
    "StoreError",
    "InMemoryRoundStore",
    "LabStatus",
    "RedisRoundStore",
    "RoundState",
    "RoundStore",
    "StandingEntry",
]

_log = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400

#: Phases at or after which the day's problems may be shown. 6.3 publishes "after the daily
#: execution window closes", which is `SCORING` onward — not `EXECUTING`, because a laboratory is
#: still running then and could read a problem it has not reached.
_DISCLOSED = frozenset({Phase.SCORING, Phase.AWAITING_WEIGHTS, Phase.DONE})


@dataclass(frozen=True, slots=True)
class LabStatus:
    """One laboratory's progress through the current round."""

    uid: int
    hotkey: str
    #: `"pending"`, `"running"`, `"complete"`, `"failed"`, or `"excluded"`.
    state: str
    challenges_done: int
    challenges_total: int
    #: Gates this laboratory has failed so far, by identifier. Published under 22.
    failed_gates: tuple[str, ...] = ()
    #: RCC measured across every challenge so far. The miner's own spend.
    rcc_spent: int = 0

    def progress_ppm(self) -> int:
        if self.challenges_total <= 0:
            return 0
        return self.challenges_done * 1_000_000 // self.challenges_total


@dataclass(frozen=True, slots=True)
class StandingEntry:
    """One laboratory's place in the standings."""

    uid: int
    hotkey: str
    rolling_score_ppm: int
    daily_score_ppm: int
    valid_challenges: int
    #: 7.2.1's overfit signal: the gap between this laboratory's score on one generator family's
    #: problems and the other's. A widening gap means it has learned a generator, not a domain — so
    #: it is shown rather than kept internal, because a miner who can see it can fix it.
    family_gap_ppm: int = 0
    qualified: bool = False
    weight_ppm: int = 0


@dataclass(frozen=True, slots=True)
class RoundState:
    """Everything the validator knows about one round. Written once per phase transition."""

    date: str
    validator_hotkey: str
    phase: str
    block: int
    #: Committed pack hash. Public from the moment it is on chain — it is a commitment, and a
    #: commitment nobody can see commits to nothing.
    pack_hash: str = ""
    challenge_count: int = 0
    #: family -> count. The *counts* are in the on-chain commitment (7.4 step 6); the per-slot
    #: attribution is not, and is not here either.
    challenges_per_generator: Mapping[str, int] = field(default_factory=dict)
    #: The qualification floor of 20.1 — the reference template's own rolling score. This is the
    #: "score to beat", and publishing it is the whole point: a floor nobody can see is a floor
    #: nobody can aim at.
    floor_ppm: int = 0
    labs: tuple[LabStatus, ...] = ()
    standings: tuple[StandingEntry, ...] = ()
    #: Present only once the phase permits it. See `public_view`.
    challenges: tuple[Mapping[str, Any], ...] = ()
    #: Rejections by pipeline step, from generation. An operator health signal (7.4).
    generation_rejections: Mapping[str, int] = field(default_factory=dict)
    #: Whether the day's emission burned because nobody cleared the floor (20.4).
    burned: bool = False
    #: Which scheduler steps have completed, by `Step.name`. This is the recovery record, and it is
    #: separate from `phase` because the two answer different questions: the phase says where the
    #: chain is, and this says what this validator has actually done. A restarting validator reading
    #: only the phase would know it was in AWAITING_RANDOMNESS and not whether it had published a
    #: salt commitment — and republishing one is the exact failure 7.3's precommitment exists to
    #: prevent.
    steps_done: tuple[str, ...] = ()
    #: The day's precommitted salt and its commitment, hex. Held so a restart between the salt
    #: commitment and generation can still derive the seed the commitment binds — without them the
    #: round is unrecoverable, because the commitment on chain is to a value only that process knew.
    #:
    #: Kept out of `public_view` deliberately. The salt does become public when the pack commitment
    #: carries it forward, but there is nothing to gain by publishing it earlier and the
    #: dashboard has no use for it, so the narrower surface is free.
    salt_hex: str = ""
    salt_commitment: str = ""
    #: Why this round was given up on, empty if it was not. Recorded rather than inferred from a
    #: missing step: "abandoned because the salt window closed" and "still working on it" look
    #: identical from a step list, and the loop would re-decide an abandoned round every tick.
    abandoned: str = ""
    updated_at_block: int = 0

    def phase_enum(self) -> Phase:
        try:
            return Phase[self.phase]
        except KeyError:
            # An unrecognised phase is a state written by a different version. Reported as such
            # rather than mapped to something plausible: a portal that rendered it as `DONE` would
            # claim a round had finished when nobody knows what it did.
            raise ValueError(
                f"round {self.date} records phase {self.phase!r}, which this version does not "
                "recognise. It was probably written by a different validator release."
            ) from None

    def disclosed(self) -> bool:
        """Whether the day's problems may be shown yet."""
        return self.phase_enum() in _DISCLOSED

    def public_view(self) -> dict[str, Any]:
        """The only renderable form. Withholds the problems until 6.3 permits them.

        Built by construction rather than by filtering: the returned dict omits `challenges`
        entirely before disclosure, so there is no field for a rendering change to start showing.
        `withheld` says so explicitly, because a portal showing an empty list where problems will
        later appear reads as "there are no problems" — which during generation is alarming and
        wrong.
        """
        view: dict[str, Any] = {
            "date": self.date,
            "validator_hotkey": self.validator_hotkey,
            "phase": self.phase,
            "block": self.block,
            "pack_hash": self.pack_hash,
            "challenge_count": self.challenge_count,
            "challenges_per_generator": dict(self.challenges_per_generator),
            "floor_ppm": self.floor_ppm,
            "burned": self.burned,
            "labs": [
                {
                    "uid": lab.uid,
                    "hotkey": lab.hotkey,
                    "state": lab.state,
                    "challenges_done": lab.challenges_done,
                    "challenges_total": lab.challenges_total,
                    "progress_ppm": lab.progress_ppm(),
                    "failed_gates": list(lab.failed_gates),
                    "rcc_spent": lab.rcc_spent,
                }
                for lab in self.labs
            ],
            "standings": [
                {
                    "uid": entry.uid,
                    "hotkey": entry.hotkey,
                    "rolling_score_ppm": entry.rolling_score_ppm,
                    "daily_score_ppm": entry.daily_score_ppm,
                    "valid_challenges": entry.valid_challenges,
                    "family_gap_ppm": entry.family_gap_ppm,
                    "qualified": entry.qualified,
                    "weight_ppm": entry.weight_ppm,
                }
                for entry in self.standings
            ],
            "generation_rejections": dict(self.generation_rejections),
            "steps_done": list(self.steps_done),
            "abandoned": self.abandoned,
        }
        if self.disclosed():
            view["challenges"] = [dict(challenge) for challenge in self.challenges]
        else:
            view["challenges_withheld"] = (
                "The day's problems are sealed until execution closes (6.2). A laboratory that "
                "could read a problem it has not been given yet would end the measurement rather "
                "than degrade it."
            )
        return view

    def as_document(self) -> dict[str, Any]:
        """The complete state, for the store. Includes the problems at every phase.

        Distinct from `public_view` on purpose: the store holds everything, and disclosure is a
        property of *reading it out to the world* rather than of writing it down. A store that
        withheld would make the validator unable to recover its own round after a restart.
        """
        document = self.public_view()
        document.pop("challenges_withheld", None)
        document["challenges"] = [dict(challenge) for challenge in self.challenges]
        document["updated_at_block"] = self.updated_at_block
        # Added here rather than in `public_view`, which is what keeps them out of the published
        # document by construction instead of by a filter somebody could remove.
        document["salt_hex"] = self.salt_hex
        document["salt_commitment"] = self.salt_commitment
        return document

    @classmethod
    def from_document(cls, body: Mapping[str, Any]) -> RoundState:
        return cls(
            date=str(body["date"]),
            validator_hotkey=str(body["validator_hotkey"]),
            phase=str(body["phase"]),
            block=int(body["block"]),
            pack_hash=str(body.get("pack_hash", "")),
            challenge_count=int(body.get("challenge_count", 0)),
            challenges_per_generator=dict(body.get("challenges_per_generator", {})),
            floor_ppm=int(body.get("floor_ppm", 0)),
            labs=tuple(
                LabStatus(
                    uid=int(lab["uid"]),
                    hotkey=str(lab["hotkey"]),
                    state=str(lab["state"]),
                    challenges_done=int(lab["challenges_done"]),
                    challenges_total=int(lab["challenges_total"]),
                    failed_gates=tuple(lab.get("failed_gates", ())),
                    rcc_spent=int(lab.get("rcc_spent", 0)),
                )
                for lab in body.get("labs", ())
            ),
            standings=tuple(
                StandingEntry(
                    uid=int(entry["uid"]),
                    hotkey=str(entry["hotkey"]),
                    rolling_score_ppm=int(entry["rolling_score_ppm"]),
                    daily_score_ppm=int(entry.get("daily_score_ppm", 0)),
                    valid_challenges=int(entry.get("valid_challenges", 0)),
                    family_gap_ppm=int(entry.get("family_gap_ppm", 0)),
                    qualified=bool(entry.get("qualified", False)),
                    weight_ppm=int(entry.get("weight_ppm", 0)),
                )
                for entry in body.get("standings", ())
            ),
            challenges=tuple(body.get("challenges", ())),
            generation_rejections=dict(body.get("generation_rejections", {})),
            burned=bool(body.get("burned", False)),
            steps_done=tuple(str(step) for step in body.get("steps_done", ())),
            abandoned=str(body.get("abandoned", "")),
            salt_hex=str(body.get("salt_hex", "")),
            salt_commitment=str(body.get("salt_commitment", "")),
            updated_at_block=int(body.get("updated_at_block", 0)),
        )


class RoundStore(Protocol):
    """What the validator writes. The dashboard has its own reader for the public half.

    `read` returns the full state and exists for the validator's own recovery after a restart.
    `read_public` returns the already-gated document, which is the only thing another process should
    ever consume.
    """

    def write(self, state: RoundState, *, ttl_days: int) -> None: ...
    def read(self, date: str) -> RoundState | None: ...
    def read_public(self, date: str) -> dict[str, Any] | None: ...
    def recent(self, limit: int) -> list[RoundState]: ...


@dataclass
class InMemoryRoundStore:
    """For tests, and for a single-process deployment."""

    rounds: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: date -> the gated document, written alongside the full one. Separate rather than derived on
    #: read, so the *writer* applies the gate exactly once and no reader can skip it.
    public: dict[str, dict[str, Any]] = field(default_factory=dict)

    def write(self, state: RoundState, *, ttl_days: int = 90) -> None:
        self.rounds[state.date] = state.as_document()
        self.public[state.date] = state.public_view()

    def read(self, date: str) -> RoundState | None:
        body = self.rounds.get(date)
        return RoundState.from_document(body) if body else None

    def read_public(self, date: str) -> dict[str, Any] | None:
        return self.public.get(date)

    def recent(self, limit: int) -> list[RoundState]:
        # Reverse date order. Dates are ISO, so lexical order is chronological — which is why the
        # key is a date string rather than an epoch.
        return [
            RoundState.from_document(self.rounds[date])
            for date in sorted(self.rounds, reverse=True)[:limit]
        ]


@dataclass
class RedisRoundStore:
    """The production store. Shares the validator's Redis, on its own key prefix.

    Same instance as the challenge packs, and the same rule applies: **not reachable from the
    sandbox**. Round state includes the day's problems, so a laboratory that could read this store
    could read the pack — the fact that it arrived by a different key prefix would not help.
    """

    url: str = "redis://127.0.0.1:6379/0"
    namespace: str = "auto-invent:round"
    _client: Any = field(default=None, repr=False)

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def write(self, state: RoundState, *, ttl_days: int = 90) -> None:
        """Write the full state and the gated public document, in one pipeline.

        Overwrites, unlike the challenge pack. Round state is *meant* to change — a phase transition
        is a rewrite — whereas a pack is committed and must not move. Two objects, two rules.

        Both keys are written in one pipeline so a reader never sees a public document from one
        phase beside a full state from another. Without that, a dashboard polling across the write
        could render a round as still executing while the problems had already been disclosed — or
        the reverse, which is the direction that leaks.
        """
        client = self._redis()
        expiry = ttl_days * _SECONDS_PER_DAY
        pipeline = client.pipeline()
        pipeline.set(
            f"{self.namespace}:{state.date}",
            json.dumps(state.as_document(), sort_keys=True),
            ex=expiry,
        )
        pipeline.set(
            f"{self.namespace}:public:{state.date}",
            json.dumps(state.public_view(), sort_keys=True),
            ex=expiry,
        )
        pipeline.execute()

    def read(self, date: str) -> RoundState | None:
        raw = self._redis().get(f"{self.namespace}:{date}")
        return RoundState.from_document(json.loads(raw)) if raw else None

    def read_public(self, date: str) -> dict[str, Any] | None:
        raw = self._redis().get(f"{self.namespace}:public:{date}")
        return json.loads(raw) if raw else None

    def recent(self, limit: int) -> list[RoundState]:
        client = self._redis()
        prefix = f"{self.namespace}:"
        # `scan_iter`, not `keys`: `keys` blocks the server for the whole scan, and the dashboard
        # polls this every fifteen seconds per open tab.
        dates = sorted(
            (
                key.removeprefix(prefix)
                for key in client.scan_iter(f"{prefix}*", count=200)
                # `public:` shares the prefix; excluded so a scan does not treat a gated document
                # as a date.
                if not key.removeprefix(prefix).startswith("public:")
            ),
            reverse=True,
        )[:limit]
        found: list[RoundState] = []
        for date in dates:
            raw = client.get(f"{prefix}{date}")
            if raw:
                found.append(RoundState.from_document(json.loads(raw)))
        return found


def summarise(rounds: Sequence[RoundState]) -> dict[str, Any]:
    """Cross-round figures for the portal's header.

    `top_score_per_round` is what a history chart draws. Taken from the standings rather than
    recomputed, because the standings are what the weight vector was built from — a chart computed
    a second way would eventually disagree with the emission it claims to explain.
    """
    history = [
        {
            "date": state.date,
            "top_rolling_ppm": max(
                (entry.rolling_score_ppm for entry in state.standings), default=0
            ),
            "floor_ppm": state.floor_ppm,
            "qualified": sum(1 for entry in state.standings if entry.qualified),
            "burned": state.burned,
        }
        for state in sorted(rounds, key=lambda state: state.date)
    ]
    return {
        "rounds_recorded": len(rounds),
        "history": history,
        "days_burned": sum(1 for entry in history if entry["burned"]),
    }


@dataclass
class PublicOnlyRedisStore:
    """A write target that *cannot* hold the day's problems.

    The reason this exists is a deployment shape the design did not anticipate. `RedisRoundStore`
    writes two keys to one Redis — `round:{date}`, the full document including the problems, and
    `round:public:{date}`, the gated one. That is correct while the Redis is the validator's own and
    unreachable from anywhere else.

    It stops being correct the moment the dashboard runs somewhere the validator's Redis cannot be
    reached privately. Exposing that Redis publicly would put the day's problems one password away
    from anyone, and 6.2 is the guarantee that a laboratory cannot read a problem before it is given
    one. A leak there does not degrade the measurement, it ends it.

    So the public surface gets its own store, and the safety is structural rather than a decision:
    this class never calls `as_document()`. There is no path through it that writes the full state,
    so no future edit can make it leak by forgetting a filter.
    """

    url: str = "redis://127.0.0.1:6379/0"
    namespace: str = "round"
    _client: Any = field(default=None, repr=False)

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def write(self, state: RoundState, *, ttl_days: int = 90) -> None:
        """Publish the gated document, and only that.

        One key per round, named the same as in the private store so a reader is identical either
        way — the dashboard does not know or care which shape of store it is pointed at.
        """
        document = state.public_view()
        # A last check on the way out. `public_view` is already the gate and this is not a second
        # one: it is an assertion that the gate ran, because this store may be internet-reachable
        # and the cost of being wrong here is the whole measurement.
        if not state.disclosed() and "challenges" in document:
            raise StoreError(
                f"refusing to publish {state.date}: the document carries `challenges` while the "
                f"round is in {state.phase}, which 6.2 does not disclose. This store may be "
                "publicly reachable, so a gate that failed here would publish the day's problems "
                "to anyone."
            )
        self._redis().set(
            f"{self.namespace}:public:{state.date}",
            json.dumps(document, sort_keys=True),
            ex=ttl_days * _SECONDS_PER_DAY,
        )

    def read(self, date: str) -> RoundState | None:
        """Always `None`. This store holds no full documents to read.

        Implemented rather than omitted so the type is a `RoundStore` — and returning `None` is
        honest: a validator that recovered its round from here would be recovering a document with
        no problems in it.
        """
        return None

    def read_public(self, date: str) -> dict[str, Any] | None:
        raw = self._redis().get(f"{self.namespace}:public:{date}")
        return json.loads(raw) if raw else None

    def recent(self, limit: int) -> list[RoundState]:
        """Empty. `recent` returns full states, and there are none here."""
        return []


@dataclass
class FanoutRoundStore:
    """Writes to the validator's own store, then publishes the gated document.

    Reads come from `primary` only. The publish target is write-only as far as the validator is
    concerned: it is the dashboard's copy, and a validator that read its own round back from a
    publicly writable place would be trusting a document anyone could have changed.

    A failure to publish is logged and does not fail the round. The private write is what the
    validator needs to recover; the public one is what a web page needs to render, and losing a
    render is not worth losing a day over.
    """

    primary: RoundStore
    publish: PublicOnlyRedisStore

    def write(self, state: RoundState, *, ttl_days: int = 90) -> None:
        self.primary.write(state, ttl_days=ttl_days)
        try:
            self.publish.write(state, ttl_days=ttl_days)
        except StoreError:
            # A gate failure is not a publishing hiccup. Re-raised, because it means the document
            # would have disclosed problems early and that is worth stopping for.
            raise
        except Exception as error:  # noqa: BLE001 - redis raises a wide family
            _log.error(
                "could not publish round %s to the dashboard store (%s). The round continues; the "
                "public page will show the last document it received.",
                state.date,
                error,
            )

    def read(self, date: str) -> RoundState | None:
        return self.primary.read(date)

    def read_public(self, date: str) -> dict[str, Any] | None:
        return self.primary.read_public(date)

    def recent(self, limit: int) -> list[RoundState]:
        return self.primary.recent(limit)
