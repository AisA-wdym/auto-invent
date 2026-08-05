"""Step 6's persistence and the dedup window: Redis, and nothing the sandbox can reach.

architecture.md 7.5. Three key families, three lifetimes:

| Key | Contents | Lifetime |
|---|---|---|
| `pack:{date}` | the day's challenges plus the committed hash | the dedup window |
| `dedup:{fingerprint}` | fingerprints and embeddings of past challenges | `dedup_lookback_days` |
| `run:{run_id}` | which challenge a run was issued, for reconciliation | until publication |

## The pack hash is committed before the pack is stored

`write_pack` refuses a pack whose hash is not already on chain. Writing to Redis is not the
commitment: a store that could be edited between generation and commitment would make the
commitment meaningless. Ordering it this way removes the window rather than trusting nobody uses
it, and the refusal is here — in the only function that can write a pack — rather than in the
pipeline that calls it.

## Redis is not reachable from the sandbox

Worth stating in code as well as prose, because "serve the problems to miners from Redis" reads as
though the miner should fetch them. It must not: a laboratory that could reach Redis could read
the entire pack — every problem including the ones it has not been given, and other rounds' packs.

The challenge reaches the laboratory as 9.1's structured input, delivered by the runner. Two
things enforce that: the sandbox network has no route to Redis (`validator/sandbox/`), and
`assert_not_sandbox_reachable` refuses to start against a Redis bound to a routable address.

## Restart mid-round must not lose a committed pack

The reason this is Redis rather than process memory. A validator that crashed after committing a
pack hash and before finishing execution would otherwise have committed to a pack it can no longer
produce — and cannot regenerate, because generation is seeded and the seed's randomness has passed.
`read_pack` verifies the stored pack against the committed hash on the way out, so a restart either
recovers the exact committed pack or reports that it cannot.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from protocol.canonical import digest_object
from validator.challenge_factory.dedup import Fingerprint

__all__ = [
    "ChallengeStore",
    "InMemoryStore",
    "RedisStore",
    "StoreError",
    "StoredPack",
    "assert_not_sandbox_reachable",
]

_log = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400


class StoreError(RuntimeError):
    """The store refused an operation, or cannot vouch for what it returned."""


@dataclass(frozen=True, slots=True)
class StoredPack:
    """A day's committed pack as it comes back out of the store."""

    date: str
    pack_hash: str
    challenges: tuple[Mapping[str, Any], ...]
    generation_protocol_version: str
    challenges_per_generator: Mapping[str, int]

    def verify(self) -> None:
        """Recompute the hash and compare.

        On the way *out*, not only on the way in. A store is mutable and the hash is on chain;
        checking at read time is what makes a mid-round restart safe, because it distinguishes
        "recovered the committed pack" from "recovered something".
        """
        recomputed = digest_object(
            {
                "date": self.date,
                "generation_protocol_version": self.generation_protocol_version,
                "challenges": [dict(challenge) for challenge in self.challenges],
            }
        )
        if recomputed != self.pack_hash:
            raise StoreError(
                f"pack for {self.date} hashes to {recomputed} but {self.pack_hash} was committed "
                "on chain. The stored pack is not the committed pack: it was edited, or a "
                "different pack was written under this date. Neither can be scored against, "
                "because every laboratory's result would be attributed to a pack nobody can "
                "reproduce."
            )


class ChallengeStore(Protocol):
    """What the pipeline needs from a store."""

    def write_pack(self, pack: StoredPack, *, committed_hash: str, ttl_days: int) -> None: ...
    def read_pack(self, date: str) -> StoredPack | None: ...
    def write_executions(self, date: str, body: Mapping[str, Any], *, ttl_days: int) -> None: ...
    def read_executions(self, date: str) -> Mapping[str, Any] | None: ...
    def record_fingerprints(
        self, entries: Sequence[tuple[str, Fingerprint]], *, ttl_days: int
    ) -> None: ...
    def fingerprints(self) -> list[tuple[str, Fingerprint]]: ...
    def record_embedding(
        self, challenge_id: str, vector: Sequence[float], *, ttl_days: int
    ) -> None: ...
    def embeddings(self) -> list[tuple[str, Sequence[float]]]: ...
    def bind_run(self, run_id: str, *, challenge_id: str, miner_hotkey: str) -> None: ...
    def run_binding(self, run_id: str) -> Mapping[str, str] | None: ...


def _pack_body(pack: StoredPack) -> dict[str, Any]:
    return {
        "date": pack.date,
        "pack_hash": pack.pack_hash,
        "generation_protocol_version": pack.generation_protocol_version,
        "challenges_per_generator": dict(pack.challenges_per_generator),
        "challenges": [dict(challenge) for challenge in pack.challenges],
    }


def _pack_from_body(body: Mapping[str, Any]) -> StoredPack:
    return StoredPack(
        date=str(body["date"]),
        pack_hash=str(body["pack_hash"]),
        challenges=tuple(body["challenges"]),
        generation_protocol_version=str(body["generation_protocol_version"]),
        challenges_per_generator=dict(body.get("challenges_per_generator", {})),
    )


def _guard_commitment(pack: StoredPack, committed_hash: str) -> None:
    """Refuse a write whose hash was not the one committed on chain.

    In both store implementations via one function, so the in-memory store used by tests cannot
    be more permissive than the real one — a test double that accepted an uncommitted pack would
    make the whole ordering property untested.
    """
    if not committed_hash:
        raise StoreError(
            f"refusing to store the pack for {pack.date}: no committed hash was supplied. The "
            "chain commitment comes first, because a store that can be edited between "
            "generation and commitment makes the commitment meaningless."
        )
    if committed_hash != pack.pack_hash:
        raise StoreError(
            f"pack for {pack.date} hashes to {pack.pack_hash} but {committed_hash} was committed. "
            "Storing it would mean serving challenges the commitment does not cover."
        )
    pack.verify()


@dataclass
class InMemoryStore:
    """A store for tests and for a validator running without Redis.

    Not a mock — it enforces the same commitment ordering and the same read-time verification, so
    a test that passes here is testing the rule rather than the double. What it does not do is
    survive a restart, which is the one reason production wants Redis; a validator using this and
    restarting mid-round will find no pack and must say so rather than regenerate.
    """

    packs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: date -> what the execution phase produced. Held here rather than in round state because it is
    #: working data between two scheduler steps: execution closes at one block and scoring begins at
    #: another, so the portfolios have to survive the gap — and a restart in between must find them
    #: or the round is unrecoverable, the executions having cost real money to produce.
    executions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _fingerprints: dict[str, Fingerprint] = field(default_factory=dict)
    _embeddings: dict[str, Sequence[float]] = field(default_factory=dict)
    runs: dict[str, Mapping[str, str]] = field(default_factory=dict)

    def write_pack(self, pack: StoredPack, *, committed_hash: str, ttl_days: int) -> None:
        _guard_commitment(pack, committed_hash)
        self.packs[pack.date] = _pack_body(pack)

    def read_pack(self, date: str) -> StoredPack | None:
        body = self.packs.get(date)
        if body is None:
            return None
        pack = _pack_from_body(body)
        pack.verify()
        return pack

    def write_executions(self, date: str, body: Mapping[str, Any], *, ttl_days: int) -> None:
        self.executions[date] = json.loads(json.dumps(body))

    def read_executions(self, date: str) -> Mapping[str, Any] | None:
        return self.executions.get(date)

    def record_fingerprints(
        self, entries: Sequence[tuple[str, Fingerprint]], *, ttl_days: int
    ) -> None:
        for challenge_id, print_ in entries:
            self._fingerprints[challenge_id] = print_

    def fingerprints(self) -> list[tuple[str, Fingerprint]]:
        return sorted(self._fingerprints.items())

    def record_embedding(
        self, challenge_id: str, vector: Sequence[float], *, ttl_days: int
    ) -> None:
        self._embeddings[challenge_id] = list(vector)

    def embeddings(self) -> list[tuple[str, Sequence[float]]]:
        return sorted(self._embeddings.items())

    def bind_run(self, run_id: str, *, challenge_id: str, miner_hotkey: str) -> None:
        self.runs[run_id] = {"challenge_id": challenge_id, "miner_hotkey": miner_hotkey}

    def run_binding(self, run_id: str) -> Mapping[str, str] | None:
        return self.runs.get(run_id)


@dataclass
class RedisStore:
    """The production store. `redis` is imported lazily so nothing else pulls it in."""

    url: str = "redis://127.0.0.1:6379/0"
    namespace: str = "auto-invent"
    _client: Any = field(default=None, repr=False)

    def _redis(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def _key(self, *parts: str) -> str:
        return ":".join((self.namespace, *parts))

    def write_pack(self, pack: StoredPack, *, committed_hash: str, ttl_days: int) -> None:
        _guard_commitment(pack, committed_hash)
        client = self._redis()
        key = self._key("pack", pack.date)
        # SET with NX: a pack already stored for this date is not overwritten. A second write for
        # one date means either a retry (harmless, and the stored copy is already the committed
        # one) or a regeneration attempt after commitment (which must not silently replace what
        # the chain vouches for).
        stored = client.set(
            key,
            json.dumps(_pack_body(pack), sort_keys=True),
            nx=True,
            ex=ttl_days * _SECONDS_PER_DAY,
        )
        if not stored:
            existing = self.read_pack(pack.date)
            if existing is not None and existing.pack_hash == pack.pack_hash:
                _log.info("pack for %s was already stored; the stored copy matches", pack.date)
                return
            raise StoreError(
                f"a different pack is already stored for {pack.date}. Overwriting would replace "
                "the pack the chain commitment vouches for."
            )

    def read_pack(self, date: str) -> StoredPack | None:
        raw = self._redis().get(self._key("pack", date))
        if raw is None:
            return None
        pack = _pack_from_body(json.loads(raw))
        pack.verify()
        return pack

    def write_executions(self, date: str, body: Mapping[str, Any], *, ttl_days: int) -> None:
        """What the execution phase produced, for the scoring step to read.

        Overwrites, unlike `write_pack`. A pack is committed on chain and must not move; executions
        are this validator's own working record between two steps, and a retried execution phase
        writing a second copy beside the first would leave the scorer choosing between them.
        """
        self._redis().set(
            self._key("executions", date),
            json.dumps(body, sort_keys=True),
            ex=ttl_days * _SECONDS_PER_DAY,
        )

    def read_executions(self, date: str) -> Mapping[str, Any] | None:
        raw = self._redis().get(self._key("executions", date))
        return json.loads(raw) if raw is not None else None

    def record_fingerprints(
        self, entries: Sequence[tuple[str, Fingerprint]], *, ttl_days: int
    ) -> None:
        client = self._redis()
        pipeline = client.pipeline()
        for challenge_id, print_ in entries:
            pipeline.set(
                self._key("dedup", challenge_id),
                json.dumps({"domain": print_.domain, "shingles": sorted(print_.shingles)}),
                ex=ttl_days * _SECONDS_PER_DAY,
            )
        pipeline.execute()

    def fingerprints(self) -> list[tuple[str, Fingerprint]]:
        client = self._redis()
        prefix = self._key("dedup", "")
        found: list[tuple[str, Fingerprint]] = []
        # `scan_iter`, not `keys`: `keys` blocks the server for the whole scan, and at 1,800
        # challenges with a shared Redis that is a visible stall on every other client.
        for key in client.scan_iter(match=f"{prefix}*", count=500):
            raw = client.get(key)
            if raw is None:
                continue
            body = json.loads(raw)
            found.append(
                (
                    key.removeprefix(prefix),
                    Fingerprint(
                        domain=str(body["domain"]), shingles=frozenset(body["shingles"])
                    ),
                )
            )
        return sorted(found)

    def record_embedding(
        self, challenge_id: str, vector: Sequence[float], *, ttl_days: int
    ) -> None:
        self._redis().set(
            self._key("embed", challenge_id),
            json.dumps(list(vector)),
            ex=ttl_days * _SECONDS_PER_DAY,
        )

    def embeddings(self) -> list[tuple[str, Sequence[float]]]:
        client = self._redis()
        prefix = self._key("embed", "")
        found: list[tuple[str, Sequence[float]]] = []
        for key in client.scan_iter(match=f"{prefix}*", count=500):
            raw = client.get(key)
            if raw is not None:
                found.append((key.removeprefix(prefix), json.loads(raw)))
        return sorted(found)

    def bind_run(self, run_id: str, *, challenge_id: str, miner_hotkey: str) -> None:
        self._redis().hset(
            self._key("run", run_id),
            mapping={"challenge_id": challenge_id, "miner_hotkey": miner_hotkey},
        )

    def run_binding(self, run_id: str) -> Mapping[str, str] | None:
        binding = self._redis().hgetall(self._key("run", run_id))
        return binding or None


def assert_not_sandbox_reachable(url: str, *, sandbox_reachable: bool) -> None:
    """Refuse to start if the store is reachable from the sandbox network.

    Two checks, and the first is the one that catches a real misconfiguration. `sandbox_reachable`
    in the season config is a *declaration*, and a declaration is not a fact — so the URL's host is
    checked too. A Redis on `0.0.0.0` or a routable address is reachable from any container on any
    network the validator host can see, whatever the config says.

    A laboratory that reached Redis could read the whole pack: every problem including ones it has
    not been given, and other rounds' packs. That is not a degradation of the measurement; it is
    the end of it.
    """
    if sandbox_reachable:
        raise StoreError(
            "the season config declares the challenge store reachable from the sandbox. A "
            "laboratory that can reach Redis can read the entire pack — including the problems it "
            "has not been given and other rounds' packs. Set sandbox_reachable to false; the "
            "challenge reaches the laboratory as 9.1's structured input, delivered by the runner."
        )

    host = _host_of(url)
    if host in {"0.0.0.0", "::", "*"}:  # noqa: S104 - naming the value in order to refuse it
        raise StoreError(
            f"the challenge store is bound to {host}, which is reachable from every network the "
            "validator host can see, including the sandbox network. Bind it to 127.0.0.1 or to a "
            "network the sandbox is not attached to."
        )
    if host not in {"127.0.0.1", "localhost", "::1"} and not host.startswith("/"):
        # Not fatal: a validator may legitimately run Redis on a private host it controls. Logged
        # loudly because it is the configuration that goes wrong silently, and the failure is
        # invisible until a miner's score is unexplainable.
        _log.warning(
            "the challenge store is at %r, not a loopback address. That can be correct, but the "
            "sandbox network must have no route to it — verify with `docker network inspect`, "
            "because a laboratory that reaches this store reads every problem in the pack.",
            host,
        )


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme in {"unix", "redis+unix"}:
        return parsed.path or "/"
    return (parsed.hostname or "").strip("[]")
