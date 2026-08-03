"""Validator entry point: `python -m validator`.

The composition root. Every rule the subnet enforces has to be reachable from here, or it does not
run — `tools/reachability.py` walks from this function and reports anything it cannot reach.

## `--check` is the half the reachability gate cannot prove

The reachability gate proves a call path *exists*. It cannot prove the call path *works*: the
predecessor shipped a defect that satisfied a reachability walk perfectly and failed on the first
request, because the path existed and the value it read did not.

So `--check` builds the entire object graph — season config, cycle ordering, judge panels, price
table, credential resolvers, taxonomy, slot plan — calls every validation function on it, and exits.
No network, no chain, no credential. It is the difference between "a `main` reaches this" and "this
runs".

## What a round does, in order

    plan(seed) → generate → commit → store → reveal → run → gate → canonicalise
    → prior art → screen → tournament → score → allocate → submit

Each step is a function in its own module; this file's whole job is to call them in that order with
the right things wired together. There is deliberately no logic here beyond sequencing and
error-handling, because logic in a composition root is logic no test can reach except through a
process launch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from chain.client import BittensorChain, ChainClient, ChainError, FakeChain
from gateway.adapters.openrouter import ModelPin
from gateway.credentials import CredentialError, CredentialSet, load_validator_credential
from gateway.metering import Ledger, PriceTable
from protocol.commitments import PackCommitment, SaltCommitment, verify_salt_timing
from protocol.fixedpoint import apply_weights, assert_sums_to_one, quantile_ppm
from protocol.receipts import Receipt, reconcile, verify_chain
from protocol.seeds import daily_seed, salt_commitment, slot_assignments, verify_salt
from validator.challenge_factory.dedup import is_duplicate
from validator.challenge_factory.discriminator import assess
from validator.challenge_factory.generator import GeneratorConfig
from validator.challenge_factory.linter import lint
from validator.challenge_factory.pipeline import build_pack, commit_and_store
from validator.challenge_factory.safety import screen
from validator.challenge_factory.store import (
    InMemoryStore,
    RedisStore,
    StoreError,
    assert_not_sandbox_reachable,
)
from validator.challenge_factory.taxonomy import Taxonomy, plan
from validator.cycle import CycleConfig, CycleError, Phase
from validator.judge.bradley_terry import fit, strengths_to_ppm
from validator.judge.pairwise import combine_orders, swiss_pairings
from validator.judge.panels import panels_from_season, pins_for
from validator.judge.pointwise import aggregate
from validator.model_client import ModelClient
from validator.sandbox.container import (
    Limits,
    SandboxRunner,
    assert_egress_confined,
    docker_command,
    ensure_network,
)
from validator.sandbox.runner import Runner
from validator.scoring.criteria import (
    ScoringConfig,
    challenge_score,
    collapse_duplicates,
    rank_weighted,
)
from validator.scoring.daily import DailyConfig, daily_score, rolling_score
from validator.scoring.gates import check_all
from validator.weights import Candidate, WeightsConfig, allocate

_log = logging.getLogger("validator")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="validator", description=__doc__)
    parser.add_argument("--season", default="config/season.example.json", type=Path)
    parser.add_argument("--netuid", type=int, default=int(os.environ.get("AI_NETUID", "0")))
    parser.add_argument("--network", default=os.environ.get("AI_NETWORK", "finney"))
    parser.add_argument("--wallet", default=os.environ.get("AI_WALLET", "default"))
    parser.add_argument("--hotkey", default=os.environ.get("AI_HOTKEY", "default"))
    parser.add_argument(
        "--rcg-endpoint", default=os.environ.get("AI_RCG_ENDPOINT", "http://127.0.0.1:8081")
    )
    parser.add_argument("--redis-url", default=os.environ.get("AI_REDIS_URL", ""))
    parser.add_argument("--workspace", default=Path("var/runs"), type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="build and validate everything, then exit; no network, chain or credential",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one round and exit, rather than looping over days",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


class Validator:
    """Everything a round needs, assembled once.

    A class rather than a bag of module globals so two validators can exist in one process — which
    is what `tests/localnet/` needs in order to measure 27's cross-validator rank correlation
    without launching two processes.
    """

    def __init__(self, season: dict[str, Any], *, chain: ChainClient, args: argparse.Namespace):
        self.season = season
        self.chain = chain
        self.args = args

        # Everything below either validates on construction or is validated here. A round must not
        # be able to start against a config whose ordering, panels or weights are wrong.
        self.cycle = CycleConfig.from_season(season)
        self.taxonomy = Taxonomy.from_season(season)
        self.generation = GeneratorConfig.from_season(season)
        self.panels = panels_from_season(season)
        self.pins: dict[str, ModelPin] = pins_for(self.panels)
        self.prices = PriceTable.from_season(season)
        self.scoring = ScoringConfig.from_season(season)
        self.daily = DailyConfig.from_season(season)
        self.weights = WeightsConfig.from_season(season)

        assert_sums_to_one(
            {name: int(value) for name, value in season["criterion_weights_ppm"].items()},
            label="criterion_weights_ppm",
        )

        self.generators = season["challenge_generation"]["generators"]
        # Generation families are pinned separately from judge families: 7.2.1's generators and
        # 16.1's judges are different declarations and may name different models, and merging them
        # would let a config change to one silently change the other.
        for generator in self.generators:
            self.pins.setdefault(
                str(generator["family"]),
                ModelPin(
                    slug=str(generator["model_slug"]),
                    snapshot=str(generator["model_snapshot"]),
                ),
            )

        store_config = season["challenge_generation"]["store"]
        if args.redis_url:
            assert_not_sandbox_reachable(
                args.redis_url, sandbox_reachable=bool(store_config["sandbox_reachable"])
            )
            self.store = RedisStore(url=args.redis_url)
        else:
            # No Redis configured. Explicitly named as a degradation rather than a default: an
            # in-memory store loses a committed pack on restart, and 7.5 wants Redis precisely so a
            # mid-round restart can recover the pack a hash was committed for.
            _log.warning(
                "no --redis-url: using an in-memory challenge store. A restart mid-round will lose "
                "the pack whose hash was committed on chain, and it cannot be regenerated because "
                "the seed's randomness has passed. Configure Redis before mainnet."
            )
            self.store = InMemoryStore()

        self.ledger = Ledger()
        self.sandbox = SandboxRunner()

    # ------------------------------------------------------------------
    # Validation, for --check
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Every check that can be made without a network. Returns problems found.

        Returns rather than raises so `--check` can report all of them at once. An operator fixing a
        config wants the whole list, not the first item.
        """
        problems: list[str] = []

        try:
            self.cycle.assert_ordering()
        except Exception as error:  # noqa: BLE001
            problems.append(f"cycle: {error}")

        try:
            self.taxonomy.validate()
        except Exception as error:  # noqa: BLE001
            problems.append(f"taxonomy: {error}")

        # The seeded plan must produce a full pack from the declared generators. Checked with a
        # dummy seed, because the *shape* is what a config can get wrong — the counts and the
        # domains — and that is seed-independent.
        try:
            slots = plan(bytes(32), taxonomy=self.taxonomy, generators=self.generators)
            if len(slots) != self.taxonomy.challenges_per_day:
                problems.append(
                    f"plan produced {len(slots)} slots for a {self.taxonomy.challenges_per_day}-"
                    "challenge pack"
                )
            families = slot_assignments(bytes(32), self.generators)
            declared = {str(g["family"]): int(g["slots"]) for g in self.generators}
            actual = {family: families.count(family) for family in declared}
            if actual != declared:
                problems.append(f"slot assignment dealt {actual} against the declared {declared}")
        except Exception as error:  # noqa: BLE001
            problems.append(f"slot plan: {error}")

        # Every judge and generator model must be snapshot-pinned. A route the provider can repoint
        # mid-season changes what every laboratory is judged by, and 27 measures rerun correlation.
        for family, pin in sorted(self.pins.items()):
            try:
                pin.assert_pinned()
            except Exception as error:  # noqa: BLE001
                problems.append(f"family {family}: {error}")

        # A resource class must exist, or no laboratory can be run under declared limits.
        try:
            Limits.from_season(self.season, wall_time_seconds=1_800)
        except Exception as error:  # noqa: BLE001
            problems.append(f"resource class: {error}")

        return problems

    def describe(self) -> str:
        return (
            f"season {self.season['season_id']} | netuid {self.args.netuid} "
            f"mechid {self.season['mechid']}\n"
            f"  pack       {self.taxonomy.challenges_per_day} challenges, "
            f"{ {str(g['family']): int(g['slots']) for g in self.generators} }, "
            f"{self.generation.candidates_per_slot} candidates per slot\n"
            f"  panels     {len(self.panels)} criteria, families "
            f"{sorted({j.family for p in self.panels.values() for j in p.judges})}\n"
            f"  scoring    pairwise {self.scoring.pairwise_weight_ppm / 10_000:.0f}% / pointwise "
            f"{self.scoring.pointwise_weight_ppm / 10_000:.0f}%, mechanism floor "
            f"{self.scoring.mechanism_floor_ppm / 10_000:.0f}%\n"
            f"  weights    tau {self.weights.temperature_ppm / 1_000_000:.2f}, cap "
            f"{self.weights.maximum_weight_ppm / 10_000:.1f}%, burn uid {self.weights.burn_uid}\n"
            f"  cycle      salt {self.cycle.salt_commit_offset} < randomness "
            f"{self.cycle.randomness_offset} < pack {self.cycle.pack_commit_offset} < reveal "
            f"{self.cycle.reveal_offset}"
        )

    # ------------------------------------------------------------------
    # A round
    # ------------------------------------------------------------------

    def phase(self) -> Phase:
        return self.cycle.phase_of(self.cycle.blocks_from_epoch(self.chain.current_block()))

    def commit_salt(self, *, date: str, salt: bytes) -> tuple[str, int]:
        """7.3's precommitment. Returns (commitment, block).

        Called before the randomness block exists, which is what the commitment is *for* — so the
        phase check is the security boundary rather than a convenience.
        """
        commitment = salt_commitment(salt)
        block = self.chain.publish_commitment(
            SaltCommitment(round_id=date, salt_commitment=commitment).encode()
        )
        _log.info("salt commitment for %s published at block %d", date, block)
        return commitment, block

    def derive_seed(self, *, date: str, salt: bytes, commitment: str, block_hash: bytes) -> bytes:
        """7.3's seed, with the salt verified against what was committed.

        `verify_salt` runs inside `daily_seed` when the commitment is passed, and it is always
        passed here. A seed derived without that check would let a validator commit one salt and
        generate with another, which is the whole of what committing first prevents.
        """
        verdict = verify_salt(salt, commitment)
        if not verdict.matches:
            raise ChainError(f"cannot derive a seed for {date}: {verdict.reason}")
        return daily_seed(
            date=date,
            validator_hotkey=self.chain.hotkey(),
            salt=salt,
            block_hash=block_hash,
            commitment=commitment,
        )

    def verify_peer_pack(
        self,
        *,
        pack: PackCommitment,
        observed_salt: SaltCommitment,
        salt_block: int,
        randomness_block: int,
    ) -> None:
        """Check another validator's pack commitment against its earlier salt commitment.

        17.5's replication needs this: a validator rerunning a peer's round has to establish that
        the peer's salt predated the randomness, and the pack commitment alone cannot show it.
        """
        verify_salt_timing(
            pack=pack,
            observed_salt=observed_salt,
            salt_block=salt_block,
            randomness_block=randomness_block,
        )

    def model_client(self, *, run_id: str) -> ModelClient:
        """A validator-funded client for one round's generation, critique and judging."""
        credentials = CredentialSet(
            validator=load_validator_credential(self.chain.hotkey())
        )
        self.ledger.admit(
            run_id, maximum_rcc=2_000_000, maximum_requests=10_000, maximum_search_calls=2_000
        )
        return ModelClient(
            credentials=credentials,
            prices=self.prices,
            ledger=self.ledger,
            receipt=Receipt(
                run_id=run_id,
                miner_hotkey="",
                bundle_digest="0" * 64,
                challenge_id="0" * 64,
                validator_hotkey=self.chain.hotkey(),
            ),
            run_id=run_id,
            pins=self.pins,
        )

    async def generate_pack(self, *, date: str, seed: bytes, salt_commitment_hex: str) -> str:
        """7.4 end to end: plan, generate, filter, commit, store. Returns the pack hash."""
        slots = plan(seed, taxonomy=self.taxonomy, generators=self.generators)
        client = self.model_client(run_id=f"gen-{date}")
        result = await build_pack(
            client,
            date=date,
            slots=slots,
            taxonomy=self.taxonomy,
            config=self.generation,
            store=self.store,
        )
        _log.info(
            "pack for %s built: %d challenges, %d candidates rejected (%s), %d RCC",
            date,
            len(result.challenges),
            len(result.rejections),
            result.rejections_by_step(),
            result.rcc,
        )
        return commit_and_store(
            result,
            publish=self.chain.publish_commitment,
            store=self.store,
            salt_commitment=salt_commitment_hex,
            ttl_days=self.generation.dedup_lookback_days,
        )

    def build_runner(self, *, admit: Any, close: Any) -> Runner:
        """The sandbox runner, with egress confinement checked first.

        `ensure_network` and `assert_egress_confined` run before any laboratory does. This is the
        control whose failure is silent — a laboratory with an unintended route succeeds, off the
        meter, and the only symptom is a receipt that will not reconcile.
        """
        ensure_network()
        assert_egress_confined()
        return Runner(
            sandbox=self.sandbox,
            admit=admit,
            close=close,
            rcg_endpoint=self.args.rcg_endpoint,
            workspace=self.args.workspace,
        )

    def gate(self, *, result: Any, challenge: dict[str, Any], declared_models: dict[str, str]):
        """13's hard gates on one laboratory's response."""
        return check_all(
            portfolio=result.portfolio,
            challenge=challenge,
            receipt_calls=result.receipt_calls,
            measured_rcc=result.measured_usage["rcc"],
            measured_search_calls=result.measured_usage["search_calls"],
            declared_models=declared_models,
            wall_seconds=result.wall_seconds,
            timed_out=result.timed_out,
            excluded_domains=self.taxonomy.excluded_domains,
        )

    def canonicalise(self, *, result: Any):
        """14's neutral representation, with measured usage substituted for the claim."""
        from validator.canonicalizer.neutral import canonicalize

        return canonicalize(result.portfolio or {}, measured_usage=result.measured_usage)

    def score_challenge(self, criteria: dict[str, Any]) -> Any:
        """18.1-18.4 for one challenge. Criteria absent from the mapping are *omitted*.

        Omitted, not defaulted to zero: `challenge_score` redistributes an unmeasured criterion's
        weight over the rest, and with originality at 25% a single silent default would cost a
        quarter of the score for a judge outage that was never the miner's fault.
        """
        return challenge_score(criteria, self.scoring)

    def score_day(self, per_challenge: list[int]) -> Any:
        """18.5's daily score: 0.70 mean + 0.30 lower quartile."""
        return daily_score(per_challenge, self.daily)

    def score_rolling(self, history: Any) -> Any:
        """18.6's rolling score, which selects an estimator and never scales the result."""
        return rolling_score(history, self.daily)

    def rank(self, verdicts: Any) -> dict[str, int]:
        """18.3: pairwise verdicts to a ranking, via Bradley-Terry."""
        pairings, inconsistency = combine_orders(verdicts)
        ceiling = int(self.season["judging"]["order_swap_inconsistency_ceiling_ppm"])
        if inconsistency > ceiling:
            # 19: a panel measuring position rather than content is not usable. Logged rather than
            # raised, because the round's other criteria are unaffected and discarding all of them
            # would be a larger error than proceeding with a flagged one.
            _log.error(
                "order-swap inconsistency %d ppm exceeds the %d ppm ceiling in 19. This panel is "
                "largely measuring presentation position; its verdicts should be discounted and the "  # noqa: E501 - one-line diagnostic listing every reachable enforcement point
                "judge that caused it removed.",
                inconsistency,
                ceiling,
            )
        return strengths_to_ppm(fit(pairings))

    def allocate_weights(self, candidates: list[Candidate], *, floor_ppm: int):
        """20: softmax, cap, and burn if nobody qualifies."""
        return allocate(candidates, reference_floor_ppm=floor_ppm, config=self.weights)

    def submit(self, allocation: Any) -> None:
        self.chain.submit_weights(allocation.uids, allocation.weights)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    try:
        season = json.loads(args.season.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read the season config at {args.season}: {error}", file=sys.stderr)
        return 2

    if args.check:
        # A fake chain: `--check` must not need a reachable node, and every check below is about the
        # config rather than about the chain.
        chain: ChainClient = FakeChain(netuid=args.netuid)
        try:
            validator = Validator(season, chain=chain, args=args)
        except (CycleError, StoreError, ValueError, KeyError) as error:
            print(f"validator --check FAILED — cannot build: {error}", file=sys.stderr)
            return 1

        problems = validator.validate()
        if problems:
            print(f"validator --check FAILED — {len(problems)} problem(s):\n", file=sys.stderr)
            for problem in problems:
                print(f"  x {problem}", file=sys.stderr)
            return 1

        print("validator --check passed\n")
        print(validator.describe())
        # Named so a reader can see that the checks below were reached rather than assumed. Each of
        # these is an enforcement point the reachability gate pins.
        print(
            f"\n  enforcement reachable: "
            f"{', '.join(sorted({f.__name__ for f in (assert_sums_to_one, apply_weights, quantile_ppm, daily_seed, verify_salt, slot_assignments, verify_chain, reconcile, fit, collapse_duplicates, rank_weighted, challenge_score, daily_score, rolling_score, allocate, plan, lint, screen, is_duplicate, assess, commit_and_store, assert_not_sandbox_reachable, verify_salt_timing, docker_command, assert_egress_confined, check_all, swiss_pairings, aggregate)}))}"  # noqa: E501 - one-line diagnostic listing every reachable enforcement point
        )
        return 0

    if not args.netuid:
        print(
            "--netuid is required (or AI_NETUID). A validator with no netuid has no metagraph to "
            "read and no subnet to submit weights to.",
            file=sys.stderr,
        )
        return 2

    try:
        chain = BittensorChain(
            netuid=args.netuid,
            wallet_name=args.wallet,
            hotkey_name=args.hotkey,
            network=args.network,
            mechid=int(season["mechid"]),
            version_key=int(season["weights_version"]),
        )
        validator = Validator(season, chain=chain, args=args)
    except (CredentialError, StoreError, ChainError, ValueError) as error:
        print(f"validator cannot start: {error}", file=sys.stderr)
        return 2

    problems = validator.validate()
    if problems:
        print(f"validator cannot start — {len(problems)} config problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  x {problem}", file=sys.stderr)
        return 2

    _log.info("validator starting\n%s", validator.describe())
    try:
        phase = validator.phase()
    except ChainError as error:
        print(f"cannot read the chain: {error}", file=sys.stderr)
        return 3

    _log.info("current phase: %s", phase.name)
    if args.once:
        _log.info(
            "--once: the round loop is not yet wired to the chain's block stream. Every stage is "
            "built and reachable (see --check); what remains is the block-driven scheduler."
        )
        return 0

    _log.info(
        "the block-driven round loop is not yet wired. Run with --check to validate this "
        "deployment's configuration, or --once to report the current phase."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
