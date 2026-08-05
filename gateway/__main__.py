"""Gateway entry point: `python -m gateway`.

One of the three process roots. Its existence is also what switches on the reachability half of
`tools/reachability.py` — a guard nothing reaches from a `main` does not run in production,
whatever its tests say.

`--check` starts nothing and calls nothing external. It builds the whole object graph, asserts
every model in the season config is snapshot-pinned, and exits. That is the difference between
"a call path exists" and "the call path works": the reachability gate proves the first and this
proves the second.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path

from gateway.api import GatewayState, build_app
from gateway.credentials import CredentialError, CredentialSet, load_validator_credential
from gateway.metering import Ledger, PriceTable
from gateway.tokens import TokenIssuer
from protocol.season import SeasonError, load_season

_log = logging.getLogger("gateway")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gateway", description=__doc__)
    parser.add_argument("--season", default="config/season.example.json", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--hotkey",
        default=os.environ.get("AI_VALIDATOR_HOTKEY", ""),
        help="the validator hotkey this gateway serves; recorded on receipts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build everything, verify the season config, and exit without listening",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _build(season: dict, hotkey: str) -> GatewayState:
    """Assemble the gateway. The composition root.

    The runner secret is generated here and printed nowhere except to the runner's own
    environment. A secret in a config file would be committed eventually.
    """
    runner_secret = os.environ.get("AI_RUNNER_SECRET", "") or secrets.token_urlsafe(32)
    return GatewayState(
        credentials=CredentialSet(validator=load_validator_credential(hotkey)),
        prices=PriceTable.from_season(season),
        issuer=TokenIssuer(secret=secrets.token_bytes(32)),
        ledger=Ledger(),
        runner_secret=runner_secret,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    try:
        season = load_season(args.season)
    except SeasonError as error:
        # The gateway and the validator read the same file, so they must agree about whether it is
        # valid. Neither checking was worse than one of them checking.
        print(str(error), file=sys.stderr)
        return 2

    if args.check:
        # No credential is loaded and no socket is opened. What is checked is everything that
        # can be checked without either: the price table parses, every model in the season is
        # pinned, and the app composes.
        prices = PriceTable.from_season(season)
        unpinned = _unpinned_models(season)
        if unpinned:
            print(f"gateway --check FAILED — {len(unpinned)} unpinned model(s):")
            for name in unpinned:
                print(f"  x {name}")
            print(
                "\nAn unpinned route can be repointed by the provider mid-season, which changes "
                "what every laboratory runs without any manifest changing."
            )
            return 1
        state = GatewayState(
            credentials=CredentialSet(
                validator=_placeholder_validator(args.hotkey or "check-hotkey")
            ),
            prices=prices,
            issuer=TokenIssuer(secret=b"0" * 32),
            ledger=Ledger(),
            runner_secret="check",
        )
        app = build_app(state)
        routes = sorted(
            route.path for route in app.routes if getattr(route, "path", "").startswith("/")
        )
        print(
            f"gateway --check passed — season {season['season_id']}, "
            f"{prices.rcc_per_1k_in}/{prices.rcc_per_1k_out} RCC per 1k tokens, "
            f"{len(routes)} routes: {' '.join(routes)}"
        )
        return 0

    if not args.hotkey:
        print(
            "--hotkey (or AI_VALIDATOR_HOTKEY) is required: receipts record which validator "
            "ran a laboratory, and an unattributed receipt cannot be reconciled.",
            file=sys.stderr,
        )
        return 2

    try:
        state = _build(season, args.hotkey)
    except CredentialError as error:
        print(f"gateway cannot start: {error}", file=sys.stderr)
        return 2

    import uvicorn

    # Names the variable the *validator* reads, which is not the one this process reads. The
    # gateway verifies `AI_RUNNER_SECRET`; the validator presents the same value as
    # `AI_RUNNER_TOKEN`. This line used to say AI_RUNNER_SECRET, so an operator who followed it
    # exactly set the gateway's name on both sides, the validator's token stayed empty, and every
    # admission failed with an unsendable `Bearer ` header.
    _log.info(
        "runner secret for this process: %s (the validator presents this as AI_RUNNER_TOKEN)",
        state.runner_secret,
    )
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def _unpinned_models(season: dict) -> list[str]:
    """Every model in the season config whose snapshot is still a placeholder.

    Walks generators and judge panels rather than a single list, because a snapshot missing from
    a judge panel is just as consequential as one missing from a generator: a judge that moves
    mid-season rescores the same portfolio differently, and 27 measures rerun correlation.
    """
    unpinned: list[str] = []
    for generator in season["challenge_generation"]["generators"]:
        if not _pinned(generator["model_snapshot"]):
            unpinned.append(f"generator {generator['family']}: {generator['model_slug']}")
    for panel in season["judging"]["panels"]:
        for judge in panel["judges"]:
            if not _pinned(judge["model_snapshot"]):
                unpinned.append(
                    f"judge {panel['criterion']}/{judge['family']}: {judge['model_slug']}"
                )
    return unpinned


def _pinned(snapshot: str) -> bool:
    return bool(snapshot) and not snapshot.startswith("<")


def _placeholder_validator(hotkey: str):
    from gateway.credentials import ValidatorCredential

    return ValidatorCredential(validator_hotkey=hotkey, api_key="check-only-not-a-real-key")


if __name__ == "__main__":
    sys.exit(main())
