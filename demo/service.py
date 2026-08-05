"""The demo service: a visitor's own problem, run through the reference laboratory.

Every page on the dashboard describes the subnet. This is the one that *does* it — a problem posed
the way a validator poses one, answered the way a miner answers, on the owner's money.

It runs here rather than on the dashboard's host because a run needs Docker, the sandbox and a
metering gateway. The browser never reaches this service: the dashboard calls it server-to-server
with a shared secret, so no certificate is needed here and no public surface beyond one port.

## The three things that make this safe to point at the internet

**A problem from a stranger is stranger input.** It goes through the same safety screen a
generated candidate does, before anything runs. Skipping it because the source is a web form rather
than a model would be exactly backwards — 13.12's domain exclusions exist for input nobody vetted,
and a visitor is less vetted than a generator this validator configured.

**Spend is bounded twice, and one bound is not this code.** Each run gets a freshly minted key
capped at `maximum_run_usd` with an hour's expiry, revoked when the run ends — so the ceiling is
enforced by the provider and a bug here cannot exceed it. A daily total is tracked across runs too,
because a per-run cap alone bounds one visitor, not a thousand.

**Concurrency is bounded rather than serialised.** The validator itself runs four containers at
once on this host, and queueing every visitor behind one run would take half an hour to answer
three people. What needs bounding is the host's CPU and memory, so the bound is a semaphore sized
like the validator's, and requests beyond it wait.

## What it deliberately is not

It is not scored, not judged, and not part of any round. There is no floor to beat and no weight
at stake — it is the laboratory answering a question, which is the part of the subnet a reader
cannot otherwise see. The response says so, because a portfolio that looked scored would
misrepresent both.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DemoConfig", "DemoService", "build_app"]

_log = logging.getLogger(__name__)

#: Ceiling on the problem text a visitor may submit. Generous for a research problem and far below
#: what would make a prompt expensive: the laboratory's own input ceiling is what actually bounds
#: cost, and this stops a megabyte of text reaching the linter at all.
MAXIMUM_PROBLEM_CHARS = 6_000


@dataclass
class DemoConfig:
    """What the service needs. Every bound is here rather than inline, so it can be read at once."""

    #: The owner's management key. Cannot spend — it mints per-run keys that can. See
    #: `gateway/provisioning.py` for why that asymmetry is the whole reason this is fundable.
    management_key: str = field(default="", repr=False)
    #: Shared secret the dashboard presents. Not a user credential: it authenticates the *caller*,
    #: and the caller is one service.
    caller_secret: str = field(default="", repr=False)
    #: Dollars per run, enforced by the provider on a minted key.
    maximum_run_usd: float = 0.50
    #: Dollars per day across every run. A per-run cap bounds one visitor; this bounds a thousand.
    maximum_daily_usd: float = 20.0
    #: Concurrent runs. Sized like the validator's own `--concurrency`, because they share a host
    #: and
    #: what needs bounding is CPU and memory rather than requests.
    concurrency: int = 3
    #: Beyond this many waiting, a request is refused rather than queued. A queue nobody will
    #: outlive
    #: is worse than a refusal that says how long to come back.
    maximum_queued: int = 12
    season: Path = Path("config/season.542.json")
    rcg_endpoint: str = "http://127.0.0.1:8081"
    runner_token: str = field(default="", repr=False)
    workspace: Path = Path("var/demo")
    #: The image the demo laboratory runs. The reference template, built locally — not a miner's
    #: bundle, so nothing here depends on a submission being fetchable or a round having happened.
    image: str = "my-lab:dev"

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> DemoConfig:
        env = dict(os.environ if environ is None else environ)
        config = cls(
            management_key=env.get("AI_OWNER_MANAGEMENT_KEY", "").strip(),
            caller_secret=env.get("AI_DEMO_SECRET", "").strip(),
            maximum_run_usd=float(env.get("AI_DEMO_RUN_USD", "0.50")),
            maximum_daily_usd=float(env.get("AI_DEMO_DAILY_USD", "20")),
            concurrency=int(env.get("AI_DEMO_CONCURRENCY", "3")),
            season=Path(env.get("AI_SEASON", "config/season.542.json")),
            rcg_endpoint=env.get("AI_RCG_ENDPOINT", "http://127.0.0.1:8081"),
            runner_token=env.get("AI_RUNNER_TOKEN", "").strip(),
            image=env.get("AI_DEMO_IMAGE", "my-lab:dev"),
        )
        missing = [
            name
            for name, value in (
                ("AI_OWNER_MANAGEMENT_KEY", config.management_key),
                ("AI_DEMO_SECRET", config.caller_secret),
                ("AI_RUNNER_TOKEN", config.runner_token),
            )
            if not value
        ]
        if missing:
            # Refused at construction. A demo service with no caller secret is an open endpoint that
            # spends the owner's money, and one with no management key fails on the first request
            # after a visitor has already waited.
            raise ValueError(
                f"the demo service needs {', '.join(missing)}. Without a caller secret it is an "
                "open endpoint spending the owner's account; without a management key it cannot "
                "mint a capped key and would either spend uncapped or fail after the visitor waits."
            )
        return config


@dataclass
class Spend:
    """What the service has spent today, and whether it may spend more.

    In memory, and that is a real limitation: a restart forgets the day's total. Stated rather than
    hidden because the *provider-side* cap is what actually bounds a single run — this is the second
    bound, and a second bound that resets on restart is still worth having against a slow drain.
    """

    maximum_daily_usd: float
    day: str = ""
    spent_usd: float = 0.0

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def remaining(self) -> float:
        if self.day != self._today():
            self.day, self.spent_usd = self._today(), 0.0
        return max(0.0, self.maximum_daily_usd - self.spent_usd)

    def record(self, usd: float) -> None:
        self.remaining()
        self.spent_usd += max(0.0, usd)


class DemoError(RuntimeError):
    """A demo request that will not run, with a reason a visitor can read."""


@dataclass
class DemoService:
    """Runs one visitor problem through the reference laboratory."""

    config: DemoConfig
    spend: Spend = field(init=False)
    _slots: asyncio.Semaphore = field(init=False)
    _waiting: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.spend = Spend(maximum_daily_usd=self.config.maximum_daily_usd)
        self._slots = asyncio.Semaphore(max(1, self.config.concurrency))

    def authenticate(self, presented: str) -> None:
        """Constant-time comparison of the caller's secret.

        `compare_digest` because this guards an endpoint that spends money, and a caller who could
        time the comparison could recover the secret one byte at a time.
        """
        if not presented or not hmac.compare_digest(presented, self.config.caller_secret):
            raise DemoError("not the dashboard")

    def as_challenge(self, body: dict[str, Any]) -> dict[str, Any]:
        """Turn a visitor's problem into the challenge shape a laboratory reads.

        Same shape a generated challenge has, so the laboratory cannot tell the difference — which
        is the point: what runs here is the subnet's own loop, not a path with different rules.
        """
        title = str(body.get("title", "")).strip()
        statement = str(body.get("problem_statement", "")).strip()
        objective = str(body.get("research_objective", "")).strip()
        if not title or not statement:
            raise DemoError(
                "a problem needs a title and a statement of what is actually hard about it. A "
                "laboratory asked for five inventions against an empty problem returns five "
                "generalities."
            )
        if len(statement) + len(objective) > MAXIMUM_PROBLEM_CHARS:
            raise DemoError(
                f"the problem is longer than {MAXIMUM_PROBLEM_CHARS} characters. A problem "
                "needing more than that is several problems, and a laboratory given several "
                "answers none of them well."
            )

        constraints = [
            str(entry).strip()
            for entry in (body.get("constraints") or [])
            if str(entry).strip()
        ][:8]
        season = self._season()
        pricing = season["providers"]["miner_pricing"]
        return {
            "challenge_id": f"demo:{secrets.token_hex(16)}",
            "domain": str(body.get("domain", "software_architecture")),
            "title": title[:200],
            "problem_statement": statement,
            "research_objective": objective or f"Make progress on: {title[:120]}",
            "current_baseline": str(body.get("current_baseline", "")).strip(),
            "known_attempts": [],
            "constraints": constraints,
            "forbidden_shortcuts": [],
            "required_output": {
                "portfolio_size": 5,
                "required_fields": [
                    "mechanism",
                    "why_non_obvious",
                    "nearest_prior_art",
                    "cheapest_kill_test",
                ],
            },
            "resource_limits": {
                # A demo run is bounded harder than a round: the owner pays, and nobody is being
                # scored, so there is no fairness argument for the full ceiling.
                "maximum_rcc": int(pricing["maximum_rcc"]) // 2,
                "maximum_search_calls": 20,
                "maximum_wall_time_seconds": 900,
            },
        }

    def screen(self, challenge: dict[str, Any]) -> None:
        """The linter and the safety screen, before anything runs.

        A visitor is *less* vetted than a generator this validator configured, so running the same
        checks on their input is the minimum rather than an extra. The safety screen is the one that
        matters most: 6.3 publishes nothing from here, but the laboratory's output is returned to a
        browser and the excluded domains exist because a published wrong answer causes harm outside
        the subnet.
        """
        from validator.challenge_factory.safety import screen as safety_screen
        from validator.challenge_factory.taxonomy import Taxonomy

        taxonomy = Taxonomy.from_season(self._season())
        verdict = safety_screen(challenge, excluded_domains=taxonomy.excluded_domains)
        if not verdict.safe:
            raise DemoError(
                f"this problem is in a domain the subnet excludes ({verdict.excluded_domain}). "
                "The exclusions are not about difficulty — they are domains where a confidently "
                "wrong answer causes harm outside the subnet."
            )

    async def run(self, body: dict[str, Any]) -> dict[str, Any]:
        """Screen, queue, mint, run, revoke, return."""
        challenge = self.as_challenge(body)
        self.screen(challenge)

        if self.spend.remaining() < self.config.maximum_run_usd:
            raise DemoError(
                f"the demo's daily budget is spent (${self.config.maximum_daily_usd:.0f}). It "
                "resets at midnight UTC. The subnet's own rounds are unaffected — this is a "
                "separate allowance so a demo cannot starve a round."
            )
        if self._waiting >= self.config.maximum_queued:
            raise DemoError(
                f"{self._waiting} runs are already waiting and each takes several minutes. A queue "
                "nobody will outlive is worse than being told to come back."
            )

        # Counted exactly once, on the way in and on the way out. Decrementing both inside the
        # semaphore and in `finally` would take the counter below zero on every completed run, and
        # the queue ceiling would drift upward with each request.
        self._waiting += 1
        try:
            async with self._slots:
                self._waiting -= 1
                try:
                    return await self._execute(challenge)
                finally:
                    self._waiting += 1
        finally:
            self._waiting = max(0, self._waiting - 1)

    async def _execute(self, challenge: dict[str, Any]) -> dict[str, Any]:
        from gateway.client import GatewayClient
        from gateway.provisioning import mint_round_key, read_usage, revoke
        from validator.sandbox.container import Limits, SandboxRunner, ensure_network
        from validator.sandbox.runner import Runner

        season = self._season()
        pricing = season["providers"]["miner_pricing"]
        minted = mint_round_key(
            self.config.management_key,
            name=f"auto-invent-demo-{challenge['challenge_id'][-8:]}",
            limit_usd=self.config.maximum_run_usd,
            lifetime_hours=1.0,
        )
        started = time.monotonic()
        try:
            ensure_network()
            client = GatewayClient(
                endpoint=self.config.rcg_endpoint, runner_token=self.config.runner_token
            )
            # Two endpoints, and conflating them is a defect this repository has now had twice.
            # `GatewayClient` runs in *this* process and reaches the gateway on localhost. The
            # `rcg_endpoint` handed to `Runner` is written into the container's environment, and
            # inside a container `127.0.0.1` is the container — so the laboratory gets the host's
            # address on the sandbox bridge. The miner rehearsal harness hit this first; the
            # symptom both times was `Connection refused` from inside the sandbox, two seconds in.
            runner = Runner(
                sandbox=SandboxRunner(),
                admit=client.admit,
                close=client.close,
                rcg_endpoint=_bridge_endpoint(self.config.rcg_endpoint),
                workspace=self.config.workspace,
            )
            deadline = int(time.time()) + int(
                challenge["resource_limits"]["maximum_wall_time_seconds"]
            )
            result = await runner.execute(
                run_id=challenge["challenge_id"].replace(":", "-"),
                miner_hotkey="demo",
                bundle_digest="demo",
                image_digest=_image_digest(self.config.image),
                validator_hotkey="demo",
                challenge=challenge,
                api_key=minted.secret,
                allowed_models=[str(slug) for slug in pricing["allowed_model_slugs"]],
                limits=Limits.from_season(season, wall_time_seconds=900),
                deadline=str(deadline),
                expires_at=deadline,
                episode_deadline=deadline,
            )
        finally:
            try:
                usage = read_usage(self.config.management_key, minted.key_hash)
                self.spend.record(usage["usage_usd"])
            except Exception as error:  # noqa: BLE001 - accounting must not fail a completed run
                # Charged at the cap rather than at zero. An unreadable usage figure is not evidence
                # of no spend, and assuming zero is how a daily ceiling stops bounding anything.
                _log.error("cannot read demo usage (%s); charging the run at its cap", error)
                self.spend.record(self.config.maximum_run_usd)
            revoke(self.config.management_key, minted.key_hash)

        if result.portfolio is None:
            # The container's own stderr is included, so the message names the cause rather than
            # only the symptom.
            tail = (result.stderr_tail or "").strip().splitlines()
            because = tail[-1][:300] if tail else "the container wrote nothing to stderr"
            _log.error(
                "demo run produced no portfolio in %.0fs (exit %s): %s",
                result.wall_seconds,
                result.exit_code,
                because,
            )
            raise DemoError(
                "the laboratory did not return a readable portfolio: "
                f"{result.failure or 'no output'}. The container said: {because}"
            )
        return {
            "challenge": {
                key: challenge[key]
                for key in ("title", "problem_statement", "research_objective", "constraints")
            },
            "portfolio": result.portfolio.get("portfolio", []),
            "run": {
                "wall_seconds": round(time.monotonic() - started, 1),
                "rcc": int(result.measured_usage.get("rcc", 0)),
                "search_calls": int(result.measured_usage.get("search_calls", 0)),
                "model_calls": len(result.receipt_calls),
            },
            # Said in the payload, not only in the page: this is the laboratory answering a
            # question,
            # not a scored result. A portfolio that looked scored would misrepresent both.
            "scored": False,
            "note": (
                "Run on the reference laboratory, funded by the subnet owner. Not judged, not "
                "scored, and not part of any round — the rounds are where laboratories compete."
            ),
        }

    def _season(self) -> dict[str, Any]:
        from protocol.season import load_season

        return load_season(self.config.season)


def _bridge_endpoint(local_endpoint: str) -> str:
    """The gateway's address as a container on the sandbox bridge sees it.

    The port is kept from the local endpoint and only the host is replaced, so an operator who moved
    the gateway's port does not also have to know about this translation.
    """
    from urllib.parse import urlparse

    from miner.cli.rehearse import _bridge_address

    port = urlparse(local_endpoint).port or 8081
    return f"http://{_bridge_address()}:{port}"


def _image_digest(image: str) -> str:
    import subprocess

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise DemoError(
            f"the demo image {image!r} is not built on this host. The sandbox runs by digest, so "
            "there is nothing to run without one."
        )
    return result.stdout.strip()


def build_app(service: DemoService) -> Any:
    """Two routes: run a problem, and report health. Nothing else is exposed."""
    from fastapi import Body, FastAPI, Header, HTTPException

    app = FastAPI(title="auto-invent demo", description=__doc__, version="AIL-3.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "concurrency": service.config.concurrency,
            "waiting": service._waiting,
            "daily_remaining_usd": round(service.spend.remaining(), 2),
        }

    @app.post("/run")
    async def run(
        body: dict[str, Any] = Body(...),
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        _, _, presented = authorization.partition(" ")
        try:
            service.authenticate(presented)
        except DemoError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        try:
            return await service.run(body)
        except DemoError as error:
            # 400 rather than 500: every `DemoError` is a statement about the request or the
            # allowance, and a visitor reading it should be able to act on it.
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app
