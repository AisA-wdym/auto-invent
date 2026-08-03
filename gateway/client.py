"""The validator's side of the runner-authenticated gateway routes.

`Runner` takes `admit` and `close` as callables so a run can be driven against a real gateway over
HTTP or an in-process one in a test, without the runner knowing which. This is the real one.

## Only two routes, and never the ones a laboratory uses

`/v1/runs` and `/v1/runs/{id}/close` are the runner-authenticated pair. `/v1/llm` and `/v1/search`
are the laboratory's, reached with a session token from inside the sandbox, and nothing in the
validator should ever call them: a validator that could spend through a laboratory's session is a
validator that can exhaust a rival-sponsored miner at will. So this client does not implement them —
not as a policy that could be relaxed, but as code that does not exist.

## The runner token is not the miner's credential

Admission carries the miner's API key *to* the gateway, and the gateway keeps it. What comes back is
a session token with ceilings and no key. The asymmetry is the point of the whole design, and it is
why `admit` is the only call in this repository that transports a provider credential.

## Failures are loud

An admission that fails means a run that never started, and a close that fails means totals nobody
can reconcile. Both raise. A client that returned an empty token on a connection error would produce
a container that starts, fails every gateway call, and is scored as a laboratory that produced
nothing — indistinguishable from a miner whose code is broken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["GatewayClient", "GatewayClientError"]

_log = logging.getLogger(__name__)


class GatewayClientError(RuntimeError):
    """The gateway could not be reached, or refused."""


@dataclass(frozen=True, slots=True)
class GatewayClient:
    """Calls the two runner-authenticated routes on the RCG."""

    endpoint: str
    runner_token: str
    timeout: float = 60.0

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.runner_token}"}

    async def admit(self, body: dict[str, Any]) -> str:
        """Open a run's ledger and return its session token."""
        import httpx

        url = f"{self.endpoint.rstrip('/')}/v1/runs"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=body, headers=self._headers())
        except httpx.HTTPError as error:
            raise GatewayClientError(
                f"cannot reach the gateway at {url}: {error}. Nothing can run without it — the "
                "sandbox has no other peer."
            ) from error

        if response.status_code != 200:
            raise GatewayClientError(
                f"the gateway refused admission for {body.get('run_id')!r}: HTTP "
                f"{response.status_code} {_body_excerpt(response)}"
            )
        token = response.json().get("session_token", "")
        if not token:
            raise GatewayClientError(
                f"the gateway admitted {body.get('run_id')!r} without a session token. An empty "
                "token would produce a container that starts, fails every call, and is scored as a "
                "laboratory that produced nothing."
            )
        return str(token)

    async def close(self, run_id: str) -> dict[str, Any]:
        """Close a run and return its measured totals and receipt chain head."""
        import httpx

        url = f"{self.endpoint.rstrip('/')}/v1/runs/{run_id}/close"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=self._headers())
        except httpx.HTTPError as error:
            raise GatewayClientError(
                f"cannot close {run_id} at {url}: {error}. The run's ledger entry and its receipt "
                "stay open, and its reservations count as leaked spend at shutdown."
            ) from error

        if response.status_code != 200:
            raise GatewayClientError(
                f"the gateway refused to close {run_id}: HTTP {response.status_code} "
                f"{_body_excerpt(response)}"
            )
        return dict(response.json())


def _body_excerpt(response: Any) -> str:
    """A short, safe excerpt of an error body.

    Truncated because a gateway error body can echo a request, and a request to `/v1/runs` carries a
    provider credential. Four hundred characters is enough to identify a validation error and not
    enough to carry a key that appears late in a long body — and the field is redacted outright
    below, because "not enough" is not a guarantee.
    """
    try:
        text = response.text
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return "<unreadable body>"
    for marker in ("api_key", "sk-or-", "sk-ant-"):
        if marker in text:
            return "<body withheld: it echoes credential material>"
    return text[:400]
