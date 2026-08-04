"""Loading a season config, validated against its own schema.

There was a schema, a generated model with `extra="forbid"`, a `make schema` gate proving the two
agree, and a test proving every *shipped* config validates. And nothing validated the config an
operator actually ran: `validator/__main__.py` and `gateway/__main__.py` each did `json.loads` and
indexed into the result.

So the whole apparatus was decorative for the only input that matters. Two ways that bites:

**A field that does nothing.** `netuid` is not in the schema — the subnet is chosen by `--netuid`.
An operator who writes `"netuid": 542` into the season config, which is the obvious place for it,
gets a validator that runs against netuid 0 and says nothing. That is how it was found.

**A field that is nearly right.** `maximum_rcc` mistyped once becomes a `KeyError` at whichever line
first reads it — possibly hours into a round, possibly in a branch that only runs on the day a
laboratory overspends. `extra="forbid"` catches it at startup, where an operator can fix it.

## Why this is a module and not two lines in each entry point

Because there were two entry points and they would have drifted. The gateway and the validator read
the same file and disagreeing about whether it is valid is worse than neither checking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SeasonError", "load_season"]


class SeasonError(ValueError):
    """A season config that cannot be read, parsed, or validated."""


def load_season(path: Path) -> dict[str, Any]:
    """Read and validate a season config, or raise with something an operator can act on.

    Returns the plain `dict`, not the model. Every consumer already indexes into a mapping, and
    handing back the model would mean rewriting all of them for no gain — validation is a gate here,
    not a representation. The gate is what was missing.
    """
    try:
        raw = path.read_text()
    except OSError as error:
        raise SeasonError(f"cannot read the season config at {path}: {error}") from error

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SeasonError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(body, dict):
        raise SeasonError(f"{path} holds a {type(body).__name__}, not an object")

    # Imported here rather than at module scope: `protocol/models/` is generated, and a module that
    # imported it eagerly would make a stale generation break every import in the package rather
    # than the one call that needs it.
    from pydantic import ValidationError

    from protocol.models.season_config import SeasonConfig

    try:
        SeasonConfig.model_validate(body)
    except ValidationError as error:
        raise SeasonError(_explain(path, error)) from error

    return body


def _explain(path: Path, error: Any) -> str:
    """Turn a pydantic error into something that names the field and says what to do.

    Pydantic's own rendering is accurate and hard to act on — it reports `extra_forbidden` at
    `providers.miner_pricing.allowed_model_slugs` without saying that the fix is to remove it. An
    operator reading this at 3am needs the field and the verb.
    """
    lines = [f"{path} is not a valid season config:"]
    for detail in error.errors()[:12]:
        where = ".".join(str(part) for part in detail.get("loc", ())) or "(root)"
        kind = detail.get("type", "")
        if kind == "extra_forbidden":
            lines.append(
                f"  x {where}: not a field this protocol version knows. Remove it — a field the "
                "schema does not define is a field nothing reads, and it will silently do nothing."
            )
        elif kind == "missing":
            lines.append(f"  x {where}: required and absent")
        else:
            lines.append(f"  x {where}: {detail.get('msg', kind)}")
    extra = len(error.errors()) - 12
    if extra > 0:
        lines.append(f"  … and {extra} more")
    return "\n".join(lines)
