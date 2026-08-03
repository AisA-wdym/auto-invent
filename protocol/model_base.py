"""Base class for the generated protocol models.

Exists for one reason: pydantic reserves the `model_` prefix for its own attributes, and
three protocol fields legitimately start with it — `model_slug`, `model_snapshot` and
`model_calls`, all named by architecture.md sections 5.3 and 9.2. Left alone, importing the
models emits eight warnings, and eight warnings on every import is how a real warning goes
unread.

Renaming the fields is not an option: they are the wire format, and the schema is the
contract. So the reservation is lifted here, in one place, rather than suppressed at each
call site or patched into generated code that the next regeneration would overwrite.

`extra="forbid"` is *not* set here. The generator emits it per model from each schema's own
`additionalProperties: false`, and setting it in the base as well would mean a schema that
one day permits extras would still refuse them — the base would silently override the
contract. The schema decides; this class only lifts a pydantic-internal reservation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["ProtocolModel"]


class ProtocolModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
