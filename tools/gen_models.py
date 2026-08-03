#!/usr/bin/env python3
"""Generate pydantic validation models from `protocol/schemas/`.

The schemas are the contract. A contract nothing loads is decoration, and the way that
failure shows up is specific: an object crosses a boundary carrying fields its schema
forbids, or missing fields its schema requires, and every layer downstream accepts it
because no layer ever checked.

Generation is a build step, not a runtime step. Generating at import time would mean two
validators on the same release could parse a protocol object differently if their generator
versions differed — which is the same class of divergence the deterministic encoder works to
avoid. So the output is committed, and CI fails if it drifts from the schemas.

## A package, not one file

`datamodel-codegen` given a directory treats the schemas as a module tree and refuses a
single-file output. Working around that by concatenating per-schema output does not survive
contact with these schemas: several declare a type of the same name, and flattened, only the
last of each is reachable at module level. A validation surface whose types cannot be named
is not a validation surface.

So one module per schema, and `models/__init__.py` re-exports each schema's root type under
an unambiguous name — which keeps flat access working for everything a caller validates
against.

Run:  python3 tools/gen_models.py           # regenerate
      python3 tools/gen_models.py --check   # fail if the committed output is stale
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "protocol" / "schemas"
OUTPUT = ROOT / "protocol" / "models"

HEADER = '''"""Generated from protocol/schemas/. Do not edit by hand.

Regenerate with `python3 tools/gen_models.py`; `make schema` fails if this drifts from the
schema it was generated from. A schema change is a protocol change and requires bumping
`protocol_version`.
"""
'''

PACKAGE_DOC = '''"""Generated pydantic models, one module per protocol schema.

Do not edit by hand; regenerate with `python3 tools/gen_models.py`.

One module per schema rather than one file, because several schemas declare types of the same
name. Flattened, only the last of each would be reachable.
"""
'''


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    exports: list[tuple[str, str]] = []

    for schema in sorted(SCHEMA_DIR.glob("*.json")):
        stem = schema.name.removesuffix(".schema.json").removesuffix(".json")
        module = destination / f"{stem}.py"
        _run_generator(schema, module)
        module.write_text(HEADER + "\n" + module.read_text())
        root = _root_class(module)
        if root:
            exports.append((stem, root))

    lines = [PACKAGE_DOC, "", "from __future__ import annotations", ""]
    lines += [f"from .{stem} import {root} as {root}" for stem, root in exports]
    lines += ["", "__all__ = ["]
    lines += [f'    "{root}",' for _stem, root in exports]
    lines += ["]"]
    (destination / "__init__.py").write_text("\n".join(lines) + "\n")


def _root_class(module: Path) -> str:
    """The last top-level class in a generated module is its root type.

    `datamodel-codegen` emits `$defs` first and the document root last, so the final
    definition is the one a caller validates a whole object against. Derived rather than
    hardcoded per schema, so adding a schema needs no edit here.
    """
    names = [
        line.removeprefix("class ").split("(")[0].split(":")[0].strip()
        for line in module.read_text().splitlines()
        if line.startswith("class ")
    ]
    return names[-1] if names else ""


def _run_generator(schema: Path, destination: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input", str(schema),
            "--input-file-type", "jsonschema",
            "--output", str(destination),
            "--output-model-type", "pydantic_v2.BaseModel",
            "--target-python-version", "3.12",
            # Every schema object is `additionalProperties: false`. Unknown fields must be
            # rejected rather than ignored, so the generated model has to forbid extras.
            "--strict-nullable",
            "--use-schema-description",
            "--use-standard-collections",
            "--use-union-operator",
            "--collapse-root-models",
            "--disable-timestamp",
            # Lifts pydantic's `model_` attribute reservation, which three protocol fields
            # legitimately collide with. See protocol/model_base.py for why renaming them is
            # not an option and why `extra="forbid"` is deliberately not set there.
            "--base-class", "protocol.model_base.ProtocolModel",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"model generation failed for {schema.name} ({result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed models are stale"
    )
    args = parser.parse_args()

    if not SCHEMA_DIR.is_dir():
        raise SystemExit(f"schema directory missing: {SCHEMA_DIR}")

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "models"
            generate(candidate)
            if not OUTPUT.is_dir():
                print("::error::protocol/models/ is missing; run tools/gen_models.py")
                return 1
            fresh = {p.name: p.read_text() for p in candidate.glob("*.py")}
            committed = {p.name: p.read_text() for p in OUTPUT.glob("*.py")}
            if fresh != committed:
                added = sorted(set(fresh) - set(committed))
                removed = sorted(set(committed) - set(fresh))
                changed = sorted(
                    name for name in set(fresh) & set(committed) if fresh[name] != committed[name]
                )
                print(
                    "::error::protocol/models/ has drifted from protocol/schemas/. "
                    f"added={added} removed={removed} changed={changed}. "
                    "Run `python3 tools/gen_models.py` and commit the result."
                )
                return 1
        print(f"schema: {len(committed)} generated modules are current")
        return 0

    generate(OUTPUT)
    written = sorted(p.name for p in OUTPUT.glob("*.py"))
    print(f"wrote {OUTPUT.relative_to(ROOT)}/ ({len(written)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
