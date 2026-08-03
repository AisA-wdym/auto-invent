#!/usr/bin/env python3
"""Static check: does every enforcement point have a *production* call path?

The failure this exists for: a module implements a rule, a test calls it directly and passes,
and nothing else calls it at all. Under `pytest` that is indistinguishable from working code.
At runtime it is indistinguishable from code that was never written.

`tests/` is deliberately not parsed. A test calling a guard is not a call path — it is the
thing this gate exists to discount.

## Two checks, and the second one switches on

**Existence** runs always. Every symbol the manifest names must exist. That is meaningful from
the first commit: it catches a rename, a move, or a deletion that would otherwise leave the
gate watching nothing.

**Reachability** runs once at least one declared entry point exists. Before that there is no
`main` to walk from, and failing on that would make the gate red for the whole build-out
period — which trains everyone to ignore it. So it reports "not yet checkable" and says how
many symbols are waiting, then becomes strict automatically the moment an entry point lands.

That is a deliberate compromise and worth naming: for now this gate protects against drift,
not against unreachability. The reachability half is the one that caught real defects in the
predecessor, so the entry points are worth building early.

## Edge resolution is by simple name, which over-approximates on purpose

Two methods called `verify` are conflated, so a symbol can be reported reachable when the real
edge belongs to the other one. The gate therefore never reports a *false* unreachability: what
it flags is genuinely called from nowhere. Missed findings are the cost; wrong findings would
make it unusable as a build gate.

Run:  python3 tools/reachability.py [--report]
Exit: 0 clean, 1 on any violation.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Packages that ship. `tests/` and `tools/` are excluded by construction.
SOURCE_PACKAGES = ("protocol", "registry", "gateway", "validator", "miner", "portal", "ops")

#: Process entry points. Anything unreachable from one of these does not run in production,
#: whatever the tests say.
ENTRY_POINTS: tuple[str, ...] = (
    "gateway.__main__:main",
    "validator.__main__:main",
    "miner.cli.main:main",
)

#: requirement -> the symbol that enforces it, by qualified name.
#:
#: The pair is checkable in both directions: the gate proves the symbol is reached, and a
#: reader can ask whether the symbol named is really what enforces the rule. A wildcard
#: ("everything under gateway/") would prove neither, and a rename would silently satisfy it.
#:
#: Entries are added as each rule is wired, never as a batch — a symbol that does not exist
#: is a hard failure, so the table cannot drift ahead of the code.
ENFORCEMENT: dict[str, str] = {
    "no float reaches a hashed object":
        "protocol.canonical:assert_no_floats",
    "one deterministic encoder for everything the subnet constructs":
        "protocol.canonical:canonical_bytes",
    "a weight set that does not sum to one whole is refused":
        "protocol.fixedpoint:assert_sums_to_one",
    "an unscoreable criterion has its weight redistributed, never scored zero":
        "protocol.fixedpoint:apply_weights",
    "18.5 the lower quartile penalises inconsistency":
        "protocol.fixedpoint:quantile_ppm",
    "7.3 the daily seed binds date, validator, precommitted salt and post-deadline randomness":
        "protocol.seeds:daily_seed",
    "a revealed salt is verified against its commitment":
        "protocol.seeds:verify_salt",
    "7.2.1 slot assignment is derived from the seed, never chosen":
        "protocol.seeds:slot_assignments",
    "13 the receipt chain is tamper-evident":
        "protocol.receipts:verify_chain",
    "27 receipt totals are reconciled against provider-reported usage":
        "protocol.receipts:reconcile",
    "3.4.4 a call is refused if billed to the wrong account":
        "protocol.receipts:Call.__post_init__",
}

#: Decorator shapes meaning "a framework calls this, not our code". `@app.post("/v1/llm")`
#: registers an endpoint that nothing in the source names.
_FRAMEWORK_REGISTRATION = (ast.Attribute,)


@dataclass
class Definition:
    qualname: str
    module: str
    lineno: int
    references: set[str] = field(default_factory=set)
    framework_entered: bool = False


def _identifiers(node: ast.AST) -> set[str]:
    """Every name the subtree mentions, however it is mentioned.

    Bare references count as well as calls, because dependency injection is this codebase's
    normal shape: `build(meter, now=now)` never *calls* `now` there, but passing it is what
    puts it on the call path.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
    return found


def _is_framework_entered(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, _FRAMEWORK_REGISTRATION):
            return True
    return False


def collect() -> tuple[dict[str, list[Definition]], dict[str, Definition]]:
    by_simple: dict[str, list[Definition]] = defaultdict(list)
    by_qual: dict[str, Definition] = {}

    for package in SOURCE_PACKAGES:
        base = ROOT / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            module = str(path.relative_to(ROOT)).removesuffix(".py").replace("/", ".")
            module = module.removesuffix(".__init__")
            tree = ast.parse(path.read_text(), filename=str(path))

            # Module-level statements run on import, so their references seed reachability.
            module_level = ast.Module(
                body=[
                    statement
                    for statement in tree.body
                    if not isinstance(
                        statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                    )
                ],
                type_ignores=[],
            )
            entry = Definition(qualname=module, module=module, lineno=0)
            entry.references = _identifiers(module_level)
            by_qual[module] = entry
            by_simple[module].append(entry)

            def visit(node: ast.AST, prefix: str, module: str = module) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        separator = "." if ":" in prefix else ":"
                        qualname = f"{prefix}{separator}{child.name}"
                        found = Definition(
                            qualname=qualname,
                            module=module,
                            lineno=child.lineno,
                            references=_identifiers(child),
                        )
                        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                            found.framework_entered = _is_framework_entered(child)
                        by_qual[qualname] = found
                        by_simple[child.name].append(found)
                        visit(child, qualname, module)

            visit(tree, module)

    return by_simple, by_qual


def reachable(
    by_simple: dict[str, list[Definition]], by_qual: dict[str, Definition]
) -> tuple[set[str], list[str]]:
    """Reached qualnames, plus any declared entry point that does not exist."""
    children: dict[str, list[Definition]] = defaultdict(list)
    for definition in by_qual.values():
        for separator in (":", "."):
            parent, found, _ = definition.qualname.rpartition(separator)
            if found and parent in by_qual:
                children[parent].append(definition)
                break

    frontier: list[Definition] = []
    missing: list[str] = []
    for entry in ENTRY_POINTS:
        if entry in by_qual:
            frontier.append(by_qual[entry])
        else:
            missing.append(entry)

    # Framework-registered handlers are entered from outside the package, so they are roots.
    frontier.extend(d for d in by_qual.values() if d.framework_entered)

    reached: set[str] = set()
    expanded: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current.qualname in reached:
            continue
        reached.add(current.qualname)
        frontier.extend(children.get(current.qualname, ()))
        for name in current.references:
            if name in expanded:
                continue
            expanded.add(name)
            frontier.extend(by_simple.get(name, ()))

    return reached, missing


def _resolve(symbol: str, by_qual: dict[str, Definition]) -> Definition | None:
    if symbol in by_qual:
        return by_qual[symbol]
    _module, _, name = symbol.partition(":")
    candidates = [d for q, d in by_qual.items() if q.endswith(":" + name) or q.endswith("." + name)]
    return candidates[0] if len(candidates) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="list unreached public definitions")
    args = parser.parse_args()

    by_simple, by_qual = collect()
    reached, missing_entries = reachable(by_simple, by_qual)
    entry_points_exist = len(missing_entries) < len(ENTRY_POINTS)

    failures: list[str] = []
    unreachable: list[str] = []

    for requirement, symbol in sorted(ENFORCEMENT.items()):
        definition = _resolve(symbol, by_qual)
        if definition is None:
            failures.append(f"{requirement}\n      no such symbol: {symbol}")
        elif entry_points_exist and definition.qualname not in reached:
            unreachable.append(
                f"{requirement}\n      {symbol} is defined but unreachable from any entry point"
                f"\n      defined at {definition.module.replace('.', '/')}.py:{definition.lineno}"
            )

    if args.report:
        candidates = sorted(
            d.qualname
            for d in by_qual.values()
            if d.qualname not in reached
            and ":" in d.qualname
            and not d.qualname.rpartition(":")[2].startswith("_")
        )
        print(f"# {len(candidates)} unreached public definitions\n")
        for qualname in candidates:
            print(f"  {qualname}")
        print()

    if failures or unreachable:
        total = len(failures) + len(unreachable)
        print(f"reachability gate FAILED — {total} enforcement point(s):\n")
        for failure in failures + unreachable:
            print(f"  x {failure}")
        print(
            "\nAn unreached guard is indistinguishable from an absent guard at runtime."
            "\nEither wire it into a composition root, or delete it and the requirement with it."
        )
        return 1

    if entry_points_exist:
        print(f"reachability gate passed — {len(ENFORCEMENT)} enforcement points on a call path")
    else:
        # Honest about what is and is not being checked. A gate that implied more than it
        # verified would be worse than one that says so.
        print(
            f"reachability: {len(ENFORCEMENT)} enforcement points pinned and present.\n"
            f"  Reachability not yet checkable: no entry point exists "
            f"({', '.join(ENTRY_POINTS)}).\n"
            "  This half of the gate switches on automatically when the first one lands."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
