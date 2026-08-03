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

**Reachability** becomes strict once *every* declared entry point exists. Until then an
unreached symbol is reported as `pending` rather than as a failure — because a guard the
validator reaches is genuinely unreachable while `validator.__main__` has not been written, and
that is a fact about the build's progress rather than a defect. Failing on it would keep the
gate red for the whole build-out, which trains everyone to ignore it.

The count is printed either way ("5 on a call path" out of 18), so the gate never implies more
than it verified, and no change here is needed when the last entry point lands: every pending
symbol becomes a hard failure automatically.

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
SOURCE_PACKAGES = (
    "protocol",
    "chain",
    "registry",
    "gateway",
    "validator",
    "miner",
    "portal",
    "ops",
)

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
    "18.3 pairwise verdicts become a ranking":
        "validator.judge.bradley_terry:fit",
    "18.1 duplicate ideas collapse to one lineage before scoring":
        "validator.scoring.criteria:collapse_duplicates",
    "18.1 the portfolio is rank-weighted 0.40/0.25/0.15/0.12/0.08":
        "validator.scoring.criteria:rank_weighted",
    "18.4 a weak mechanism caps value and originality":
        "validator.scoring.criteria:challenge_score",
    "18.5 the daily score weights the lower quartile at 30%":
        "validator.scoring.daily:daily_score",
    "18.6 the rolling score selects an estimator and never scales the result":
        "validator.scoring.daily:rolling_score",
    "20.1/20.4 a field entirely below the reference floor burns the emission":
        "validator.weights:allocate",
    "7.2 the pack is stratified across the taxonomy and split exactly between generators":
        "validator.challenge_factory.taxonomy:plan",
    "7.4.2 a candidate missing any of the eight requirements is rejected":
        "validator.challenge_factory.linter:lint",
    "2 a candidate in an excluded domain never enters a pack":
        "validator.challenge_factory.safety:screen",
    "7.4.4 a candidate duplicating the last 90 days is rejected":
        "validator.challenge_factory.dedup:is_duplicate",
    "7.4.5 a problem that does not discriminate between laboratories is rejected":
        "validator.challenge_factory.discriminator:assess",
    "7.5 the pack hash is committed on chain before the pack is stored":
        "validator.challenge_factory.pipeline:commit_and_store",
    "7.5 the challenge store is never reachable from the sandbox":
        "validator.challenge_factory.store:assert_not_sandbox_reachable",
    "7.3 a salt committed at or after the randomness block is refused":
        "protocol.commitments:verify_salt_timing",
    "10 a laboratory image is run by digest, never by a mutable tag":
        "validator.sandbox.container:docker_command",
    "10 the sandbox network is internal, so there is no route off the meter":
        "validator.sandbox.container:assert_egress_confined",
    "9.2 measured usage replaces the laboratory's self-reported claim":
        "validator.sandbox.runner:Runner.execute",
    "13 a hard-gate failure invalidates the response and cannot be compensated for":
        "validator.scoring.gates:check_all",
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


def _enclosing(qualname: str) -> str | None:
    """The definition that lexically encloses `qualname`, or None for a module.

    A qualname is `package.module:Outer.inner`. The module part and the within-module part are
    separated by the single `:`, and both use `.` internally — so a naive `rpartition` on either
    separator resolves the wrong parent. `mod:Call.__post_init__` rpartitioned on `:` yields the
    *module*, which is how `Call.__post_init__` came to be reported unreachable while `Call`
    itself was reached: the method was hung off the module rather than off its class, and module
    entries are not roots.

    That mattered because `__post_init__` has no textual call edge at all — the dataclass
    machinery invokes it — so being a child of its class is the *only* way it can be reached.
    Getting this wrong silently downgraded a real enforcement point to unverifiable.
    """
    module, colon, inner = qualname.partition(":")
    if not colon:
        return None
    outer, dot, _ = inner.rpartition(".")
    return f"{module}:{outer}" if dot else module


def reachable(
    by_simple: dict[str, list[Definition]], by_qual: dict[str, Definition]
) -> tuple[set[str], list[str]]:
    """Reached qualnames, plus any declared entry point that does not exist."""
    children: dict[str, list[Definition]] = defaultdict(list)
    for definition in by_qual.values():
        parent = _enclosing(definition.qualname)
        if parent is not None and parent in by_qual:
            children[parent].append(definition)

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
    # Strict only once *every* declared entry point exists. While one is still missing, a symbol
    # that only that process reaches is unreachable for a reason that is not a defect — the
    # scoring path is reached from the validator, so it is unreachable until `validator.__main__`
    # lands, and failing on that would keep the gate red for the whole build-out and train
    # everyone to ignore it. The moment the last entry point appears, every pending symbol
    # becomes a hard failure with no further change here.
    strict = not missing_entries

    failures: list[str] = []
    unreachable: list[str] = []
    pending: list[str] = []

    for requirement, symbol in sorted(ENFORCEMENT.items()):
        definition = _resolve(symbol, by_qual)
        if definition is None:
            failures.append(f"{requirement}\n      no such symbol: {symbol}")
        elif definition.qualname not in reached:
            where = f"{definition.module.replace('.', '/')}.py:{definition.lineno}"
            if strict:
                unreachable.append(
                    f"{requirement}\n      {symbol} is defined but unreachable from any entry "
                    f"point\n      defined at {where}"
                )
            else:
                pending.append(f"{symbol}  ({where})")

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

    if strict:
        print(f"reachability gate passed — {len(ENFORCEMENT)} enforcement points on a call path")
        return 0

    # Honest about what is and is not being checked. A gate that implied more than it verified
    # would be worse than one that says so.
    reached_count = len(ENFORCEMENT) - len(pending)
    print(
        f"reachability: {len(ENFORCEMENT)} enforcement points pinned and present, "
        f"{reached_count} on a call path.\n"
        f"  Not yet strict — {len(missing_entries)} entry point(s) absent: "
        f"{', '.join(missing_entries)}"
    )
    for symbol in pending:
        print(f"    pending  {symbol}")
    print("  Every pending symbol becomes a hard failure when the last entry point lands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
