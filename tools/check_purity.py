#!/usr/bin/env python3
"""Static check: the scoring path may not read the clock, a global RNG, or the network.

A single `time.time()` in a scoring module does not raise, does not fail a test, and does
not show up in review. What it does is make a rerun of the same bundle produce a different
score — and architecture.md 27 requires same-bundle rerun rank correlation at 0.80 or above,
which is exactly the property such a call silently destroys.

No test can catch this by itself, because a process compared against itself agrees perfectly.
So the check is structural: these modules may not even *import* a source of ambient state.

## What is in scope, and why the list grows rather than shrinks

Every module here was clean when it was added. That is the point — the gate is a ratchet, not
a migration. What it buys is that the next edit to a scoring module cannot quietly introduce
a clock read.

The scope is drawn by consequence, not by package: a module is listed if a non-deterministic
value in it would change a score. That includes the seed derivation (a different seed is a
different pack), the fixed-point arithmetic (the scoring primitives), and the receipt chain
(a digest is a commitment).

Run:  python3 tools/check_purity.py
Exit: 0 clean, 1 on any violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Modules that must be pure functions of their arguments.
PURE_MODULES = (
    "protocol/canonical.py",
    "protocol/fixedpoint.py",
    "protocol/seeds.py",
    "protocol/receipts.py",
    # The reward path. A non-deterministic value in any of these changes what a miner is
    # paid, and changes it identically for everyone — so it is invisible in a ranking right
    # up to the point it decides a qualification floor.
    "validator/judge/bradley_terry.py",
    "validator/scoring/criteria.py",
    "validator/scoring/daily.py",
    "validator/weights.py",
    # The challenge pack is upstream of every score, so a non-deterministic value here cannot be
    # separated from signal anywhere downstream. These four are the steps that must decide the
    # same way on every validator and every host: the seeded plan, the linter, the dedup
    # comparison and the safety screen. The generator and critic are deliberately absent — they
    # call models, so they cannot be deterministic and are not listed.
    "protocol/commitments.py",
    "validator/challenge_factory/taxonomy.py",
    "validator/challenge_factory/linter.py",
    "validator/challenge_factory/dedup.py",
    "validator/challenge_factory/safety.py",
    "validator/challenge_factory/discriminator.py",
)

# Importing any of these admits ambient state into a supposedly pure module.
FORBIDDEN_IMPORTS = {
    "time": "reads the clock",
    "datetime": "reads the clock",
    "random": "global random source; derive from the daily seed instead",
    "secrets": "non-reproducible randomness",
    "socket": "network access",
    "http": "network access",
    "urllib": "network access",
    "httpx": "network access",
    "requests": "network access",
    "redis": "external state; the store belongs above the pure layer",
    "asyncio": "concurrency implies ordering non-determinism",
    "os": "ambient environment and os.urandom",
    "pathlib": "filesystem access",
    "sqlite3": "external state",
    "uuid": "uuid4 is non-reproducible",
}

# Attribute calls forbidden even where the module was reached indirectly.
FORBIDDEN_CALLS = {
    ("time", "time"),
    ("time", "monotonic"),
    ("time", "time_ns"),
    ("random", "random"),
    ("random", "randint"),
    ("random", "shuffle"),
    ("random", "choice"),
    ("os", "urandom"),
    ("os", "getenv"),
    ("uuid", "uuid4"),
    ("secrets", "token_bytes"),
    ("secrets", "token_hex"),
}


class PurityVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.violations: list[str] = []

    def _root(self, dotted: str) -> str:
        return dotted.split(".", 1)[0]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = self._root(alias.name)
            if root in FORBIDDEN_IMPORTS:
                self.violations.append(
                    f"{self.path}:{node.lineno}: imports {alias.name} "
                    f"({FORBIDDEN_IMPORTS[root]})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # `level == 0` is an absolute import. A relative import stays inside the package and
        # is checked on its own terms when that module is itself listed.
        if node.module and node.level == 0:
            root = self._root(node.module)
            if root in FORBIDDEN_IMPORTS:
                self.violations.append(
                    f"{self.path}:{node.lineno}: imports from {node.module} "
                    f"({FORBIDDEN_IMPORTS[root]})"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in FORBIDDEN_CALLS:
                self.violations.append(f"{self.path}:{node.lineno}: calls {pair[0]}.{pair[1]}()")
        self.generic_visit(node)


def check(root: Path) -> list[str]:
    violations: list[str] = []
    for relative in PURE_MODULES:
        path = root / relative
        if not path.exists():
            # Listed but absent: a rename that left the gate pointing at nothing would
            # otherwise pass silently, and a gate watching a file that does not exist is a
            # gate watching nothing.
            violations.append(f"{relative}: listed as pure but missing")
            continue
        visitor = PurityVisitor(relative)
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        violations.extend(visitor.violations)
    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations = check(root)
    if violations:
        print("Purity violations — these modules must be pure functions of their arguments:")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nA non-deterministic value here changes a score without raising, and "
            "architecture.md 27 requires same-bundle rerun correlation at 0.80 or above."
        )
        return 1
    print(f"purity: {len(PURE_MODULES)} modules clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
