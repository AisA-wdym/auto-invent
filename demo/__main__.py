"""`python -m demo` — the demo service, on one port the dashboard calls.

Bound to 0.0.0.0 rather than localhost because the caller is on another host. What guards it is the
shared secret every request must present, checked in constant time; there is no route that does
anything without it except `/health`, which reports only bounds and a queue depth.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from demo.service import DemoConfig, DemoService, build_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo", description=__doc__)
    parser.add_argument("--host", default=os.environ.get("AI_DEMO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AI_DEMO_PORT", "8090")))
    parser.add_argument("--check", action="store_true", help="build and validate, then exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s | %(message)s"
    )

    try:
        config = DemoConfig.from_environment()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    service = DemoService(config=config)
    app = build_app(service)

    if args.check:
        print(
            f"demo --check passed — ${config.maximum_run_usd:.2f}/run, "
            f"${config.maximum_daily_usd:.0f}/day, {config.concurrency} concurrent, "
            f"image {config.image}, season {config.season}"
        )
        return 0

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
