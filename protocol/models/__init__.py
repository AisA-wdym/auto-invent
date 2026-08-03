"""Generated pydantic models, one module per protocol schema.

Do not edit by hand; regenerate with `python3 tools/gen_models.py`.

One module per schema rather than one file, because several schemas declare types of the same
name. Flattened, only the last of each would be reachable.
"""


from __future__ import annotations

from .bundle_manifest import BundleManifest as BundleManifest
from .challenge import Challenge as Challenge
from .execution_receipt import ExecutionReceipt as ExecutionReceipt
from .judge_result import JudgeResult as JudgeResult
from .model_manifest import ModelManifest as ModelManifest
from .portfolio import ResearchPortfolio as ResearchPortfolio
from .season_config import SeasonConfig as SeasonConfig

__all__ = [
    "BundleManifest",
    "Challenge",
    "ExecutionReceipt",
    "JudgeResult",
    "ModelManifest",
    "ResearchPortfolio",
    "SeasonConfig",
]
