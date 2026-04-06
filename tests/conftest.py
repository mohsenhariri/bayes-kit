import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "tests" / "data"
TOP_P_PATH = DATA_DIR / "R_top_p.npz"
GREEDY_PATH = DATA_DIR / "R_greedy.npz"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].astype(int, copy=False) for key in data.files}


@pytest.fixture(scope="session")
def top_p_data() -> dict[str, np.ndarray]:
    return _load_npz(TOP_P_PATH)


@pytest.fixture(scope="session")
def greedy_data() -> dict[str, np.ndarray]:
    return _load_npz(GREEDY_PATH)


@pytest.fixture(scope="session")
def top_p_task_aime25(top_p_data: dict[str, np.ndarray]) -> np.ndarray:
    return top_p_data["aime25"]


@pytest.fixture(scope="session")
def greedy_task_aime25(greedy_data: dict[str, np.ndarray]) -> np.ndarray:
    return greedy_data["aime25"]
