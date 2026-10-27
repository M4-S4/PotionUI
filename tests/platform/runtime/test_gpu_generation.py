import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.platform.runtime.gpu_generation import GpuGeneration, generation_for
from tests.fixtures.kernel_capabilities import fake_device

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "capability,generation",
    [
        (None, GpuGeneration.NONE),
        ((), GpuGeneration.NONE),
        ((0, 0), GpuGeneration.NONE),
        ((0, 5), GpuGeneration.NONE),
        ((3, 5), GpuGeneration.PRE_AMPERE),
        ((6, 1), GpuGeneration.PRE_AMPERE),
        ((7, 5), GpuGeneration.PRE_AMPERE),
        ((8, 0), GpuGeneration.AMPERE),
        ((8, 6), GpuGeneration.AMPERE),
        ((8, 7), GpuGeneration.AMPERE),
        ((8, 9), GpuGeneration.ADA),
        ((9, 0), GpuGeneration.HOPPER),
        ((9, 5), GpuGeneration.HOPPER),
        ((10, 0), GpuGeneration.BLACKWELL),
        ((10, 3), GpuGeneration.BLACKWELL),
        ((11, 0), GpuGeneration.BLACKWELL),
        ((12, 0), GpuGeneration.BLACKWELL),
        ((12, 1), GpuGeneration.BLACKWELL),
        ((13, 0), GpuGeneration.UNKNOWN),
        ((20, 4), GpuGeneration.UNKNOWN),
    ],
)
def test_generation_for_capability_table(capability, generation):
    assert generation_for(capability) is generation


def test_kernel_snapshot_shares_the_enum_and_the_mapping():
    from src.platform.runtime.native.kernels import GpuGeneration as kernel_generation

    assert kernel_generation is GpuGeneration
    for major in range(0, 16):
        for minor in range(0, 10):
            assert fake_device((major, minor)).generation is generation_for((major, minor)), (major, minor)


def test_generation_modules_import_without_torch_or_the_native_engine():
    script = (
        "import sys\n"
        "import src.platform.runtime.gpu_generation\n"
        "import src.platform.runtime.gpu_profile\n"
        "loaded = sorted(m for m in sys.modules if m == 'torch' or m.startswith('src.platform.runtime.native'))\n"
        "print('LOADED:' + ','.join(loaded))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in [str(REPO_ROOT), *sys.path] if path)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "LOADED:"
