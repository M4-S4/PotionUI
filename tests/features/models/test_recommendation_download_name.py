"""A recommendation download lands under the recommendation's file name, and the RIFE picks of Video Tools carry names the indexer reads."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from src.features.models.dto import RecommendationDownloadRequest
from src.features.models.routes import ModelController
from src.platform.filesystem.model_types import SUPPORTED_MODEL_EXTENSIONS
from src.platform.security.user import AccountType

INTERPOLATE_TAB = (
    Path(__file__).resolve().parents[3]
    / "content/presets/marketplace/VideoTools/modes/interpolate/tabs/generation.yml"
)
RIFE_425 = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl"


async def _queued(name):
    queue = Mock()
    queue.queue_model_download = AsyncMock(return_value=Mock(id="d1"))
    controller = ModelController(Mock(), Mock(), queue)

    response = await controller.queue_recommendation_download(
        RecommendationDownloadRequest(name=name, model_type="vfi", link=RIFE_425),
        Mock(id="u1", account_type=AccountType.ADMIN),
    )

    assert response.success is True
    return queue.queue_model_download.await_args.kwargs


async def test_a_model_file_name_replaces_the_url_name():
    queued = await _queued("flownet_v4.25.pth")

    assert queued["filename"] == "flownet_v4.25.pth"
    assert queued["url"] == RIFE_425


@pytest.mark.parametrize("name", [
    "RealESRGAN 4x+ Anime",
    "flownet_v4.25.pkl",
    "vfi/flownet_v4.25.pth",
    "..\\flownet_v4.25.pth",
    "",
])
async def test_any_other_name_keeps_the_url_name(name):
    assert (await _queued(name))["filename"] is None


def test_every_rife_pick_has_a_name_the_indexer_reads():
    tab = yaml.safe_load(INTERPOLATE_TAB.read_text(encoding="utf-8"))
    model = next(field for field in tab["fields"] if field["name"] == "model")
    names = [rec["name"] for rec in model["configuration"]["recommendations"]]

    assert set(names) == {"flownet_v4.25.pth", "flownet_v4.26.pth", "rife49.pth"}
    assert all(Path(name).suffix in SUPPORTED_MODEL_EXTENSIONS for name in names)
