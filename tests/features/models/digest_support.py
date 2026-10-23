from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from src.features.models.locations_repository import ModelLocationsRepository
from src.features.models.records import Model
from src.features.models.repository import model_repo
from src.platform.database import get_cursor
from src.platform.database.rows import now_iso
from tests.features.models.test_indexer_roots import _binding, _root


class DigestLibrary:
    def __init__(self, tmp_path):
        self.home = tmp_path / "home"
        self.loras = self.home / "loras"
        self.loras.mkdir(parents=True)
        _root("r_home", self.home)
        self.binding = _binding("r_home", "lora", self.loras, 0)
        self.locations = ModelLocationsRepository()

    def path_for(self, row) -> Optional[Path]:
        return self.loras / row["rel_path"]

    def add(self, name, content, sha256=None, rel_path=None, model=None, status="present", mtime_ns=None):
        rel_path = rel_path or name
        path = self.loras / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        stat = path.stat()
        if model is None:
            model = model_repo.create(
                Model(filename=name, file_size=stat.st_size, sha256=sha256, model_type="lora")
            )
        location_id = self.locations.upsert(
            model_id=model.id,
            binding_id=self.binding.binding_id,
            root_id="r_home",
            model_type="lora",
            rel_path=rel_path,
            rel_key=rel_path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns if mtime_ns is None else mtime_ns,
            sha256=sha256,
            status=status,
            seen_at=now_iso(),
        )
        return SimpleNamespace(model=model, location_id=location_id, path=path)

    def set_digest(self, item, sha256, quick_sig=None):
        with get_cursor() as cursor:
            cursor.execute(
                "UPDATE model_locations SET sha256 = ?, quick_sig = ? WHERE id = ?", (sha256, quick_sig, item.location_id)
            )

    def set_status(self, item, status):
        with get_cursor() as cursor:
            cursor.execute("UPDATE model_locations SET status = ? WHERE id = ?", (status, item.location_id))
