from src.features.models.digest_queue_repository import DigestQueueRepository
from src.features.models.digests import DigestManager
from src.features.models.file_digest import signature_ranges
from src.features.models.indexer import ModelScanner
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.read_throttle import ReadThrottle
from src.features.models.repository import model_repo
from src.platform.settings.repository import SettingRepository
from tests.features.models.fakes import FakeClock, drain_manager
from tests.features.models.test_indexer_roots import _binding, _models, _resolver, _root, _write

SMALL_LIMIT = 10
BLOB = 200


def _blob(seed, size=BLOB):
    return bytes((seed + i * 7) % 251 for i in range(size))


def _unsampled_offset(size):
    covered = [(start, start + length) for start, length in signature_ranges(size)]
    for offset in range(1, size):
        if not any(start <= offset < end for start, end in covered):
            return offset
    raise AssertionError("every byte is sampled")


class Harness:
    def __init__(self, tmp_path, with_llm=False, extra_types=()):
        self.home = tmp_path / "home"
        self.lib = tmp_path / "lib"
        roots = [_root("r_home", self.home), _root("r_lib", self.lib)]
        bindings = [
            _binding("r_home", "lora", self.home / "loras", 0),
            _binding("r_lib", "lora", self.lib / "loras", 1),
        ]
        if with_llm:
            bindings.append(_binding("r_home", "llm", self.home / "llm", 0))
        for extra in extra_types:
            bindings.append(_binding("r_home", extra, self.home / extra, 0))
        self.roots = {"home": self.home, "lib": self.lib}
        self.clock = FakeClock()
        self.governor = IndexingGovernor(throttle=ReadThrottle(clock=self.clock.monotonic, sleep=self.clock.sleep))
        self.scanner = ModelScanner(_resolver(roots, bindings), governor=self.governor)
        self.queue = DigestQueueRepository()
        self.applied = []
        self.contested = []
        self.manager = DigestManager(
            self.queue,
            self.governor,
            self.scanner.location_path,
            locations_write_lock=self.scanner.write_lock,
            on_applied=self._on_applied,
            setting_repository=SettingRepository(),
            clock=self.clock.time,
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep,
        )
        self.scanner.attach_digests(self.manager)
        self.scanner.contested_listener = lambda found, outcome, error: self.contested.append((found, outcome, error))

    def _on_applied(self, model_ids):
        self.applied.append(set(model_ids))
        self.scanner.settle_digested(set(model_ids))

    def put(self, where, name, content, folder="loras"):
        path = self.roots[where] / folder / name
        _write(path, content)
        return path

    def index(self):
        return self.scanner.index_models(max_workers=1)

    def settle(self, max_jobs=50):
        return drain_manager(self.manager, max_jobs=max_jobs)

    def model(self, filename):
        rows = [m for m in _models() if m.filename == filename]
        assert len(rows) == 1, [(m.filename, m.model_type) for m in _models()]
        return model_repo.get_by_id(rows[0].id, include_providers=False, include_tags=False)

    def locations(self, model):
        return sorted(self.scanner.locations.list_for_model(model.id), key=lambda row: row["root_id"])

    def queued(self):
        with_rows = []
        for model in _models():
            for location in self.scanner.locations.list_for_model(model.id):
                row = self.queue.get(location["id"])
                if row is not None:
                    with_rows.append((model.filename, location["root_id"], row["priority"], row["reason"]))
        return sorted(with_rows)
