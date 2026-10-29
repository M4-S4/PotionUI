"""Identity vectors of faces from AuraFace, an ArcFace face recognition model run by onnxruntime on the CPU, each face aligned on the five landmarks YuNet finds."""

import importlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from PIL import Image

TEMPLATE = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041))
SIZE = 112
THREADS = 2

Face = Tuple[Union[str, Path, Image.Image], Sequence[Sequence[float]]]

_sessions: Dict[str, Any] = {}
_sessions_lock = threading.Lock()


def _rgb(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def align(image: Union[str, Path, Image.Image], landmarks: Sequence[Sequence[float]]) -> Any:
    """The face of an image turned, scaled and moved so that its five landmarks sit on the ArcFace template, as a 112×112 RGB array; ValueError when the landmarks give no alignment."""
    import cv2
    import numpy as np

    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(points, np.asarray(TEMPLATE, dtype=np.float32), method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("The five landmarks of the face give no alignment.")
    return cv2.warpAffine(np.asarray(_rgb(image)), matrix, (SIZE, SIZE), borderValue=0)


def runtime_imports() -> bool:
    """Whether onnxruntime, which runs the model, imports in this process."""
    try:
        importlib.import_module("onnxruntime")
    except ImportError:
        return False
    return True


def _session(model: Path) -> Any:
    key = str(model)
    session = _sessions.get(key)
    if session is not None:
        return session
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _sessions[key] = _new_session(key)
    return session


def _new_session(path: str) -> Any:
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def embed(model: Path, faces: Sequence[Face]) -> List[List[float]]:
    """The L2-normalized identity vector of each face, given as an image and the five landmarks of that face in it, one face at a time; the model file is given, never downloaded."""
    import numpy as np

    if not faces:
        return []
    session = _session(model)
    name = session.get_inputs()[0].name
    vectors = []
    for image, landmarks in faces:
        aligned = (align(image, landmarks).astype(np.float32) - 127.5) / 127.5
        vector = np.asarray(session.run(None, {name: aligned.transpose(2, 0, 1)[None]})[0][0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        vectors.append((vector / norm if norm else vector).tolist())
    return vectors
