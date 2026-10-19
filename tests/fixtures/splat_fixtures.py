import gzip
import struct
from typing import Iterable, Optional, Sequence

import pytest

GAUSSIAN_PROPERTIES = (
    'x', 'y', 'z',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
)

SPLAT_COUNT = 4

VIEW_COMMENT = (
    "potionui_view fov_y=50 aspect=1.5 depth=4 width=768 height=512 convention=opencv"
)


def _gaussian_row(index: int, properties: Sequence[str]) -> list:
    values = {
        'x': 0.1 * index, 'y': -0.1 * index, 'z': 4.0 + 0.1 * index,
        'f_dc_0': 0.5, 'f_dc_1': 0.0, 'f_dc_2': -0.5,
        'opacity': 2.0,
        'scale_0': -4.0, 'scale_1': -4.0, 'scale_2': -4.0,
        'rot_0': 1.0, 'rot_1': 0.0, 'rot_2': 0.0, 'rot_3': 0.0,
    }
    return [values.get(name, 0.0) for name in properties]


def build_gaussian_ply(
    count: int = SPLAT_COUNT,
    comments: Iterable[str] = (VIEW_COMMENT,),
    properties: Sequence[str] = GAUSSIAN_PROPERTIES,
    property_type: str = 'float',
    file_format: str = 'binary_little_endian',
    declared_count: Optional[int] = None,
    truncate: int = 0,
    first_element: Optional[str] = None,
) -> bytes:
    lines = ['ply', f'format {file_format} 1.0']
    lines += [f'comment {comment}' for comment in comments]
    if first_element:
        lines.append(first_element)
    lines.append(f'element vertex {declared_count if declared_count is not None else count}')
    lines += [f'property {property_type} {name}' for name in properties]
    lines.append('end_header')
    header = ('\n'.join(lines) + '\n').encode('ascii')

    code = {'float': 'f', 'double': 'd'}[property_type]
    rows = [value for index in range(count) for value in _gaussian_row(index, properties)]
    body = struct.pack('<' + code * len(rows), *rows)
    payload = header + body
    return payload[:len(payload) - truncate] if truncate else payload


def build_spz(
    count: int = SPLAT_COUNT,
    version: int = 2,
    magic: bytes = b'NGSP',
    sh_degree: int = 0,
) -> bytes:
    header = struct.pack('<4sIIBBBB', magic, version, count, sh_degree, 12, 0, 0)
    body = bytes(count * (9 + 1 + 3 + 3 + 3))
    return gzip.compress(header + body, mtime=0)


def build_antimatter_splat(count: int = SPLAT_COUNT) -> bytes:
    rows = b''
    for index in range(count):
        rows += struct.pack('<3f', 0.1 * index, 0.0, 4.0)
        rows += struct.pack('<3f', 0.01, 0.01, 0.01)
        rows += bytes((200, 100, 50, 255))
        rows += bytes((255, 128, 128, 128))
    return rows


@pytest.fixture
def gaussian_ply_bytes() -> bytes:
    return build_gaussian_ply()


@pytest.fixture
def gaussian_ply_file(tmp_path, gaussian_ply_bytes):
    path = tmp_path / "source_scene.ply"
    path.write_bytes(gaussian_ply_bytes)
    return path


@pytest.fixture
def spz_file(tmp_path):
    path = tmp_path / "source_scene.spz"
    path.write_bytes(build_spz())
    return path


@pytest.fixture
def antimatter_splat_file(tmp_path):
    path = tmp_path / "source_scene.splat"
    path.write_bytes(build_antimatter_splat())
    return path
