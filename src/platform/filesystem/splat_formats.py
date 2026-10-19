from __future__ import annotations

import gzip
import math
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

PathLike = Union[str, os.PathLike]
SplatProbe = Callable[[str], Optional[int]]


class InvalidSplatError(ValueError):
    pass


@dataclass(frozen=True)
class SplatFormat:
    extension: str
    mime_type: str
    probe: SplatProbe
    generated: bool = True


class SplatFormatRegistry:

    def __init__(self) -> None:
        self._by_extension: Dict[str, SplatFormat] = {}

    def register(self, fmt: SplatFormat) -> None:
        self._by_extension[fmt.extension.lower()] = fmt

    def get(self, extension: str) -> Optional[SplatFormat]:
        return self._by_extension.get(extension.lower())

    def is_registered(self, extension: str) -> bool:
        return extension.lower() in self._by_extension

    def extensions(self) -> Tuple[str, ...]:
        return tuple(self._by_extension.keys())

    def mime_types(self) -> Dict[str, str]:
        return {ext: fmt.mime_type for ext, fmt in self._by_extension.items()}


splat_format_registry = SplatFormatRegistry()


GAUSSIAN_PLY_PROPERTIES: Tuple[str, ...] = (
    'x', 'y', 'z',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
)

PLY_BINARY_LITTLE_ENDIAN = 'binary_little_endian'
VIEW_COMMENT_KEY = 'potionui_view'
VIEW_CONVENTION = 'opencv'

_PLY_SCALAR_SIZES = {
    'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
    'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
    'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
    'float': 4, 'float32': 4,
    'double': 8, 'float64': 8,
}
_PLY_FLOAT_TYPES = frozenset({'float', 'float32'})
_PLY_END_HEADER = b'end_header'
_MAX_PLY_HEADER_BYTES = 64 * 1024


@dataclass(frozen=True)
class PlyProperty:
    name: str
    type: str
    is_list: bool = False


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: Tuple[PlyProperty, ...]

    @property
    def stride(self) -> Optional[int]:
        if any(prop.is_list for prop in self.properties):
            return None
        return sum(_PLY_SCALAR_SIZES[prop.type] for prop in self.properties)

    def property_names(self) -> Tuple[str, ...]:
        return tuple(prop.name for prop in self.properties)


@dataclass(frozen=True)
class PlyHeader:
    format: str
    version: str
    elements: Tuple[PlyElement, ...]
    comments: Tuple[str, ...]
    length: int

    def element(self, name: str) -> Optional[PlyElement]:
        return next((element for element in self.elements if element.name == name), None)


@dataclass(frozen=True)
class SplatView:
    fov_y: float
    aspect: float
    depth: float
    width: int
    height: int
    convention: str = VIEW_CONVENTION

    def to_comment(self) -> str:
        return (
            f"{VIEW_COMMENT_KEY} fov_y={self.fov_y:.6g} aspect={self.aspect:.6g} "
            f"depth={self.depth:.6g} width={int(self.width)} height={int(self.height)} "
            f"convention={self.convention}"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            'fov_y': self.fov_y,
            'aspect': self.aspect,
            'depth': self.depth,
            'width': self.width,
            'height': self.height,
            'convention': self.convention,
        }


@dataclass(frozen=True)
class SplatInfo:
    format: str
    count: Optional[int]
    view: Optional[SplatView]


def parse_view_comment(comment: str) -> Optional[SplatView]:
    tokens = comment.split()
    if not tokens or tokens[0] != VIEW_COMMENT_KEY:
        return None
    fields: Dict[str, str] = {}
    for token in tokens[1:]:
        key, sep, value = token.partition('=')
        if sep:
            fields[key] = value
    try:
        view = SplatView(
            fov_y=float(fields['fov_y']),
            aspect=float(fields['aspect']),
            depth=float(fields['depth']),
            width=int(fields['width']),
            height=int(fields['height']),
            convention=fields.get('convention', VIEW_CONVENTION),
        )
    except (KeyError, ValueError):
        return None
    numbers = (view.fov_y, view.aspect, view.depth)
    if not all(math.isfinite(value) and value > 0 for value in numbers):
        return None
    if not 0 < view.fov_y < 180 or view.width <= 0 or view.height <= 0:
        return None
    return view


def read_ply_header(path: PathLike) -> PlyHeader:
    with open(path, 'rb') as f:
        head = f.read(_MAX_PLY_HEADER_BYTES)
    if not head.startswith(b'ply\n') and not head.startswith(b'ply\r\n'):
        raise InvalidSplatError(f"not a PLY file (bad magic): {path}")
    end = _find_end_header(head)
    if end is None:
        raise InvalidSplatError(f"PLY header has no end_header within {_MAX_PLY_HEADER_BYTES} bytes: {path}")
    try:
        text = head[:end].decode('ascii')
    except UnicodeDecodeError as e:
        raise InvalidSplatError(f"PLY header is not ASCII: {path}") from e
    return _parse_header_lines(text.splitlines()[1:], end, path)


def _find_end_header(head: bytes) -> Optional[int]:
    position = 0
    while True:
        index = head.find(_PLY_END_HEADER, position)
        if index < 0:
            return None
        after = index + len(_PLY_END_HEADER)
        at_line_start = index == 0 or head[index - 1:index] == b'\n'
        if at_line_start and head[after:after + 1] == b'\n':
            return after + 1
        if at_line_start and head[after:after + 2] == b'\r\n':
            return after + 2
        position = after


def _parse_header_lines(lines, length: int, path: PathLike) -> PlyHeader:
    file_format: Optional[str] = None
    version = ''
    comments = []
    elements = []
    current: Optional[Tuple[str, int, list]] = None
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        keyword = parts[0]
        if keyword == 'format' and len(parts) == 3:
            file_format, version = parts[1], parts[2]
        elif keyword == 'comment':
            comments.append(raw.strip()[len('comment'):].strip())
        elif keyword == 'obj_info':
            continue
        elif keyword == 'element' and len(parts) == 3:
            if current is not None:
                elements.append(PlyElement(current[0], current[1], tuple(current[2])))
            try:
                count = int(parts[2])
            except ValueError as e:
                raise InvalidSplatError(f"PLY element count is not an integer: {path}") from e
            if count < 0:
                raise InvalidSplatError(f"PLY element count is negative: {path}")
            current = (parts[1], count, [])
        elif keyword == 'property':
            if current is None:
                raise InvalidSplatError(f"PLY property declared before any element: {path}")
            current[2].append(_parse_property(parts, path))
        elif keyword == 'end_header':
            break
        else:
            raise InvalidSplatError(f"unrecognised PLY header line {raw!r}: {path}")
    if current is not None:
        elements.append(PlyElement(current[0], current[1], tuple(current[2])))
    if file_format is None:
        raise InvalidSplatError(f"PLY header has no format line: {path}")
    return PlyHeader(file_format, version, tuple(elements), tuple(comments), length)


def _parse_property(parts, path: PathLike) -> PlyProperty:
    if len(parts) == 5 and parts[1] == 'list':
        if parts[2] not in _PLY_SCALAR_SIZES or parts[3] not in _PLY_SCALAR_SIZES:
            raise InvalidSplatError(f"unknown PLY list property type: {path}")
        return PlyProperty(parts[4], parts[3], is_list=True)
    if len(parts) == 3 and parts[1] in _PLY_SCALAR_SIZES:
        return PlyProperty(parts[2], parts[1])
    raise InvalidSplatError(f"malformed PLY property line {' '.join(parts)!r}: {path}")


def gaussian_vertex_element(header: PlyHeader, path: PathLike = '') -> PlyElement:
    if header.format != PLY_BINARY_LITTLE_ENDIAN:
        raise InvalidSplatError(f"PLY is {header.format}, not binary_little_endian: {path}")
    if not header.elements or header.elements[0].name != 'vertex':
        raise InvalidSplatError(f"PLY's first element is not vertex: {path}")
    vertex = header.elements[0]
    if vertex.stride is None:
        raise InvalidSplatError(f"PLY vertex element has a list property: {path}")
    types = {prop.name: prop.type for prop in vertex.properties}
    missing = [name for name in GAUSSIAN_PLY_PROPERTIES if name not in types]
    if missing:
        raise InvalidSplatError(f"PLY is not a 3D Gaussian splat (missing {', '.join(missing)}): {path}")
    not_float = [name for name in GAUSSIAN_PLY_PROPERTIES if types[name] not in _PLY_FLOAT_TYPES]
    if not_float:
        raise InvalidSplatError(f"PLY Gaussian properties must be float ({', '.join(not_float)}): {path}")
    return vertex


def probe_gaussian_ply(path: str) -> Optional[int]:
    header = read_ply_header(path)
    vertex = gaussian_vertex_element(header, path)
    if vertex.count == 0:
        raise InvalidSplatError(f"PLY splat has no Gaussians: {path}")
    expected = header.length + vertex.count * vertex.stride
    actual = os.path.getsize(path)
    if actual < expected:
        raise InvalidSplatError(f"PLY splat is truncated ({actual} < {expected} bytes): {path}")
    return vertex.count


def read_ply_view(path: PathLike) -> Optional[SplatView]:
    header = read_ply_header(path)
    for comment in header.comments:
        view = parse_view_comment(comment)
        if view is not None:
            return view
    return None


SPZ_MAGIC = b'NGSP'
_GZIP_MAGIC = b'\x1f\x8b'
_SPZ_HEADER = struct.Struct('<4sIIBBBB')
_SPZ_MAX_SH_DEGREE = 3


def probe_spz(path: str) -> Optional[int]:
    with open(path, 'rb') as f:
        if f.read(2) != _GZIP_MAGIC:
            raise InvalidSplatError(f"not an SPZ file (not gzip): {path}")
    try:
        with gzip.open(path, 'rb') as f:
            header = f.read(_SPZ_HEADER.size)
    except (OSError, EOFError, zlib.error) as e:
        raise InvalidSplatError(f"SPZ gzip stream does not decompress: {path}") from e
    if len(header) < _SPZ_HEADER.size:
        raise InvalidSplatError(f"SPZ header is truncated: {path}")
    magic, version, count, sh_degree, _fractional_bits, _flags, _reserved = _SPZ_HEADER.unpack(header)
    if magic != SPZ_MAGIC:
        raise InvalidSplatError(f"not an SPZ file (bad magic): {path}")
    if version < 1:
        raise InvalidSplatError(f"unsupported SPZ version {version}: {path}")
    if sh_degree > _SPZ_MAX_SH_DEGREE:
        raise InvalidSplatError(f"SPZ spherical-harmonics degree {sh_degree} is out of range: {path}")
    if count == 0:
        raise InvalidSplatError(f"SPZ splat has no Gaussians: {path}")
    return count


ANTIMATTER_SPLAT_STRIDE = 32


def probe_antimatter_splat(path: str) -> Optional[int]:
    size = os.path.getsize(path)
    if size == 0 or size % ANTIMATTER_SPLAT_STRIDE:
        raise InvalidSplatError(
            f".splat size {size} is not a positive multiple of {ANTIMATTER_SPLAT_STRIDE} bytes: {path}"
        )
    return size // ANTIMATTER_SPLAT_STRIDE


def splat_format_name(path: PathLike) -> Optional[str]:
    extension = os.path.splitext(str(path))[1].lower()
    if not splat_format_registry.is_registered(extension):
        return None
    return extension.lstrip('.')


def inspect_splat(path: PathLike) -> SplatInfo:
    extension = os.path.splitext(str(path))[1]
    fmt = splat_format_registry.get(extension)
    if fmt is None:
        raise InvalidSplatError(f"no splat format registered for extension {extension!r}: {path}")
    count = fmt.probe(str(path))
    view = read_ply_view(path) if fmt.extension == '.ply' else None
    return SplatInfo(format=fmt.extension.lstrip('.'), count=count, view=view)


splat_format_registry.register(
    SplatFormat(extension='.ply', mime_type='model/x-ply', probe=probe_gaussian_ply)
)
splat_format_registry.register(
    SplatFormat(extension='.spz', mime_type='model/x-spz', probe=probe_spz)
)
splat_format_registry.register(
    SplatFormat(extension='.splat', mime_type='model/x-splat', probe=probe_antimatter_splat, generated=False)
)
