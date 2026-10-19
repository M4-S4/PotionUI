import gzip

import pytest

from src.platform.filesystem.mesh_formats import mesh_format_registry
from src.platform.filesystem.splat_formats import (
    GAUSSIAN_PLY_PROPERTIES,
    InvalidSplatError,
    SplatFormat,
    SplatFormatRegistry,
    SplatView,
    inspect_splat,
    parse_view_comment,
    probe_antimatter_splat,
    probe_gaussian_ply,
    probe_spz,
    read_ply_header,
    read_ply_view,
    splat_format_name,
    splat_format_registry,
)
from tests.fixtures.splat_fixtures import (
    SPLAT_COUNT,
    build_antimatter_splat,
    build_gaussian_ply,
    build_spz,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


class TestRegisteredFormats:

    @pytest.mark.parametrize("extension,mime", [
        ('.ply', 'model/x-ply'),
        ('.spz', 'model/x-spz'),
        ('.splat', 'model/x-splat'),
    ])
    def test_each_format_is_registered_with_its_mime(self, extension, mime):
        assert splat_format_registry.is_registered(extension)
        assert splat_format_registry.is_registered(extension.upper())
        assert splat_format_registry.get(extension).mime_type == mime

    def test_glb_is_not_a_splat(self):
        assert splat_format_registry.get('.glb') is None

    def test_ply_is_a_splat_and_still_not_a_mesh(self):
        assert splat_format_registry.is_registered('.ply')
        assert not mesh_format_registry.is_registered('.ply')

    def test_only_antimatter_splat_is_import_only(self):
        assert splat_format_registry.get('.ply').generated is True
        assert splat_format_registry.get('.spz').generated is True
        assert splat_format_registry.get('.splat').generated is False

    def test_registered_probes_are_the_module_functions(self):
        assert splat_format_registry.get('.ply').probe is probe_gaussian_ply
        assert splat_format_registry.get('.spz').probe is probe_spz
        assert splat_format_registry.get('.splat').probe is probe_antimatter_splat

    def test_private_registry_is_case_normalized(self):
        registry = SplatFormatRegistry()
        fmt = SplatFormat(extension='.KSPLAT', mime_type='model/x-ksplat', probe=lambda path: 1)
        registry.register(fmt)
        assert registry.get('.ksplat') is fmt
        assert registry.extensions() == ('.ksplat',)
        assert registry.mime_types() == {'.ksplat': 'model/x-ksplat'}


class TestProbeGaussianPly:

    def test_valid_ply_returns_its_gaussian_count(self, gaussian_ply_file):
        assert probe_gaussian_ply(str(gaussian_ply_file)) == SPLAT_COUNT

    def test_extra_properties_are_accepted(self, tmp_path):
        properties = GAUSSIAN_PLY_PROPERTIES[:3] + ('nx', 'ny', 'nz') + GAUSSIAN_PLY_PROPERTIES[3:] + ('f_rest_0',)
        path = _write(tmp_path, 'trained.ply', build_gaussian_ply(properties=properties))
        assert probe_gaussian_ply(path) == SPLAT_COUNT

    def test_crlf_header_is_accepted(self, tmp_path):
        payload = build_gaussian_ply()
        header_end = payload.index(b'end_header\n') + len(b'end_header\n')
        crlf = payload[:header_end].replace(b'\n', b'\r\n') + payload[header_end:]
        assert probe_gaussian_ply(_write(tmp_path, 'crlf.ply', crlf)) == SPLAT_COUNT

    @pytest.mark.parametrize("reason,payload", [
        ("mesh ply without gaussian properties", build_gaussian_ply(properties=('x', 'y', 'z'))),
        ("missing one rotation component", build_gaussian_ply(properties=GAUSSIAN_PLY_PROPERTIES[:-1])),
        ("ascii ply", build_gaussian_ply(file_format='ascii')),
        ("big-endian ply", build_gaussian_ply(file_format='binary_big_endian')),
        ("double properties", build_gaussian_ply(property_type='double')),
        ("truncated body", build_gaussian_ply(truncate=7)),
        ("declared count beyond the data", build_gaussian_ply(declared_count=SPLAT_COUNT + 1)),
        ("no gaussians", build_gaussian_ply(count=0)),
        ("vertex is not the first element", build_gaussian_ply(first_element='element face 0\nproperty list uchar int vertex_indices')),
        ("not a ply at all", b'glTF' + bytes(64)),
        ("no end_header", b'ply\nformat binary_little_endian 1.0\nelement vertex 1\n'),
    ])
    def test_invalid_ply_is_rejected(self, tmp_path, reason, payload):
        with pytest.raises(InvalidSplatError):
            probe_gaussian_ply(_write(tmp_path, 'bad.ply', payload))

    def test_header_reports_elements_comments_and_length(self, gaussian_ply_file, gaussian_ply_bytes):
        header = read_ply_header(gaussian_ply_file)
        vertex = header.element('vertex')
        assert header.format == 'binary_little_endian'
        assert vertex.count == SPLAT_COUNT
        assert vertex.property_names() == GAUSSIAN_PLY_PROPERTIES
        assert vertex.stride == 4 * len(GAUSSIAN_PLY_PROPERTIES)
        assert header.length == gaussian_ply_bytes.index(b'end_header\n') + len(b'end_header\n')
        assert header.comments[0].startswith('potionui_view ')


class TestProbeSpz:

    def test_valid_spz_returns_its_point_count(self, spz_file):
        assert probe_spz(str(spz_file)) == SPLAT_COUNT

    @pytest.mark.parametrize("reason,payload", [
        ("not gzip", b'NGSP' + bytes(60)),
        ("bad magic", build_spz(magic=b'XXXX')),
        ("version zero", build_spz(version=0)),
        ("sh degree out of range", build_spz(sh_degree=4)),
        ("no points", build_spz(count=0)),
        ("header truncated", gzip.compress(b'NGSP\x02\x00', mtime=0)),
        ("corrupt gzip stream", b'\x1f\x8b\x08\x00' + bytes(40)),
    ], ids=lambda value: value if isinstance(value, str) else "")
    def test_invalid_spz_is_rejected(self, tmp_path, reason, payload):
        with pytest.raises(InvalidSplatError):
            probe_spz(_write(tmp_path, 'bad.spz', payload))


class TestProbeAntimatterSplat:

    def test_count_is_size_over_32(self, antimatter_splat_file):
        assert probe_antimatter_splat(str(antimatter_splat_file)) == SPLAT_COUNT

    @pytest.mark.parametrize("payload", [b'', bytes(31), bytes(33)])
    def test_size_not_a_multiple_of_32_is_rejected(self, tmp_path, payload):
        with pytest.raises(InvalidSplatError):
            probe_antimatter_splat(_write(tmp_path, 'bad.splat', payload))


class TestViewComment:

    def test_comment_round_trips(self):
        view = SplatView(fov_y=47.5, aspect=16 / 9, depth=3.25, width=1280, height=720)
        parsed = parse_view_comment(view.to_comment())
        assert parsed is not None
        assert parsed.fov_y == pytest.approx(47.5)
        assert parsed.aspect == pytest.approx(16 / 9, rel=1e-5)
        assert parsed.depth == pytest.approx(3.25)
        assert (parsed.width, parsed.height, parsed.convention) == (1280, 720, 'opencv')

    def test_comment_has_the_contract_shape(self):
        view = SplatView(fov_y=50, aspect=1.5, depth=4, width=768, height=512)
        assert view.to_comment() == (
            'potionui_view fov_y=50 aspect=1.5 depth=4 width=768 height=512 convention=opencv'
        )

    @pytest.mark.parametrize("comment", [
        'something else entirely',
        'potionui_view fov_y=50 aspect=1.5 depth=4 width=768',
        'potionui_view fov_y=abc aspect=1.5 depth=4 width=768 height=512',
        'potionui_view fov_y=0 aspect=1.5 depth=4 width=768 height=512',
        'potionui_view fov_y=50 aspect=1.5 depth=-1 width=768 height=512',
        'potionui_view fov_y=50 aspect=nan depth=4 width=768 height=512',
        'potionui_view fov_y=200 aspect=1.5 depth=4 width=768 height=512',
    ])
    def test_malformed_comment_is_ignored(self, comment):
        assert parse_view_comment(comment) is None

    def test_view_is_read_from_the_ply_header(self, gaussian_ply_file):
        view = read_ply_view(gaussian_ply_file)
        assert view == SplatView(fov_y=50.0, aspect=1.5, depth=4.0, width=768, height=512)

    def test_ply_without_comment_has_no_view(self, tmp_path):
        path = _write(tmp_path, 'plain.ply', build_gaussian_ply(comments=('generated elsewhere',)))
        assert read_ply_view(path) is None


class TestInspectSplat:

    def test_ply_reports_format_count_and_view(self, gaussian_ply_file):
        info = inspect_splat(gaussian_ply_file)
        assert info.format == 'ply'
        assert info.count == SPLAT_COUNT
        assert info.view.as_dict()['fov_y'] == 50.0

    def test_spz_and_splat_report_no_view(self, spz_file, antimatter_splat_file):
        assert inspect_splat(spz_file).view is None
        assert inspect_splat(antimatter_splat_file).format == 'splat'

    def test_unregistered_extension_is_rejected(self, tmp_path):
        path = _write(tmp_path, 'scene.glb', build_antimatter_splat())
        with pytest.raises(InvalidSplatError):
            inspect_splat(path)

    def test_format_name_follows_the_registry(self):
        assert splat_format_name('a/b/scene.PLY') == 'ply'
        assert splat_format_name('scene.spz') == 'spz'
        assert splat_format_name('scene.glb') is None
