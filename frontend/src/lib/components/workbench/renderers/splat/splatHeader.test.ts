import { describe, it, expect } from 'vitest';
import { MAX_HEADER_BYTES, parseSourceView, readPlyHeader } from './splatHeader';

const VIEW = 'comment potionui_view fov_y=47.5 aspect=1.5 depth=4.0 width=1536 height=1024 convention=opencv';

function ply(lines: string[], body: Uint8Array = new Uint8Array([1, 2, 3, 4]), eol = '\n'): Uint8Array {
	const header = new TextEncoder().encode(lines.join(eol) + eol);
	const bytes = new Uint8Array(header.length + body.length);
	bytes.set(header, 0);
	bytes.set(body, header.length);
	return bytes;
}

const BASE = ['ply', 'format binary_little_endian 1.0', 'element vertex 1', 'property float x', 'end_header'];

function withComment(comment: string): string[] {
	return [...BASE.slice(0, 2), comment, ...BASE.slice(2)];
}

describe('parseSourceView', () => {
	it('reads the camera the generator wrote into the header comment', () => {
		expect(parseSourceView(ply(withComment(VIEW)))).toEqual({
			fovY: 47.5,
			aspect: 1.5,
			depth: 4,
			width: 1536,
			height: 1024
		});
	});

	it('reads it among other comments and with CRLF line endings', () => {
		const lines = ['ply', 'format binary_little_endian 1.0', 'comment made by a generator', VIEW, 'comment more', 'element vertex 1', 'property float x', 'end_header'];
		expect(parseSourceView(ply(lines, undefined, '\r\n'))?.depth).toBe(4);
	});

	it('keeps the fields it is not given as null but still needs the field of view and the depth', () => {
		const view = parseSourceView(ply(withComment('comment potionui_view fov_y=60 depth=2.5')));
		expect(view).toEqual({ fovY: 60, depth: 2.5, aspect: null, width: null, height: null });
		expect(parseSourceView(ply(withComment('comment potionui_view fov_y=60 aspect=1.5')))).toBeNull();
		expect(parseSourceView(ply(withComment('comment potionui_view depth=4')))).toBeNull();
	});

	it('is null when the header has no such comment', () => {
		expect(parseSourceView(ply(BASE))).toBeNull();
		expect(parseSourceView(ply(withComment('comment potionui_viewer fov_y=50 depth=4')))).toBeNull();
	});

	it('is null for a header that stops before end_header', () => {
		const whole = ply(withComment(VIEW));
		const cut = whole.subarray(0, new TextDecoder().decode(whole).indexOf('end_header') + 5);
		expect(parseSourceView(cut)).toBeNull();
		expect(parseSourceView(whole.subarray(0, 20))).toBeNull();
		expect(parseSourceView(new TextEncoder().encode('ply\nformat binary_little_endian 1.0\n' + VIEW))).toBeNull();
	});

	it('is null when end_header is farther than the size limit', () => {
		const filler = Array.from({ length: 4000 }, (_, i) => `comment filler ${i} xxxxxxxxxxxxxxxxxxxxxxxx`);
		const bytes = ply(['ply', 'format binary_little_endian 1.0', ...filler, VIEW, 'end_header']);
		expect(bytes.length).toBeGreaterThan(MAX_HEADER_BYTES);
		expect(parseSourceView(bytes)).toBeNull();
	});

	it('does not look past end_header, even if the binary data spells the comment', () => {
		const body = new TextEncoder().encode(VIEW + '\n');
		expect(parseSourceView(ply(BASE, body))).toBeNull();
	});

	it('is null for a file that is not a PLY, and for an empty one', () => {
		expect(parseSourceView(new Uint8Array([0x1f, 0x8b, 0x08, 0x00]))).toBeNull();
		expect(parseSourceView(new TextEncoder().encode(VIEW))).toBeNull();
		expect(parseSourceView(new Uint8Array())).toBeNull();
	});

	it('ignores a camera it cannot use', () => {
		for (const comment of [
			'comment potionui_view fov_y=0 depth=4',
			'comment potionui_view fov_y=180 depth=4',
			'comment potionui_view fov_y=50 depth=0',
			'comment potionui_view fov_y=50 depth=-3',
			'comment potionui_view fov_y=abc depth=4',
			'comment potionui_view fov_y=50 depth=nan',
			'comment potionui_view fov_y=50 depth=4 convention=opengl'
		]) {
			expect(parseSourceView(ply(withComment(comment))), comment).toBeNull();
		}
	});
});

describe('readPlyHeader', () => {
	it('returns the text up to and including the end_header line, without the binary data', () => {
		const header = readPlyHeader(ply(BASE, new Uint8Array([0xff, 0xfe, 0x00])));
		expect(header).toBe(BASE.join('\n') + '\n');
	});

	it('is null for a non-PLY file', () => {
		expect(readPlyHeader(new Uint8Array([1, 2, 3, 4, 5]))).toBeNull();
	});
});
