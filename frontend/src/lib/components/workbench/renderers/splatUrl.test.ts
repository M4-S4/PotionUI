import { describe, it, expect } from 'vitest';
import { resolveSplatUrl, resolveSplatMetadata, resolveSplatFormat } from './splatUrl';

describe('resolveSplatUrl', () => {
	it('prefers current_splat, the live-generation field', () => {
		expect(
			resolveSplatUrl({ current_splat: '/api/splat/a.ply', url: '/api/splat/b.ply' })
		).toBe('/api/splat/a.ply');
	});

	it('falls back to originalUrl over url for a gallery-item shape', () => {
		expect(
			resolveSplatUrl({ originalUrl: '/api/splat/orig.ply', url: '/api/splat/thumb.ply' })
		).toBe('/api/splat/orig.ply');
	});

	it('falls back to url when nothing else is present', () => {
		expect(resolveSplatUrl({ url: '/api/splat/only.spz' })).toBe('/api/splat/only.spz');
	});

	it('returns null when the file has no resolvable URL', () => {
		expect(resolveSplatUrl({})).toBeNull();
	});

	it('returns null for a nullish file', () => {
		expect(resolveSplatUrl(null)).toBeNull();
		expect(resolveSplatUrl(undefined)).toBeNull();
	});
});

describe('resolveSplatMetadata', () => {
	it('reads nested splat_metadata for the live-generation shape', () => {
		expect(
			resolveSplatMetadata({
				current_splat: '/api/splat/a.ply',
				splat_metadata: { splat_count: 120000, filename: 'a.ply', format: 'ply' }
			})
		).toEqual({ splatCount: 120000, filename: 'a.ply', format: 'ply' });
	});

	it('reads flat fields for a gallery/history-item shape', () => {
		expect(
			resolveSplatMetadata({
				url: '/api/splat/b.spz',
				splat_count: 8,
				splat_name: 'b.spz',
				splat_format: 'spz'
			})
		).toEqual({ splatCount: 8, filename: 'b.spz', format: 'spz' });
	});

	it('prefers nested splat_metadata over flat fields when both are present', () => {
		expect(
			resolveSplatMetadata({
				splat_metadata: { splat_count: 1 },
				splat_count: 999
			}).splatCount
		).toBe(1);
	});

	it('returns all-null metadata for a file with none, or a nullish file', () => {
		const empty = { splatCount: null, filename: null, format: null };
		expect(resolveSplatMetadata({ url: '/api/splat/c.ply' })).toEqual(empty);
		expect(resolveSplatMetadata(null)).toEqual(empty);
	});
});

describe('resolveSplatFormat', () => {
	it('surfaces the format the server sent, lowercased', () => {
		expect(resolveSplatFormat({ splat_format: 'SPZ' })).toBe('spz');
		expect(
			resolveSplatFormat({ current_splat: '/api/splat/a.splat', splat_metadata: { format: 'splat' } })
		).toBe('splat');
	});

	it('falls back to the extension of the url, then of the file path', () => {
		expect(resolveSplatFormat({ url: '/api/media/generations/g/1.ply?v=2' })).toBe('ply');
		expect(resolveSplatFormat({ file_path: 'generations/2026-10-19/g/2.spz' })).toBe('spz');
	});

	it('ignores a declared format the viewer cannot read and falls back to the extension', () => {
		expect(resolveSplatFormat({ splat_format: 'glb', url: '/api/splat/a.ply' })).toBe('ply');
	});

	it('is null when neither the server nor the name says what the container is', () => {
		expect(resolveSplatFormat({ url: '/api/splat/no-extension' })).toBeNull();
		expect(resolveSplatFormat({ url: '/api/splat/model.glb' })).toBeNull();
		expect(resolveSplatFormat(null)).toBeNull();
	});
});
