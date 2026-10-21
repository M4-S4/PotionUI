import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
	clearProbeCache,
	deriveMediaMetadata,
	probeContentLength,
	shouldProbeSize,
	withProbedSize
} from './mediaMetadata';

describe('deriveMediaMetadata video', () => {
	it('prefers file metadata, then the gallery item, then video_metadata', () => {
		const file = { width: 640, height: 360, file_size: 1536 * 1024 };
		const item = { resolution: [1280, 720] };
		const generation = { video_metadata: { resolution: [1920, 1080] } };
		expect(deriveMediaMetadata('video', file, item, generation, '/a.mp4')).toMatchObject({ width: 640, height: 360 });
		expect(deriveMediaMetadata('video', null, item, generation, '/a.mp4')).toMatchObject({ width: 1280, height: 720 });
		expect(deriveMediaMetadata('video', null, null, generation, '/a.mp4')).toMatchObject({ width: 1920, height: 1080 });
		expect(deriveMediaMetadata('video', null, null, null, '/a.mp4')).toMatchObject({ width: 0, height: 0 });
	});

	it('formats the size with the shared byte formatter', () => {
		const meta = deriveMediaMetadata('video', { file_size: 1536 * 1024 }, null, null, '/a.mp4');
		expect(meta?.fileSize).toBe('1.5 MB');
		expect(deriveMediaMetadata('video', {}, null, null, '/a.mp4')?.fileSize).toBe('Unknown');
	});

	it('derives the container from the url', () => {
		expect(deriveMediaMetadata('video', null, null, null, '/a.webm')?.format).toBe('WEBM');
		expect(deriveMediaMetadata('video', null, null, null, '/a.mov')?.format).toBe('MOV');
		expect(deriveMediaMetadata('video', null, null, null, '/a.avi')?.format).toBe('AVI');
		expect(deriveMediaMetadata('video', null, null, null, '/a.mp4')?.format).toBe('MP4');
		expect(deriveMediaMetadata('video', null, null, null)?.format).toBe('MP4');
	});
});

describe('deriveMediaMetadata image', () => {
	it('uses file dimensions with the mime subtype or extension as format', () => {
		const byMime = deriveMediaMetadata('image', { width: 10, height: 20, mime_type: 'image/webp', file_size: 512 }, null, null);
		expect(byMime).toEqual({ width: 10, height: 20, fileSize: '512 B', format: 'WEBP' });
		const byPath = deriveMediaMetadata('image', { width: 10, height: 20, file_path: 'out/x.jpg' }, null, null);
		expect(byPath?.format).toBe('JPG');
	});

	it('falls back to the item resolution with an unknown size', () => {
		expect(deriveMediaMetadata('image', null, { resolution: [512, 768] }, null)).toEqual({
			width: 512,
			height: 768,
			fileSize: 'Unknown',
			format: 'PNG'
		});
	});

	it('returns null when nothing is known', () => {
		expect(deriveMediaMetadata('image', null, null, null)).toBeNull();
		expect(deriveMediaMetadata('image', { width: 10 }, { resolution: [1] }, null)).toBeNull();
	});
});

describe('shouldProbeSize', () => {
	const unknown = { width: 1, height: 1, fileSize: 'Unknown', format: 'PNG' };
	const base = { kind: 'video' as const, filesReady: true, fileMetadata: null, metadata: unknown, url: '/api/x.mp4' };

	it('probes an unknown size on a network url once files have settled', () => {
		expect(shouldProbeSize(base)).toBe(true);
	});

	it('waits while the generation files are still being fetched', () => {
		expect(shouldProbeSize({ ...base, filesReady: false })).toBe(false);
	});

	it('does not probe when the size is already known', () => {
		expect(shouldProbeSize({ ...base, metadata: { ...unknown, fileSize: '1 MB' } })).toBe(false);
	});

	it('never probes blob, data or missing urls or missing metadata', () => {
		expect(shouldProbeSize({ ...base, url: 'blob:abc' })).toBe(false);
		expect(shouldProbeSize({ ...base, url: 'data:image/png;base64,AA' })).toBe(false);
		expect(shouldProbeSize({ ...base, url: null })).toBe(false);
		expect(shouldProbeSize({ ...base, metadata: null })).toBe(false);
	});

	it('skips images whose file record already has dimensions but probes videos', () => {
		const fileMetadata = { width: 10, height: 20 };
		expect(shouldProbeSize({ ...base, kind: 'image', fileMetadata })).toBe(false);
		expect(shouldProbeSize({ ...base, kind: 'image', fileMetadata: null })).toBe(true);
		expect(shouldProbeSize({ ...base, kind: 'video', fileMetadata })).toBe(true);
	});
});

describe('probeContentLength', () => {
	const fetchMock = vi.fn();
	beforeEach(() => {
		clearProbeCache();
		fetchMock.mockReset();
		vi.stubGlobal('fetch', fetchMock);
	});
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('issues one HEAD per url and reads content-length', async () => {
		fetchMock.mockResolvedValue({ headers: new Headers({ 'content-length': '2048' }) });
		expect(await probeContentLength('/a.mp4')).toBe(2048);
		expect(await probeContentLength('/a.mp4')).toBe(2048);
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(fetchMock).toHaveBeenCalledWith('/a.mp4', { method: 'HEAD' });
	});

	it('returns null without a usable length or on failure', async () => {
		fetchMock.mockResolvedValueOnce({ headers: new Headers() });
		expect(await probeContentLength('/b.mp4')).toBeNull();
		fetchMock.mockRejectedValueOnce(new Error('offline'));
		expect(await probeContentLength('/c.mp4')).toBeNull();
	});
});

describe('withProbedSize', () => {
	it('replaces only the size', () => {
		expect(withProbedSize({ width: 1, height: 2, fileSize: 'Unknown', format: 'MP4' }, 1024 * 1024)).toEqual({
			width: 1,
			height: 2,
			fileSize: '1 MB',
			format: 'MP4'
		});
	});
});
