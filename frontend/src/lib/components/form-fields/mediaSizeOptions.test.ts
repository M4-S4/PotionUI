import { describe, expect, it, vi } from 'vitest';
import {
	deriveMediaSizeOptions,
	matchingSide,
	mediaValueKey,
	preferredMediaSize,
	resolveMediaSize,
	type MediaSizeDeps
} from './mediaSizeOptions';
import { parseCustomResolution } from './resolutionPicker';

const rows = (width: number, height: number) =>
	deriveMediaSizeOptions({ width, height }).map((opt) => [opt.description, opt.value, opt.tier]);

describe('deriveMediaSizeOptions', () => {
	it('offers the original, smaller common heights and scale steps at the source aspect, deduplicated', () => {
		expect(rows(1920, 1080)).toEqual([
			['Original', '1920x1080', 'Source'],
			['720p', '1280x720', 'Common sizes'],
			['540p', '960x540', 'Common sizes'],
			['480p', '854x480', 'Common sizes'],
			['360p', '640x360', 'Common sizes'],
			['75%', '1440x810', 'Scale'],
			['33%', '634x356', 'Scale'],
			['25%', '480x270', 'Scale']
		]);
	});

	it('measures common heights on the short edge so portrait videos stay portrait', () => {
		const options = deriveMediaSizeOptions({ width: 1080, height: 1920 });
		expect(options.find((opt) => opt.description === '720p')?.value).toBe('720x1280');
		expect(options.every((opt) => opt.ratio.join(':') === '9:16')).toBe(true);
	});

	it('never offers a common height at or above the source', () => {
		expect(rows(640, 360).map(([label]) => label)).toEqual(['Original', '75%', '50%', '33%', '25%']);
	});

	it('keeps every side even and drops sizes below the 64px minimum', () => {
		expect(rows(641, 361)[0]).toEqual(['Original', '640x360', 'Source']);
		const values = deriveMediaSizeOptions({ width: 100, height: 80 }).map((opt) => opt.value);
		expect(values).toEqual(['100x80']);
		for (const opt of deriveMediaSizeOptions({ width: 1279, height: 721 })) {
			const [w, h] = opt.value.split('x').map(Number);
			expect(w % 2 + (h % 2)).toBe(0);
		}
	});

	it('has nothing to offer without a known size', () => {
		expect(deriveMediaSizeOptions(null)).toEqual([]);
		expect(deriveMediaSizeOptions({ width: 0, height: 720 })).toEqual([]);
	});
});

describe('preferredMediaSize', () => {
	it('picks the largest common height below the source, else half size', () => {
		expect(preferredMediaSize(deriveMediaSizeOptions({ width: 3840, height: 2160 }))).toBe('1920x1080');
		expect(preferredMediaSize(deriveMediaSizeOptions({ width: 640, height: 360 }))).toBe('320x180');
		expect(preferredMediaSize([])).toBeNull();
	});
});

describe('matchingSide', () => {
	it('fills the other side from the source aspect, rounded to even', () => {
		expect(matchingSide({ width: 1920, height: 1080 }, 'width', '1000')).toBe('562');
		expect(matchingSide({ width: 1920, height: 1080 }, 'height', '720')).toBe('1280');
		expect(matchingSide({ width: 1920, height: 1080 }, 'width', '')).toBe('');
		expect(matchingSide({ width: 1920, height: 1080 }, 'width', 'abc')).toBe('');
	});
});

describe('parseCustomResolution with an even step', () => {
	it('accepts even sizes that are not multiples of 8 and rejects odd ones', () => {
		expect(parseCustomResolution('854', '480', 2)).toEqual({ ok: true, value: '854x480' });
		expect(parseCustomResolution('855', '480', 2).ok).toBe(false);
		expect(parseCustomResolution('854', '480').ok).toBe(false);
	});
});

describe('resolveMediaSize', () => {
	const deps = (overrides: Partial<MediaSizeDeps> = {}): MediaSizeDeps => ({
		metadata: {
			listGenerationMedia: vi.fn(async () => ({ success: false })),
			getUploadInfo: vi.fn(async () => ({ success: true, data: { width: 1280, height: 720 } }))
		},
		probe: vi.fn(async () => ({ width: 320, height: 240 })),
		...overrides
	});

	it('reads the size the media value already carries', async () => {
		const d = deps();
		const size = await resolveMediaSize(
			{ path: '/x/uploads/a.mp4', url: '/api/media/uploads/a.mp4', metadata: { width: 1920, height: 1080 } },
			d
		);
		expect(size).toEqual({ width: 1920, height: 1080 });
		expect(d.metadata.getUploadInfo).not.toHaveBeenCalled();
	});

	it('asks the server for an upload path without metadata', async () => {
		expect(await resolveMediaSize('uploads/a.mp4', deps())).toEqual({ width: 1280, height: 720 });
	});

	it('probes the file itself when the server has no size', async () => {
		const d = deps({
			metadata: {
				listGenerationMedia: vi.fn(async () => ({ success: false })),
				getUploadInfo: vi.fn(async () => {
					throw new Error('offline');
				})
			}
		});
		expect(await resolveMediaSize('uploads/a.mp4', d)).toEqual({ width: 320, height: 240 });
		expect(d.probe).toHaveBeenCalledWith('/api/media/uploads/a.mp4', 'video');
	});

	it('has no size and no key without a value', async () => {
		expect(await resolveMediaSize(null, deps())).toBeNull();
		expect(mediaValueKey(null)).toBeNull();
		expect(mediaValueKey([{ relative_path: 'uploads/a.mp4' }])).toBe('uploads/a.mp4');
	});
});
