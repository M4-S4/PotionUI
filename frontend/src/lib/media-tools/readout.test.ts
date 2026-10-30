import { describe, it, expect } from 'vitest';
import { computeReadout, formatReadoutTime } from './readout';
import type { MediaToolSource } from './types';

const clip: MediaToolSource = {
	id: 'a',
	kind: 'video',
	url: '/a.mp4',
	filename: 'a.mp4',
	width: 1920,
	height: 1080,
	durationSeconds: 8,
	fps: 24
};

const still: MediaToolSource = { id: 'b', kind: 'image', url: '/b.png', filename: 'b.png', width: 1254, height: 1254 };

const tool = (id: string, cardinality: 'one_to_one' | 'one_to_many' | 'many_to_one' = 'one_to_one') => ({
	id,
	cardinality,
	produces: 'same' as const
});

describe('formatReadoutTime', () => {
	it('writes minutes, seconds and tenths', () => {
		expect(formatReadoutTime(4)).toBe('00:04.0');
		expect(formatReadoutTime(65.26)).toBe('01:05.3');
	});

	it('survives nonsense', () => {
		expect(formatReadoutTime(Number.NaN)).toBe('00:00.0');
		expect(formatReadoutTime(-3)).toBe('00:00.0');
	});
});

describe('computeReadout', () => {
	it('halves a clip at double speed and keeps its size', () => {
		expect(computeReadout(tool('speed'), { speed: 2 }, [clip])).toBe('1920×1080 · 00:04.0');
	});

	it('swaps the sides for a quarter turn and keeps them for a half turn', () => {
		expect(computeReadout(tool('rotate'), { degrees: 90 }, [still])).toBe('1254×1254');
		const wide = { ...still, width: 1920, height: 1080 };
		expect(computeReadout(tool('rotate'), { degrees: 90 }, [wide])).toBe('1080×1920');
		expect(computeReadout(tool('rotate'), { degrees: 180 }, [wide])).toBe('1920×1080');
		expect(computeReadout(tool('rotate'), { degrees: '270' }, [wide])).toBe('1080×1920');
	});

	it('works out the missing side of a resize from the source shape', () => {
		expect(computeReadout(tool('resize'), { width: 960 }, [clip])).toBe('960×540 · 00:08.0');
		expect(computeReadout(tool('resize'), { height: 540 }, [clip])).toBe('960×540 · 00:08.0');
	});

	it('rounds a video resize down to even sides', () => {
		expect(computeReadout(tool('resize'), { width: 961, height: 541 }, [clip])).toContain('960×540');
	});

	it('names the canvas size for fit canvas', () => {
		expect(computeReadout(tool('fit_canvas'), { width: 1024, height: 768 }, [still])).toBe('1024×768');
		expect(computeReadout(tool('fit_canvas'), {}, [still])).toBeNull();
	});

	it('says a color key produces transparency', () => {
		expect(computeReadout(tool('color_key'), {}, [still])).toBe('1254×1254 · PNG with transparency');
	});

	it('counts the clips and the joined length for a concat, minus the crossfade', () => {
		const second = { ...clip, id: 'c' };
		expect(computeReadout(tool('concat', 'many_to_one'), { overlap: 'cut' }, [clip, second])).toBe(
			'1 video from 2 clips · 00:16.0'
		);
		expect(
			computeReadout(tool('concat', 'many_to_one'), { overlap: 'crossfade', overlap_frames: 24 }, [clip, second])
		).toBe('1 video from 2 clips · 00:15.0');
	});

	it('counts the parts of a split', () => {
		const song = { ...clip, kind: 'audio' as const, durationSeconds: 65 };
		expect(computeReadout(tool('split', 'one_to_many'), { part_seconds: 10 }, [song])).toBe('7 parts of 00:10.0');
		expect(computeReadout(tool('split', 'one_to_many'), {}, [song])).toBeNull();
	});

	it('says what a frame grab makes', () => {
		expect(computeReadout(tool('frame'), { time: 1 }, [clip])).toBe('1 image · 1920×1080');
	});

	it('says what a batch does instead of guessing sizes', () => {
		expect(computeReadout(tool('speed'), { speed: 2 }, [clip, { ...clip, id: 'c' }])).toBe(
			'2 items, each saved as a new item'
		);
	});

	it('says nothing for a tool it cannot size, and nothing without a source', () => {
		expect(computeReadout(tool('acme.boomerang'), {}, [clip])).toBeNull();
		expect(computeReadout(tool('speed'), { speed: 2 }, [])).toBeNull();
	});
});
