import { describe, it, expect } from 'vitest';
import {
	formatEnergyKept,
	formatModelFileSizes,
	hasQualitySummary,
	modelFileHref,
	qualityNote,
	splitFileName,
	sizeReductionPercent
} from './modelFile';

const GB = 1024 ** 3;

describe('formatModelFileSizes', () => {
	it('shows before and after in one shared unit', () => {
		expect(formatModelFileSizes(23.8 * GB, 11.9 * GB)).toBe('23.8 → 11.9 GB');
	});

	it('picks the unit of the larger side so the smaller one is not rounded to zero', () => {
		expect(formatModelFileSizes(1.5 * GB, 0.2 * GB)).toBe('1.5 → 0.2 GB');
		expect(formatModelFileSizes(800 * 1024 ** 2, 400 * 1024 ** 2)).toBe('800.0 → 400.0 MB');
	});

	it('shows only the file size when the source size is unknown', () => {
		expect(formatModelFileSizes(null, 11.9 * GB)).toBe('11.9 GB');
		expect(formatModelFileSizes(undefined, 512)).toBe('512 B');
	});

	it('is empty when there is no size at all', () => {
		expect(formatModelFileSizes(null, null)).toBe('');
		expect(formatModelFileSizes(0, 0)).toBe('');
	});
});

describe('sizeReductionPercent', () => {
	it('rounds the saving', () => {
		expect(sizeReductionPercent(100, 50)).toBe(50);
		expect(sizeReductionPercent(3, 2)).toBe(33);
	});

	it('is null without both sizes', () => {
		expect(sizeReductionPercent(null, 5)).toBeNull();
		expect(sizeReductionPercent(5, 0)).toBeNull();
	});
});

describe('formatEnergyKept', () => {
	it('reads a fraction as a percentage', () => {
		expect(formatEnergyKept(0.987)).toBe('98.7%');
	});

	it('keeps a value already expressed as a percentage', () => {
		expect(formatEnergyKept(93)).toBe('93.0%');
	});

	it('is null when absent', () => {
		expect(formatEnergyKept(undefined)).toBeNull();
	});
});

describe('splitFileName', () => {
	it('splits at the last dot', () => {
		expect(splitFileName('a.b.safetensors')).toEqual({ stem: 'a.b', ext: '.safetensors' });
		expect(splitFileName('noext')).toEqual({ stem: 'noext', ext: '' });
		expect(splitFileName('.hidden')).toEqual({ stem: '.hidden', ext: '' });
	});
});

describe('modelFileHref', () => {
	it('links to the model page only once the file is indexed', () => {
		expect(modelFileHref({ model_id: 'abc/1' })).toBe('/models/abc%2F1');
		expect(modelFileHref({ model_id: null })).toBeNull();
	});
});

describe('hasQualitySummary', () => {
	it('counts the layers kept for their shape as part of the quality summary', () => {
		expect(hasQualitySummary({ layers_shape_kept: 0 })).toBe(true);
		expect(hasQualitySummary({ licence_note: 'mit' })).toBe(false);
		expect(hasQualitySummary(null)).toBe(false);
	});
});

describe('qualityNote', () => {
	it('returns the trimmed note a format attaches to its result', () => {
		expect(qualityNote({ quality_note: ' INT4 output quality is unverified until a GPU check ' })).toBe(
			'INT4 output quality is unverified until a GPU check'
		);
	});

	it('is empty when the summary carries no usable note', () => {
		expect(qualityNote({})).toBeNull();
		expect(qualityNote({ quality_note: '   ' })).toBeNull();
		expect(qualityNote(null)).toBeNull();
	});
});
