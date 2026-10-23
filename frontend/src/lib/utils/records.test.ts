import { describe, it, expect } from 'vitest';
import { isRecord, mergeRawCapabilities } from './records';

describe('isRecord', () => {
	it('accepts plain objects only', () => {
		expect(isRecord({})).toBe(true);
		expect(isRecord([])).toBe(false);
		expect(isRecord(null)).toBe(false);
		expect(isRecord('x')).toBe(false);
	});
});

describe('mergeRawCapabilities', () => {
	it('overrides top-level keys and merges modes one level deep', () => {
		const merged = mergeRawCapabilities(
			{ limits: { a: 1 }, modes: { t2v: { max: 5, fps: 24 }, i2v: { max: 3 } } },
			{ limits: { b: 2 }, modes: { t2v: { max: 8 }, v2v: { max: 1 } } }
		);
		expect(merged).toEqual({
			limits: { b: 2 },
			modes: { t2v: { max: 8, fps: 24 }, i2v: { max: 3 }, v2v: { max: 1 } }
		});
	});

	it('replaces a non-record mode entry instead of merging it', () => {
		const merged = mergeRawCapabilities({ modes: { t2v: { max: 5 } } }, { modes: { t2v: false } });
		expect(merged.modes).toEqual({ t2v: false });
	});

	it('does not mutate its inputs', () => {
		const base = { modes: { t2v: { max: 5 } } };
		mergeRawCapabilities(base, { modes: { t2v: { max: 9 } } });
		expect(base).toEqual({ modes: { t2v: { max: 5 } } });
	});
});
