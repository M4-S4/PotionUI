import { describe, expect, it } from 'vitest';
import type { Segment } from '$lib/types/segments';
import { segmentUsageNumbers } from './variableUsageNumbering';

const seg = (content: string, extra: Partial<Segment> = {}): Segment => ({ id: content, content, ...extra });

describe('segmentUsageNumbers', () => {
	it('numbers content chips across segments', () => {
		const result = segmentUsageNumbers([seg('${x} a'), seg('b ${x}')]);
		expect(result.map((r) => r.map((u) => `${u.ordinal}/${u.total}`))).toEqual([['1/2'], ['2/2']]);
	});

	it('lets a prefix usage consume an ordinal before the content chip', () => {
		const result = segmentUsageNumbers([seg('body ${x}', { prefix: '${x} ' })]);
		expect(result[0]).toEqual([{ name: 'x', ordinal: 2, total: 2 }]);
	});

	it('lets a suffix usage consume an ordinal after the content chip', () => {
		const result = segmentUsageNumbers([seg('${x} body', { suffix: ' ${x}' })]);
		expect(result[0]).toEqual([{ name: 'x', ordinal: 1, total: 2 }]);
	});

	it('drops the affixes of an empty-body segment', () => {
		const result = segmentUsageNumbers([seg('  ', { prefix: '${x} ' }), seg('${x}')]);
		expect(result[1]).toEqual([{ name: 'x', ordinal: 1, total: 1 }]);
	});

	it('skips disabled segments', () => {
		const result = segmentUsageNumbers([seg('${x}', { enabled: false }), seg('${x}')]);
		expect(result[0]).toEqual([]);
		expect(result[1]).toEqual([{ name: 'x', ordinal: 1, total: 1 }]);
	});

	it('counts a ${name} inside a chip value but does not map it to a content chip', () => {
		const chips = {
			c1: { categoryPath: 'mood', value: 'wearing ${x}', label: 'l', valueId: 'v', allValues: [], shuffle: false }
		} as unknown as Segment['chips'];
		const result = segmentUsageNumbers([seg('#mood and ${x}', { chips })]);
		expect(result[0]).toEqual([{ name: 'x', ordinal: 2, total: 2 }]);
	});

	it('numbers negative segments after the positive ones', () => {
		const result = segmentUsageNumbers([seg('${x}')], [seg('${x}')]);
		expect(result[1]).toEqual([{ name: 'x', ordinal: 2, total: 2 }]);
	});
});
