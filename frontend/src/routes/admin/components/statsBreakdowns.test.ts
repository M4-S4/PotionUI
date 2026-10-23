import { describe, it, expect } from 'vitest';
import { BREAKDOWNS, BREAKDOWN_BY_SECTION } from './statsBreakdowns';
import { statsRowLimits } from '$lib/stores/statsRowLimits';
import { get } from 'svelte/store';

describe('BREAKDOWNS', () => {
	it('has eight breakdowns with unique sections and dimensions', () => {
		expect(BREAKDOWNS).toHaveLength(8);
		expect(new Set(BREAKDOWNS.map((b) => b.section)).size).toBe(8);
		expect(new Set(BREAKDOWNS.map((b) => b.dimension)).size).toBe(8);
	});

	it('maps every section to a row limit in the store', () => {
		const limits = get(statsRowLimits);
		for (const b of BREAKDOWNS) expect(limits[b.section]).toBeGreaterThan(0);
	});

	it('colours only the models card by group and gives it the Type column', () => {
		const grouped = BREAKDOWNS.filter((b) => b.colorBy === 'group');
		expect(grouped.map((b) => b.section)).toEqual(['models']);
		expect(BREAKDOWN_BY_SECTION.models.groupLabel).toBe('Type');
		expect(BREAKDOWNS.filter((b) => b.groupLabel)).toHaveLength(1);
	});

	it('rounds activity values and leaves parameter values raw', () => {
		for (const b of BREAKDOWNS) {
			expect(Boolean(b.valueFormat)).toBe(b.group === 'activity');
		}
		expect(BREAKDOWN_BY_SECTION.presets.valueFormat?.(1234.4)).toBe('1,234');
	});
});
