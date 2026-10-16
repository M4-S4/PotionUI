import { describe, expect, it } from 'vitest';
import { fitToolbar, fixedChildWidth } from './toolbarFit';

describe('fixedChildWidth', () => {
	it('counts a shrinkable child at its minimum instead of its natural width', () => {
		expect(fixedChildWidth(320, 132)).toBe(132);
		expect(fixedChildWidth(90, 132)).toBe(90);
		expect(fixedChildWidth(320, null)).toBe(320);
	});
});

describe('fitToolbar with a wide prompt picker', () => {
	const widths = { prompts: 100, segments: 110, templates: 112, variables: 106 };
	const order = ['prompts', 'segments', 'templates', 'variables'];
	const collapseOrder = ['variables', 'templates', 'segments', 'prompts'];

	it('collapses actions into the menu when the room runs out, never the other way round', () => {
		const roomy = fitToolbar({ available: 600, gap: 8, order, collapseOrder, widths });
		const tight = fitToolbar({ available: 330, gap: 8, order, collapseOrder, widths });
		const tiny = fitToolbar({ available: 20, gap: 8, order, collapseOrder, widths });
		expect(roomy).toEqual([]);
		expect(tight.length).toBeGreaterThan(0);
		expect(tight.length).toBeLessThan(order.length);
		expect(tiny).toEqual(order);
	});

	it('measuring the picker at its shrink minimum leaves more room than its natural width', () => {
		const inner = 970;
		const others = 70 + 66;
		const gap = 8;
		const more = 28;
		const room = (picker: number) => inner - others - picker - gap * 3 - more - gap;
		const natural = fitToolbar({ available: room(320), gap, order, collapseOrder, widths });
		const shrunk = fitToolbar({ available: room(fixedChildWidth(320, 132)), gap, order, collapseOrder, widths });
		expect(shrunk.length).toBeLessThanOrEqual(natural.length);
	});
});
