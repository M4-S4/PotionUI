import { describe, expect, it } from 'vitest';
import { BAR_DIVIDER, BAR_GAP, BAR_SPACER, foldOrder, foldedIds, type BarLayoutInput } from './barLayout';

const widths = { edit: 78, effects: 94, favorite: 80, tags: 66, reuse: 118, copy: 124, download: 100, more: 62 };

function input(available: number, extra: Partial<BarLayoutInput> = {}): BarLayoutInput {
	return {
		available,
		widths,
		favorite: true,
		tags: true,
		pageIds: ['reuse', 'copy'],
		download: true,
		effects: true,
		edit: true,
		...extra
	};
}

const total = (ids: number[]) => ids.reduce((sum, width) => sum + width, 0) + BAR_GAP * (ids.length - 1);

describe('foldOrder', () => {
	it('folds Download, then page actions from the right, then Tags, then Favorite', () => {
		expect(foldOrder(input(0))).toEqual(['download', 'copy', 'reuse', 'tags', 'favorite']);
	});

	it('leaves out what the bar does not have', () => {
		expect(foldOrder(input(0, { download: false, tags: false, pageIds: ['auto'] }))).toEqual(['auto', 'favorite']);
	});
});

describe('foldedIds', () => {
	it('folds nothing while the width is unknown', () => {
		expect(foldedIds(input(0)).size).toBe(0);
	});

	it('folds nothing when everything fits exactly', () => {
		const all = [78, 94, BAR_DIVIDER, 80, 66, BAR_DIVIDER, 118, 124, BAR_SPACER, 100, 62];
		expect(foldedIds(input(total(all))).size).toBe(0);
		expect([...foldedIds(input(total(all) - 1))]).toEqual(['download']);
	});

	it('folds one more button at each step down', () => {
		const steps = [700, 580, 454, 374, 304].map((available) => [...foldedIds(input(available))]);
		expect(steps).toEqual([
			['download'],
			['download', 'copy'],
			['download', 'copy', 'reuse'],
			['download', 'copy', 'reuse', 'tags'],
			['download', 'copy', 'reuse', 'tags', 'favorite']
		]);
	});

	it('never folds Edit, Effects or More', () => {
		const folded = foldedIds(input(10));
		expect([...folded].sort()).toEqual(['copy', 'download', 'favorite', 'reuse', 'tags']);
	});

	it('treats a button with no measured width as zero wide', () => {
		expect(foldedIds({ ...input(300), widths: {} }).size).toBe(0);
	});
});
