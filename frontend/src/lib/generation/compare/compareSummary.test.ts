import { describe, expect, it } from 'vitest';
import { compareStatusLine } from './compareSummary';

describe('compareStatusLine', () => {
	it('joins the cell count and the estimate with a separator', () => {
		expect(compareStatusLine(4, 'about 1 min')).toBe('4 cells · about 1 min');
		expect(compareStatusLine(1, 'about 5 s')).toBe('1 cell · about 5 s');
	});

	it('drops the separator when there is no estimate', () => {
		expect(compareStatusLine(4, '-')).toBe('4 cells');
		expect(compareStatusLine(1, '')).toBe('1 cell');
	});
});
