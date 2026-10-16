import { describe, expect, it } from 'vitest';
import { filterOptionIndices, showOptionFilter } from './variableOptionList';

describe('filterOptionIndices', () => {
	const texts = ['Red dress', 'blue coat', 'RED scarf', ''];

	it('returns every index for a blank query', () => {
		expect(filterOptionIndices(texts, '')).toEqual([0, 1, 2, 3]);
		expect(filterOptionIndices(texts, '   ')).toEqual([0, 1, 2, 3]);
	});

	it('matches case-insensitively and keeps original indices', () => {
		expect(filterOptionIndices(texts, 'red')).toEqual([0, 2]);
		expect(filterOptionIndices(texts, 'COAT')).toEqual([1]);
	});

	it('returns nothing when no option matches', () => {
		expect(filterOptionIndices(texts, 'zzz')).toEqual([]);
	});
});

describe('showOptionFilter', () => {
	it('appears only above six options', () => {
		expect(showOptionFilter(6)).toBe(false);
		expect(showOptionFilter(7)).toBe(true);
	});
});
