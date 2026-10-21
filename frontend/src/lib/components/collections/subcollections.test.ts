import { describe, it, expect } from 'vitest';
import { excludesSubcollections, hasSubcollections } from './subcollections';

const collections = [
	{ id: 'potions', parent_id: null, name: 'Potion studies', item_count: 42 },
	{ id: 'glass', parent_id: 'potions', name: 'Blue glass', item_count: 18 },
	{ id: 'amber', parent_id: 'potions', name: 'Amber & violet', item_count: 12 }
];

describe('hasSubcollections', () => {
	it('is true only for a collection with children', () => {
		expect(hasSubcollections(collections, 'potions')).toBe(true);
		expect(hasSubcollections(collections, 'glass')).toBe(false);
		expect(hasSubcollections(collections, undefined)).toBe(false);
	});
});

describe('excludesSubcollections', () => {
	it('counts as a filter only when direct only is on and the collection has children', () => {
		expect(excludesSubcollections(collections, 'potions', true)).toBe(true);
		expect(excludesSubcollections(collections, 'potions', false)).toBe(false);
		expect(excludesSubcollections(collections, 'glass', true)).toBe(false);
		expect(excludesSubcollections(collections, undefined, true)).toBe(false);
	});
});
