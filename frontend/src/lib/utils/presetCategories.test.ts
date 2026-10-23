import { describe, it, expect } from 'vitest';
import { PRESET_CATEGORIES, presetCategoryLabel, presetCategoryOrder } from './presetCategories';

describe('presetCategoryLabel', () => {
	it('shows the utility category as Tools', () => {
		expect(presetCategoryLabel('utility')).toBe('Tools');
	});

	it('labels the media categories', () => {
		expect(presetCategoryLabel('image')).toBe('Image');
		expect(presetCategoryLabel('3d')).toBe('3D');
	});

	it('falls back to a capitalised id for unknown categories and Uncategorized for none', () => {
		expect(presetCategoryLabel('other')).toBe('Other');
		expect(presetCategoryLabel(null)).toBe('Uncategorized');
	});
});

describe('presetCategoryOrder', () => {
	it('places Tools after the media categories', () => {
		const ids = PRESET_CATEGORIES.map((category) => category.id);
		expect(ids.at(-1)).toBe('utility');
		expect(presetCategoryOrder('utility')).toBeGreaterThan(presetCategoryOrder('3d'));
	});
});
