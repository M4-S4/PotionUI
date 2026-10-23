import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const { default: SubcollectionFilters } = await import(
	'../../src/lib/components/collections/SubcollectionFilters.svelte'
);
const { resetLayerStackForTests } = await import('../../src/lib/actions/layerStack');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const collections = [
	{ id: 'potions', parent_id: null, name: 'Potion studies', item_count: 42 },
	{ id: 'glass', parent_id: 'potions', name: 'Blue glass', item_count: 18 }
];

const filtersButton = () =>
	Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.includes('Filters'));
const includeBox = () =>
	document.querySelector<HTMLInputElement>('[data-testid="include-subcollections"] input[type="checkbox"]');
const clearAll = () =>
	Array.from(document.querySelectorAll<HTMLButtonElement>('[aria-label="Advanced filters"] button')).find(
		(b) => b.textContent?.trim() === 'Clear all'
	)!;

function mountFilters(overrides: Record<string, unknown> = {}) {
	const props = { collections, activeId: 'potions', directOnly: false, onDirectOnlyChange: vi.fn(), ...overrides };
	const { rerender } = mountComponent(SubcollectionFilters as never, props);
	return { props, rerender };
}

beforeEach(() => {
	resetLayerStackForTests();
});

afterEach(() => {
	document.body.innerHTML = '';
});

describe('subcollection filters', () => {
	it('shows the Filters button only for a collection with subcollections', () => {
		mountFilters({ activeId: 'glass' });
		expect(filtersButton()).toBeUndefined();
		document.body.innerHTML = '';
		mountFilters({ activeId: undefined });
		expect(filtersButton()).toBeUndefined();
		document.body.innerHTML = '';
		mountFilters({ activeId: 'potions' });
		expect(filtersButton()).toBeDefined();
	});

	it('holds the include row, which reports direct only when unchecked', async () => {
		const { props } = mountFilters();
		filtersButton()!.click();
		await settle();
		expect(includeBox()!.checked).toBe(true);
		includeBox()!.click();
		await settle();
		expect(props.onDirectOnlyChange).toHaveBeenCalledWith(true);
	});

	it('counts excluded subcollections in the badge and Clear all includes them again', async () => {
		const { props, rerender } = mountFilters();
		expect(filtersButton()!.textContent?.trim()).toBe('Filters');
		filtersButton()!.click();
		await settle();
		expect(clearAll().disabled).toBe(true);
		rerender({ directOnly: true });
		await settle();
		expect(filtersButton()!.textContent?.replace(/\s+/g, ' ').trim()).toBe('Filters 1');
		expect(clearAll().disabled).toBe(false);
		clearAll().click();
		await settle();
		expect(props.onDirectOnlyChange).toHaveBeenLastCalledWith(false);
	});
});
