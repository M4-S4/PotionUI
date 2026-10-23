import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

vi.mock('$lib/stores/confirm', () => ({ confirmDialog: vi.fn() }));

const { default: IncludeSubcollections } = await import(
	'../../src/lib/components/collections/IncludeSubcollections.svelte'
);
const { default: CollectionActionsMenu } = await import(
	'../../src/lib/components/collections/CollectionActionsMenu.svelte'
);
const { resetLayerStackForTests } = await import('../../src/lib/actions/layerStack');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const collections = [
	{ id: 'potions', parent_id: null, name: 'Potion studies', item_count: 42 },
	{ id: 'glass', parent_id: 'potions', name: 'Blue glass', item_count: 18 },
	{ id: 'deep', parent_id: 'glass', name: 'Deep blue', item_count: 3 },
	{ id: 'pets', parent_id: null, name: 'Pets', item_count: 1 }
];

const toggle = (target: HTMLElement) =>
	target.querySelector<HTMLInputElement>('[data-testid="include-subcollections"] input[type="checkbox"]');

beforeEach(() => {
	resetLayerStackForTests();
});

afterEach(() => {
	document.body.innerHTML = '';
});

describe('Include subcollections', () => {
	function mountToggle(overrides: Record<string, unknown> = {}) {
		const props = { collections, activeId: 'glass', directOnly: false, onDirectOnlyChange: vi.fn(), ...overrides };
		const { target, rerender } = mountComponent(IncludeSubcollections as never, props);
		return { target, props, rerender };
	}

	it('appears only for a selected collection that has subcollections', () => {
		expect(toggle(mountToggle({ activeId: 'pets' }).target)).toBeNull();
		expect(toggle(mountToggle({ activeId: undefined }).target)).toBeNull();
		const { target } = mountToggle({ activeId: 'glass' });
		expect(toggle(target)).not.toBeNull();
		expect(target.textContent).toContain('Include subcollections');
	});

	it('is checked while direct only is off and reports the inverse when changed', async () => {
		const { target, props, rerender } = mountToggle();
		const box = toggle(target)!;
		expect(box.checked).toBe(true);
		box.click();
		await settle();
		expect(props.onDirectOnlyChange).toHaveBeenLastCalledWith(true);
		rerender({ directOnly: true });
		expect(box.checked).toBe(false);
		box.click();
		await settle();
		expect(props.onDirectOnlyChange).toHaveBeenLastCalledWith(false);
	});
});

describe('toolbar collection actions', () => {
	function mountActions() {
		const treeActions = {
			onSelect: vi.fn(),
			onRename: vi.fn().mockResolvedValue({ success: true }),
			onCreate: vi.fn().mockResolvedValue({ success: true }),
			onDelete: vi.fn().mockResolvedValue({ success: true }),
			onMove: vi.fn().mockResolvedValue({ success: true })
		};
		const { target } = mountComponent(CollectionActionsMenu as never, {
			variant: 'toolbar',
			item: collections[1],
			collections,
			treeActions
		});
		return { target, treeActions };
	}

	it('is an icon button that opens the menu for the selected collection', async () => {
		const { target } = mountActions();
		const trigger = target.querySelector<HTMLButtonElement>('button[aria-label="Collection actions"]')!;
		expect(trigger.querySelector('svg')).not.toBeNull();
		trigger.click();
		await settle();
		const menu = document.querySelector<HTMLElement>('[role="menu"]')!;
		expect(menu.querySelector('[data-testid="collection-menu-title"]')?.textContent?.trim()).toBe('Blue glass');
		expect(Array.from(menu.querySelectorAll('[role="menuitem"]')).map((i) => i.textContent?.trim())).toEqual([
			'New subcollection',
			'Rename…',
			'Move to…',
			'Delete collection…'
		]);
	});

	it('offers only destinations outside the collection and its current parent', async () => {
		const { target } = mountActions();
		target.querySelector<HTMLButtonElement>('button[aria-label="Collection actions"]')!.click();
		await settle();
		Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitem"]'))
			.find((el) => el.textContent?.trim() === 'Move to…')!
			.click();
		await settle();
		const options = Array.from(document.querySelectorAll<HTMLOptionElement>('[role="dialog"] select option')).map(
			(o) => o.value
		);
		expect(options).toEqual(['', 'pets']);
	});
});
