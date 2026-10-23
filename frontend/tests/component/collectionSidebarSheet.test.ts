import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

vi.mock('../../src/lib/services/api/index', () => ({
	api: { getClient: () => ({ get: vi.fn().mockRejectedValue({ response: { status: 404 } }) }) }
}));

vi.mock('$lib/stores/confirm', () => ({ confirmDialog: vi.fn() }));

const { default: Fixture } = await import('./fixtures/CollectionSheetFixture.svelte');
const { resetLayerStackForTests } = await import('../../src/lib/actions/layerStack');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const collections = [
	{ id: 'a', parent_id: null, name: 'Avatars', item_count: 5 },
	{ id: 'b', parent_id: null, name: 'Pets', item_count: 1 }
];

function view(id: string, label: string, active = false) {
	return { id, icon: 'folder', label, active, count: 9, onSelect: vi.fn() };
}

function sidebarProps() {
	return {
		storageKey: 'sheet-test',
		label: 'History',
		collections,
		activeId: undefined,
		allView: view('all', 'All generations', true),
		favoritesView: view('favorites', 'Favorites'),
		unsortedView: view('unsorted', 'Unsorted'),
		treeActions: {
			onSelect: vi.fn(),
			onRename: vi.fn().mockResolvedValue({ success: true }),
			onCreate: vi.fn().mockResolvedValue({ success: true }),
			onDelete: vi.fn().mockResolvedValue({ success: true }),
			onMove: vi.fn().mockResolvedValue({ success: true })
		},
		onCreateRoot: vi.fn().mockResolvedValue({ success: true }),
		onCollapse: vi.fn()
	};
}

const sheet = () => document.querySelector<HTMLElement>('[role="dialog"][aria-label="History collections"]');
const trigger = () => document.querySelector<HTMLButtonElement>('button[aria-label="Show collections"]')!;

async function openSheet() {
	trigger().focus();
	trigger().click();
	await settle();
	await new Promise((resolve) => setTimeout(resolve, 40));
	return sheet()!;
}

const originalMatchMedia = window.matchMedia;

beforeEach(() => {
	localStorage.clear();
	resetLayerStackForTests();
});

afterEach(() => {
	window.matchMedia = originalMatchMedia;
	document.body.innerHTML = '';
	document.body.style.overflow = '';
});

describe('collections sidebar sheet', () => {
	it('opens the same sidebar in a modal sheet with a close control', async () => {
		mountComponent(Fixture as never, { sidebar: sidebarProps() });
		expect(trigger().className).toContain('md:hidden');
		expect(sheet()).toBeNull();
		const panel = await openSheet();
		expect(panel.getAttribute('aria-modal')).toBe('true');
		expect(panel.querySelector('button[aria-label="Close collections"]')).not.toBeNull();
		expect(panel.querySelector('button[aria-label="Collapse collections"]')).toBeNull();
		expect(panel.textContent).toContain('All generations');
		expect(panel.textContent).toContain('Avatars');
		expect(document.body.style.overflow).toBe('hidden');
	});

	it('closes after a view is picked', async () => {
		const sidebar = sidebarProps();
		mountComponent(Fixture as never, { sidebar });
		const panel = await openSheet();
		const unsorted = Array.from(panel.querySelectorAll<HTMLElement>('[role="option"]')).find((r) =>
			r.textContent?.includes('Unsorted')
		)!;
		unsorted.click();
		await settle();
		expect(sidebar.unsortedView.onSelect).toHaveBeenCalledTimes(1);
		expect(sheet()).toBeNull();
	});

	it('closes after a collection is picked', async () => {
		const sidebar = sidebarProps();
		mountComponent(Fixture as never, { sidebar });
		const panel = await openSheet();
		Array.from(panel.querySelectorAll<HTMLElement>('[role="treeitem"]'))
			.find((r) => r.textContent?.includes('Pets'))!
			.click();
		await settle();
		expect(sidebar.treeActions.onSelect).toHaveBeenCalledWith('b');
		expect(sheet()).toBeNull();
	});

	it('closes on Escape and gives focus back to the trigger', async () => {
		mountComponent(Fixture as never, { sidebar: sidebarProps() });
		const panel = await openSheet();
		expect(panel.contains(document.activeElement)).toBe(true);
		panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await settle();
		expect(sheet()).toBeNull();
		expect(document.activeElement).toBe(trigger());
	});

	it('closes from the backdrop but not from a click inside the panel', async () => {
		mountComponent(Fixture as never, { sidebar: sidebarProps() });
		const panel = await openSheet();
		panel.click();
		await settle();
		expect(sheet()).not.toBeNull();
		(panel.parentElement as HTMLElement).click();
		await settle();
		expect(sheet()).toBeNull();
	});

	it('keeps the sheet open while a row menu opens a dialog', async () => {
		mountComponent(Fixture as never, { sidebar: sidebarProps() });
		const panel = await openSheet();
		const actions = panel.querySelector<HTMLButtonElement>('button[aria-label="Avatars actions"]')!;
		actions.focus();
		actions.click();
		await settle();
		expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull();
		Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitem"]'))
			.find((el) => el.textContent?.trim() === 'Rename…')!
			.click();
		await settle();
		const rename = document.querySelector<HTMLInputElement>('input[id^="collection-name-"]');
		expect(rename?.value).toBe('Avatars');
		expect(sheet()).not.toBeNull();
	});

	it('draws the show control differently from the collapse control', async () => {
		mountComponent(Fixture as never, { sidebar: sidebarProps(), collapsed: true });
		const { default: CollectionLibrarySidebar } = await import(
			'../../src/lib/components/collections/CollectionLibrarySidebar.svelte'
		);
		mountComponent(CollectionLibrarySidebar as never, sidebarProps());
		await settle();
		const show = trigger();
		const collapse = document.querySelector<HTMLButtonElement>('button[aria-label="Collapse collections"]')!;
		const glyph = (button: HTMLButtonElement) =>
			Array.from(button.querySelectorAll('svg path')).map((path) => path.getAttribute('d'));
		expect(show.getAttribute('aria-label')).not.toBe(collapse.getAttribute('aria-label'));
		expect(glyph(show)).toHaveLength(3);
		expect(glyph(collapse)).toHaveLength(3);
		expect(glyph(show)).not.toEqual(glyph(collapse));
	});

	it('expands the desktop sidebar instead when it is collapsed on a wide screen', async () => {
		window.matchMedia = ((query: string) => ({
			matches: query.includes('min-width: 768px'),
			media: query,
			onchange: null,
			addListener() {},
			removeListener() {},
			addEventListener() {},
			removeEventListener() {},
			dispatchEvent: () => false
		})) as typeof window.matchMedia;
		const onExpand = vi.fn();
		mountComponent(Fixture as never, { sidebar: sidebarProps(), collapsed: true, onExpand });
		expect(trigger().className).not.toContain('md:hidden');
		trigger().click();
		await settle();
		expect(onExpand).toHaveBeenCalledTimes(1);
		expect(sheet()).toBeNull();
	});
});
