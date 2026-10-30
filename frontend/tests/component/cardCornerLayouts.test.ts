// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { createClassComponent } from 'svelte/legacy';

vi.stubGlobal('ClipboardItem', class {});
vi.stubGlobal(
	'ResizeObserver',
	class {
		observe() {}
		unobserve() {}
		disconnect() {}
	}
);

const { default: GenerationCard } = await import('$lib/components/GenerationCard.svelte');
const { default: LibraryCard } = await import('../../src/routes/library/components/LibraryCard.svelte');
const { default: JustifiedGenerationGallery } = await import('$lib/components/JustifiedGenerationGallery.svelte');
const { createHistoryEntryMenu, createLibraryEntryMenu } = await import('$lib/tools/entryMenu');
const { registerEntryTools } = await import('$lib/tools/entryTools');

registerEntryTools();

const RETIRED = ['View generation details', 'Delete generation', 'Delete library item', 'Add to favorites'];
const WIDTHS = [320, 220, 150, 120, 90];

function generation(id: string, favorite = false) {
	return {
		id,
		preset_id: 'p',
		form_data: {},
		status: 'completed',
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files: [
			{
				id: 1,
				file_type: 'image',
				is_final: true,
				file_path: `generations/${id}/1.png`,
				width: 640,
				height: 480,
				created_at: '2026-01-01T00:00:00Z'
			}
		],
		rating: 0,
		is_favorite: favorite
	} as never;
}

const ITEM = {
	id: 'item-1',
	filename: 'a.png',
	original_filename: 'a.png',
	media_type: 'image',
	url: '/a.png',
	width: 640,
	height: 480,
	tags: [],
	is_favorite: false
};

let destroy: (() => void) | undefined;

function mount(component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({ component: component as never, target, props });
	flushSync();
	destroy = () => {
		instance.$destroy();
		target.remove();
	};
	return target;
}

const labels = (root: ParentNode) =>
	Array.from(root.querySelectorAll('button')).map((button) => button.getAttribute('aria-label') ?? '');

const corner = (root: ParentNode) =>
	Array.from(root.querySelectorAll('button'))
		.filter((button) => button.closest('[data-entry-actions]') || /^(Select|Deselect) /.test(button.getAttribute('aria-label') ?? ''))
		.map((button) => button.getAttribute('aria-label') ?? '');

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
});

function historyMenu(gen: unknown) {
	return createHistoryEntryMenu({
		generations: () => [gen as never],
		selectedIds: () => [],
		host: { open: vi.fn(), remove: vi.fn(), favorite: vi.fn() },
		onTool: vi.fn()
	});
}

describe('History card corner at every tile width', () => {
	for (const width of WIDTHS) {
		it(`holds the checkbox, Favorite and the menu chip at ${width}px`, () => {
			const gen = generation('g1');
			const root = mount(GenerationCard, {
				generation: gen,
				tile: { width, height: 200 },
				entryMenu: historyMenu(gen),
				showActions: true,
				showCheckbox: true
			});

			const names = corner(root);
			expect(names).toEqual(['Select generation', 'Favorite', 'Item actions']);
			for (const retired of RETIRED) expect(names).not.toContain(retired);
		});
	}

	it('keeps the checkbox and the menu chip in selection mode', () => {
		const gen = generation('g1');
		const root = mount(GenerationCard, {
			generation: gen,
			tile: { width: 220, height: 200 },
			entryMenu: historyMenu(gen),
			showActions: false,
			selectable: true,
			showCheckbox: true
		});

		expect(corner(root)).toEqual(['Select generation', 'Item actions']);
	});

	it('shows no Favorite chip on a stack entry', () => {
		const gen = { ...(generation('g1') as object), grid: { id: 'grid-1', cell_count: 4 }, grid_id: 'grid-1' } as never;
		const root = mount(GenerationCard, {
			generation: gen,
			tile: { width: 220, height: 200 },
			entryMenu: historyMenu(gen),
			showActions: true,
			showCheckbox: true
		});

		expect(labels(root)).not.toContain('Favorite');
	});

	it('has no legacy icons on a card with no tile and no menu', () => {
		const root = mount(GenerationCard, { generation: generation('g1'), showActions: true });

		for (const retired of RETIRED) expect(labels(root)).not.toContain(retired);
	});
});

describe('Justified gallery cards', () => {
	beforeEach(() => {
		Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 1000 });
	});

	afterEach(() => {
		delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
	});

	it('give every card the Favorite toggle and the menu chip when actions are on', () => {
		const root = mount(JustifiedGenerationGallery, {
			generations: [generation('g1'), generation('g2', true)],
			showActions: true
		});
		const cards = root.querySelectorAll('[data-generation-card]');
		expect(cards).toHaveLength(2);

		for (const card of Array.from(cards)) {
			const names = corner(card);
			expect(names).toContain('Item actions');
			expect(names.some((name) => name === 'Favorite' || name === 'Remove favorite')).toBe(true);
			for (const retired of RETIRED) expect(names).not.toContain(retired);
		}
	});

	it('is menu-only on read-only views that turn actions off', () => {
		const root = mount(JustifiedGenerationGallery, { generations: [generation('g1')], showActions: false });
		const card = root.querySelector('[data-generation-card]')!;
		expect(card).not.toBeNull();

		expect(corner(card)).toEqual(['Item actions']);
	});
});

describe('Library card corner at every tile width', () => {
	const menu = createLibraryEntryMenu({
		items: () => [ITEM as never],
		selectedIds: () => [],
		host: { open: vi.fn(), remove: vi.fn(), favorite: vi.fn() },
		onTool: vi.fn()
	});

	for (const width of WIDTHS) {
		it(`holds the checkbox, Favorite and the menu chip at ${width}px`, () => {
			const root = mount(LibraryCard, {
				item: ITEM,
				entryMenu: menu,
				tile: { width, height: 160 },
				showActions: true,
				showCheckbox: true
			});

			const names = corner(root);
			expect(names).toEqual(['Select item', 'Favorite', 'Item actions']);
			for (const retired of RETIRED) expect(names).not.toContain(retired);
		});
	}
});
