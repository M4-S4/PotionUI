// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
vi.mock('$lib/components/plugins/PluginSlot.svelte', async () => ({
	default: (await import('./fixtures/PluginSlotStub.svelte')).default
}));
import { mount, unmount, flushSync } from 'svelte';
import { createClassComponent } from 'svelte/legacy';
import { CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const toastSuccess = vi.fn();
const listMediaTools = vi.fn();

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: vi.fn() }));
vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: (...args: unknown[]) => listMediaTools(...args) }));
vi.mock('$lib/stores/toast', () => ({
	toasts: {
		show: vi.fn(),
		success: (...args: unknown[]) => toastSuccess(...args),
		error: vi.fn(),
		info: vi.fn(),
		warning: vi.fn()
	}
}));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/PaintResultStub.svelte')).default
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		listLibraryItems: vi.fn(),
		getLibraryFacets: vi.fn().mockResolvedValue({ success: false }),
		listCollections: vi.fn().mockResolvedValue({ success: false }),
		getTags: vi.fn().mockResolvedValue({ success: false }),
		setLibraryItemTags: vi.fn()
	}
}));

const { api } = await import('$lib/services/api/index');
const { libraryStore } = await import('$lib/stores/library');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools, buildLibraryToolContext } = await import('$lib/tools/tools');
const { default: LibraryItemModal } = await import('../../src/lib/components/library/LibraryItemModal.svelte');
const { default: HistoryEditImageModal } = await import(
	'../../src/routes/history/components/HistoryEditImageModal.svelte'
);

function row(id: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		filename: `${id}-file.png`,
		original_filename: `${id}.png`,
		media_type: 'image',
		mime_type: 'image/png',
		url: `/api/media/uploads/${id}-file.png`,
		thumbnail_medium: `/thumb/${id}.jpg`,
		width: 1024,
		height: 1536,
		created_at: new Date().toISOString(),
		tags: [{ id: 't1', name: 'keep' }],
		...extra
	};
}

const SERVER = [row('row-0'), row('row-1'), row('row-2')];

let cleanup: (() => void) | undefined;

const snapshot = () => {
	let state: any;
	libraryStore.subscribe((value) => (state = value))();
	return {
		ids: state.items.map((item: { id: string }) => item.id),
		urls: state.items.map((item: { url: string }) => item.url),
		thumbs: state.items.map((item: { thumbnail_medium?: string }) => item.thumbnail_medium ?? null),
		tags: state.items.map((item: { tags: unknown }) => item.tags),
		total: state.totalCount,
		fresh: state.freshIds,
		placed: state.placedIds,
		page: state.currentPage,
		selected: state.selectedItem ? { id: state.selectedItem.id, url: state.selectedItem.url } : null,
		toasts: toastSuccess.mock.calls.map((call) => call[0])
	};
};

async function seedLibrary() {
	toastSuccess.mockClear();
	libraryStore.reset();
	vi.mocked(api.listLibraryItems).mockResolvedValue({
		success: true,
		data: { items: SERVER, total: 3, limit: 24, offset: 0 }
	} as never);
	await libraryStore.load();
	libraryStore.setSelectedItem(SERVER[1] as never);
}

async function throughDetailsBar(result: 'new' | 'replace') {
	await seedLibrary();
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({
		component: LibraryItemModal as never,
		target,
		props: { item: SERVER[1], onClose: vi.fn(), onDeleteRequest: vi.fn() }
	});
	cleanup = () => {
		instance.$destroy();
		target.remove();
	};
	await settle();
	document.body.querySelector<HTMLElement>('[data-bar-item="edit"]')!.click();
	await settle();
	document.body.querySelector<HTMLElement>('[data-bar-row="edit-image"]')!.click();
	flushSync();
	document.body.querySelector<HTMLButtonElement>(`[data-paint-${result}]`)!.click();
	await settle();
	const after = snapshot();
	cleanup();
	cleanup = undefined;
	document.body.innerHTML = '';
	return after;
}

async function throughMenu(result: 'new' | 'replace', initialTool?: string, initialFilter?: string) {
	await seedLibrary();
	const context = buildLibraryToolContext([SERVER[1] as never], null);
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(HistoryEditImageModal, {
		target,
		props: { context, onClose: vi.fn(), onDone: vi.fn(), initialTool, initialFilter }
	});
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	await settle();
	document.body.querySelector<HTMLButtonElement>(`[data-paint-${result}]`)!.click();
	await settle();
	const after = snapshot();
	cleanup();
	cleanup = undefined;
	document.body.innerHTML = '';
	return after;
}

beforeEach(() => {
	toastSuccess.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue(CORE_TOOLS);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
	libraryStore.reset();
});

describe('Edit image on the Library, from the details bar or from the Tools and card menus', () => {
	it('adds the new image next to its source tagged New, the same either way', async () => {
		const fromRow = await throughDetailsBar('new');
		const fromMenu = await throughMenu('new');

		expect(fromMenu).toEqual(fromRow);
		expect(fromMenu.ids).toEqual(['row-0', 'row-new', 'row-1', 'row-2']);
		expect(fromMenu.fresh).toEqual(['row-new']);
	});

	it('stays on the page the user is on instead of jumping to page 1', async () => {
		const fromRow = await throughDetailsBar('new');
		const fromMenu = await throughMenu('new');

		expect(fromMenu.page).toBe(1);
		expect(fromMenu.page).toBe(fromRow.page);
	});

	it('says it with the same toast either way', async () => {
		const fromRow = await throughDetailsBar('new');
		const fromMenu = await throughMenu('new');

		expect(fromMenu.toasts).toEqual(['Saved to your library as a new image']);
		expect(fromMenu.toasts).toEqual(fromRow.toasts);
	});

	it('patches a replaced image in place with its new file, keeping its tags, the same either way', async () => {
		const fromRow = await throughDetailsBar('replace');
		const fromMenu = await throughMenu('replace');

		expect(fromMenu).toEqual(fromRow);
		expect(fromMenu.ids).toEqual(['row-0', 'row-1', 'row-2']);
		expect(fromMenu.urls[1]).toBe('/api/media/uploads/replaced-uuid.png');
		expect(fromMenu.tags[1]).toEqual([{ id: 't1', name: 'keep' }]);
		expect(fromMenu.fresh).toEqual([]);
	});

	it('does not leave the replaced card on the thumbnail of the file it replaced', async () => {
		const fromRow = await throughDetailsBar('replace');
		const fromMenu = await throughMenu('replace');

		expect(fromMenu.thumbs[1]).toBeNull();
		expect(fromRow.thumbs[1]).toBeNull();
		expect(fromMenu.thumbs[0]).toBe('/thumb/row-0.jpg');
	});

	it('says a replace with the same toast either way', async () => {
		const fromRow = await throughDetailsBar('replace');
		const fromMenu = await throughMenu('replace');

		expect(fromMenu.toasts).toEqual(['Replaced — tags and collections kept']);
		expect(fromMenu.toasts).toEqual(fromRow.toasts);
	});

	it('behaves the same when the menu opens the editor on a look', async () => {
		const plain = await throughMenu('new');
		const onLook = await throughMenu('new', 'filters', 'noir');

		expect(onLook).toEqual(plain);
	});
});
