// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const openMediaTool = vi.fn();
const listMediaTools = vi.fn();

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));
vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: (...args: unknown[]) => listMediaTools(...args),
	startMediaToolRun: vi.fn(),
	getMediaToolRun: vi.fn(),
	cancelMediaToolRun: vi.fn(),
	previewMediaTool: vi.fn()
}));
vi.mock('$lib/stores/toast', () => ({
	toasts: { show: vi.fn(), success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() }
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		listLibraryItems: vi.fn(),
		getLibraryFacets: vi.fn(),
		listCollections: vi.fn(),
		getTags: vi.fn(),
		addUploadsToCollection: vi.fn(),
		removeUploadsFromCollection: vi.fn(),
		setLibraryFavorite: vi.fn()
	}
}));

class FakeResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

const { api } = await import('$lib/services/api/index');
const { libraryStore } = await import('$lib/stores/library');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { default: LibraryPage } = await import('../../src/routes/library/+page.svelte');
const { createClassComponent } = await import('svelte/legacy');

function row(id: string, kind: string, name: string) {
	return {
		id,
		filename: `${id}-file`,
		original_filename: name,
		media_type: kind,
		url: `/u/${id}`,
		thumbnail_medium: `/t/${id}.jpg`,
		width: 640,
		height: 480,
		created_at: new Date().toISOString(),
		tags: []
	};
}

let destroy: (() => void) | undefined;
let widthSpy: ReturnType<typeof vi.spyOn> | undefined;
let originalResizeObserver: unknown;

async function mountPage() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: LibraryPage as never, target, props: {} });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle(12);
	return target;
}

const cardIds = () =>
	Array.from(document.body.querySelectorAll('[data-library-card]')).map((card) => card.getAttribute('data-library-card'));

const toolRow = (id: string) => document.body.querySelector<HTMLButtonElement>(`[data-tools-menu] [data-tool="${id}"]`);

async function openBarMenu() {
	Array.from(document.body.querySelectorAll('button'))
		.find((button) => (button.textContent ?? '').trim().startsWith('Tools'))!
		.click();
	await settle(4);
}

beforeEach(() => {
	originalResizeObserver = (globalThis as any).ResizeObserver;
	(globalThis as any).ResizeObserver = FakeResizeObserver;
	widthSpy = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(1200);
	libraryStore.reset();
	openMediaTool.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue(CORE_TOOLS);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
	vi.mocked(api.listLibraryItems).mockResolvedValue({
		success: true,
		data: {
			items: [row('a', 'image', 'a.png'), row('b', 'video', 'b.mp4'), row('c', 'image', 'c.png')],
			total: 3,
			limit: 24,
			offset: 0
		}
	} as never);
	vi.mocked(api.getLibraryFacets).mockResolvedValue({ success: true, data: { media_types: { image: 2, video: 1 } } } as never);
	vi.mocked(api.listCollections).mockResolvedValue({ success: true, data: { collections: [], total: 0 } } as never);
	vi.mocked(api.getTags).mockResolvedValue({ success: true, data: { tags: [] } } as never);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	widthSpy?.mockRestore();
	libraryStore.reset();
	(globalThis as any).ResizeObserver = originalResizeObserver;
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('Library page media tools', () => {
	it('draws the cards the wiring tests below act on', async () => {
		await mountPage();

		expect(cardIds()).toEqual(['a', 'b', 'c']);
	});

	it('opens a catalog tool picked from the selection bar on the selected items', async () => {
		await mountPage();
		libraryStore.toggleSelect('b');
		await settle(4);
		await openBarMenu();

		toolRow('trim')!.click();

		expect(openMediaTool).toHaveBeenCalledTimes(1);
		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('trim');
		expect(call.scope).toBe('library');
		expect(call.items.map((entry: { id: string }) => entry.id)).toEqual(['b']);
	});

	it('puts a result next to its source tagged New and clears the selection', async () => {
		await mountPage();
		libraryStore.toggleSelect('b');
		await settle(4);
		await openBarMenu();
		toolRow('trim')!.click();

		await openMediaTool.mock.calls[0][0].onResult(
			[
				{
					id: 'b2',
					filename: 'b2-file.mp4',
					original_filename: 'b-trimmed.mp4',
					media_type: 'video',
					url: '/u/b2-file.mp4'
				}
			],
			{ toolId: 'trim', replaced: false }
		);
		await settle(6);

		expect(cardIds()).toEqual(['a', 'b2', 'b', 'c']);
		const fresh = document.body.querySelector('[data-library-card="b2"] [data-new-tag]');
		expect(fresh?.textContent).toBe('New');
		expect(document.body.querySelector('[data-library-card="b"] [data-new-tag]')).toBeNull();
		expect(document.body.textContent).not.toContain('1 selected');
	});

	it('keeps a result newer than its source in the source day, without breaking the grid', async () => {
		const old = new Date(Date.now() - 3 * 86_400_000).toISOString();
		vi.mocked(api.listLibraryItems).mockResolvedValue({
			success: true,
			data: {
				items: [
					row('a', 'image', 'a.png'),
					{ ...row('b', 'video', 'b.mp4'), created_at: old },
					{ ...row('c', 'image', 'c.png'), created_at: old }
				],
				total: 3,
				limit: 24,
				offset: 0
			}
		} as never);
		await mountPage();
		libraryStore.toggleSelect('c');
		await settle(4);
		await openBarMenu();
		toolRow('crop')!.click();

		await openMediaTool.mock.calls[0][0].onResult(
			[
				{
					id: 'c2',
					filename: 'c2-file.png',
					original_filename: 'c-crop.png',
					media_type: 'image',
					url: '/u/c2-file.png',
					created_at: new Date().toISOString()
				}
			],
			{ toolId: 'crop', replaced: false }
		);
		await settle(6);

		expect(cardIds()).toEqual(['a', 'b', 'c2', 'c']);
		const headers = Array.from(document.body.querySelectorAll('.font-mono.uppercase')).map((node) => node.textContent?.trim());
		expect(headers.filter((text) => text === 'Today')).toHaveLength(1);
	});

	it('refreshes the counts after a result', async () => {
		await mountPage();
		libraryStore.toggleSelect('a');
		await settle(4);
		await openBarMenu();
		toolRow('crop')!.click();
		vi.mocked(api.getLibraryFacets).mockClear();
		vi.mocked(api.listCollections).mockClear();

		await openMediaTool.mock.calls[0][0].onResult(
			[{ id: 'a2', filename: 'a2-file.png', media_type: 'image', url: '/u/a2' }],
			{ toolId: 'crop', replaced: false }
		);

		expect(api.getLibraryFacets).toHaveBeenCalled();
		expect(api.listCollections).toHaveBeenCalled();
	});

	it('opens a catalog tool picked from a card menu and leaves the selection alone', async () => {
		await mountPage();
		libraryStore.toggleSelect('c');
		libraryStore.toggleSelect('a');
		await settle(4);

		const card = document.body.querySelector('[data-library-card="b"]')!;
		card.querySelector<HTMLButtonElement>('[data-entry-menu-trigger]')!.click();
		await settle(4);
		expect(toolRow('trim')).toBeTruthy();
		expect(toolRow('concat')).toBeNull();
		toolRow('trim')!.click();

		const call = openMediaTool.mock.calls[0][0];
		expect(call.items.map((entry: { id: string }) => entry.id)).toEqual(['b']);

		await call.onResult([], { toolId: 'trim', replaced: false });
		await settle(2);
		let state: any;
		libraryStore.subscribe((value) => (state = value))();
		expect(state.selectedIds).toEqual(['c', 'a']);
	});

	it('favorites the card from its menu through the library store', async () => {
		vi.mocked(api.setLibraryFavorite).mockResolvedValue({ success: true, data: { id: 'a', is_favorite: true } } as never);
		await mountPage();

		const card = document.body.querySelector('[data-library-card="a"]')!;
		card.querySelector<HTMLButtonElement>('[data-entry-menu-trigger]')!.click();
		await settle(4);
		toolRow('favorite')!.click();
		await settle(4);

		expect(api.setLibraryFavorite).toHaveBeenCalledWith('a', true);
		let state: any;
		libraryStore.subscribe((value) => (state = value))();
		expect(state.items.find((entry: { id: string }) => entry.id === 'a').is_favorite).toBe(true);
	});
});
