// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
vi.mock('$lib/components/plugins/PluginSlot.svelte', async () => ({
	default: (await import('./fixtures/PluginSlotStub.svelte')).default
}));
import { BOOMERANG_TOOL, CORE_TOOLS } from './fixtures/mediaToolCatalog';

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

const openMediaTool = vi.fn();
const listMediaTools = vi.fn();
const startRuleFrom = vi.fn();

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));
vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: (...args: unknown[]) => listMediaTools(...args) }));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));
vi.mock('$lib/organize/handoff', () => ({ startRuleFrom: (...args: unknown[]) => startRuleFrom(...args) }));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		listLibraryItems: vi.fn().mockResolvedValue({ success: false }),
		getLibraryFacets: vi.fn().mockResolvedValue({ success: false }),
		listCollections: vi.fn().mockResolvedValue({ success: false }),
		getTags: vi.fn().mockResolvedValue({ success: false }),
		setLibraryItemTags: vi.fn()
	}
}));

const { libraryStore } = await import('$lib/stores/library');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { default: LibraryItemModal } = await import(
	'../../src/routes/library/components/LibraryItemModal.svelte'
);
const { createClassComponent } = await import('svelte/legacy');
const { tick } = await import('svelte');

function mount(item: Record<string, unknown>, props: Record<string, unknown> = {}) {
	libraryStore.setSelectedItem(item as never);
	const target = document.createElement('div');
	document.body.appendChild(target);
	mounted.push(
		createClassComponent({
			component: LibraryItemModal as any,
			target,
			props: { item, onClose: vi.fn(), onDeleteRequest: vi.fn(), ...props }
		})
	);
	return document.body;
}

const mounted: Array<{ $destroy: () => void }> = [];

function resetDom() {
	for (const instance of mounted.splice(0)) instance.$destroy();
	document.body.innerHTML = '';
}

function buttonByText(target: HTMLElement, text: string): HTMLButtonElement | undefined {
	return Array.from(target.querySelectorAll('button')).find(
		(button) => (button.textContent || '').trim() === text
	);
}

async function settle() {
	for (let i = 0; i < 8; i += 1) {
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

const state = () => {
	let value: any;
	libraryStore.subscribe((next) => (value = next))();
	return value;
};

const barIds = () =>
	Array.from(document.body.querySelectorAll<HTMLElement>('[data-details-action-bar] [data-bar-item]')).map(
		(node) => node.dataset.barItem
	);

async function menuLabels(id: string) {
	document.body.querySelector<HTMLElement>(`[data-bar-item="${id}"]`)!.click();
	await settle();
	return Array.from(document.body.querySelectorAll<HTMLElement>('[data-bar-menu] [role="menuitem"] [data-bar-row-label]')).map(
		(node) => (node.textContent || '').trim()
	);
}

async function pickMenu(id: string, label: string) {
	await menuLabels(id);
	const row = Array.from(document.body.querySelectorAll<HTMLElement>('[data-bar-menu] [role="menuitem"]')).find(
		(node) => (node.querySelector('[data-bar-row-label]')?.textContent || '').trim() === label
	)!;
	row.click();
	await settle();
}

const pickTool = (label: string) => pickMenu('edit', label);

const image = {
	id: 'row-1',
	filename: 'stored-uuid.png',
	original_filename: 'portrait.png',
	media_type: 'image',
	mime_type: 'image/png',
	url: '/api/media/uploads/stored-uuid.png',
	width: 1024,
	height: 1536,
	size: 900000,
	tags: []
};

const clip = {
	id: 'row-2',
	filename: 'stored-clip.mp4',
	original_filename: 'take-3.mp4',
	media_type: 'video',
	url: '/api/media/uploads/stored-clip.mp4',
	width: 1280,
	height: 720,
	duration_seconds: 8.4,
	fps: 24,
	tags: []
};

const track = {
	id: 'row-3',
	filename: 'stored-vo.wav',
	original_filename: 'vo.wav',
	media_type: 'audio',
	url: '/api/media/uploads/stored-vo.wav',
	duration_seconds: 32.6,
	tags: []
};

beforeEach(() => {
	resetDom();
	libraryStore.reset();
	openMediaTool.mockReset();
	startRuleFrom.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue([...CORE_TOOLS, BOOMERANG_TOOL]);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

describe('LibraryItemModal action bar', () => {
	it('offers only the tools the medium has', async () => {
		mount(image);
		await settle();
		expect(await menuLabels('edit')).toEqual([
			'Crop',
			'Resize',
			'Rotate',
			'Flip',
			'Fit canvas',
			'Color key',
			'Crop to subject',
			'Edit image'
		]);

		resetDom();
		mount(clip);
		await settle();
		expect(await menuLabels('edit')).toEqual([
			'Trim',
			'Crop',
			'Resize',
			'Rotate',
			'Flip',
			'Speed',
			'Frame',
			'Boomerang'
		]);

		resetDom();
		mount(track);
		await settle();
		expect(await menuLabels('edit')).toEqual(['Trim', 'Speed', 'Split']);
	});

	it('has no stacked rows of tool buttons', async () => {
		mount(image);
		await settle();

		expect(document.body.querySelector('[data-media-tool-row]')).toBeNull();
		expect(document.body.querySelector('[data-row-group]')).toBeNull();
		expect(document.body.querySelectorAll('[data-details-action-bar]')).toHaveLength(1);
	});

	it('shows Effects for an image and not for a clip or a track', async () => {
		mount(image);
		await settle();
		expect(await menuLabels('effects')).toEqual(['Apply a look…']);

		resetDom();
		mount(clip);
		await settle();
		expect(barIds()).not.toContain('effects');

		resetDom();
		mount(track);
		await settle();
		expect(barIds()).not.toContain('effects');
	});

	it('keeps the library actions in the bar, in order, with Delete last in More', async () => {
		mount(image);
		await settle();

		expect(barIds()).toEqual(['edit', 'effects', 'favorite', 'tags', 'auto-file', 'download', 'more']);
		expect(await menuLabels('more')).toEqual(['Open in new tab', 'Delete']);
	});

	it('offers the file as a plain Download button', async () => {
		const target = mount(image);
		await settle();

		expect(buttonByText(target, 'Download')).toBeDefined();
	});

	it('places the bar above the media', async () => {
		mount(image);
		await settle();

		const bar = document.body.querySelector('[data-details-action-bar]')!;
		const media = document.body.querySelector('img[alt="portrait.png"]')!;
		expect(bar.compareDocumentPosition(media) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it('shows the selected tags as chips under the media', async () => {
		mount({ ...image, tags: [{ id: 't1', name: 'keepers' }] });
		await settle();

		expect(document.body.querySelector('[data-library-item-tags]')?.textContent).toContain('keepers');
	});

	it('asks the page to delete the item it shows', async () => {
		const onDeleteRequest = vi.fn();
		mount(image, { onDeleteRequest });
		await settle();

		await pickMenu('more', 'Delete');

		expect(onDeleteRequest).toHaveBeenCalledWith(expect.objectContaining({ id: 'row-1' }));
	});

	it('hands the item to the auto-file rule', async () => {
		const target = mount(image);
		await settle();

		buttonByText(target, 'Auto-file…')!.click();

		expect(startRuleFrom).toHaveBeenCalledTimes(1);
	});
});

describe('LibraryItemModal running a tool', () => {
	it('opens the tool on the row it is showing', async () => {
		const target = mount(image);
		await settle();

		await pickTool('Crop');

		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('crop');
		expect(call.scope).toBe('library');
		expect(call.items).toEqual([expect.objectContaining({ id: 'row-1', itemId: 'row-1', kind: 'image' })]);
	});

	it('folds a saved-as-new result into the library next to its source, marked new', async () => {
		libraryStore.reset();
		await libraryStoreLoad([image, clip]);
		const target = mount(image);
		await settle();
		await pickTool('Resize');

		await openMediaTool.mock.calls[0][0].onResult(
			[
				{
					id: 'row-9',
					filename: 'resized-uuid.png',
					original_filename: 'portrait-resized.png',
					media_type: 'image',
					url: '/api/media/uploads/resized-uuid.png'
				}
			],
			{ toolId: 'resize', replaced: false }
		);

		expect(state().items.map((entry: { id: string }) => entry.id)).toEqual(['row-9', 'row-1', 'row-2']);
		expect(state().freshIds).toEqual(['row-9']);
	});

	it('patches the open preview onto the replaced file, keeping the row and its tags', async () => {
		const tagged = { ...image, tags: [{ id: 'tag-1', name: 'keepers' }] };
		const target = mount(tagged);
		await settle();
		await pickTool('Resize');

		await openMediaTool.mock.calls[0][0].onResult(
			[
				{
					id: 'row-1',
					filename: 'edited-uuid.png',
					media_type: 'image',
					url: '/api/media/uploads/edited-uuid.png',
					width: 1024,
					height: 1024
				}
			],
			{ toolId: 'resize', replaced: true }
		);

		expect(state().selectedItem.url).toBe('/api/media/uploads/edited-uuid.png');
		expect(state().selectedItem.id).toBe('row-1');
		expect(state().selectedItem.tags).toEqual([{ id: 'tag-1', name: 'keepers' }]);
		expect(state().freshIds).toEqual([]);
	});

	it('tells the store whether the run replaced the row, rather than leaving it to guess', async () => {
		const apply = vi.spyOn(libraryStore, 'applyToolResult');
		const target = mount(image);
		await settle();
		await pickTool('Resize');
		const produced = [{ id: 'row-1', filename: 'x.png', media_type: 'image', url: '/x.png' }];

		await openMediaTool.mock.calls[0][0].onResult(produced, { toolId: 'resize', replaced: true });
		await openMediaTool.mock.calls[0][0].onResult(produced, { toolId: 'resize', replaced: false });

		expect(apply.mock.calls[0]).toEqual([produced, ['row-1'], { replaced: true }]);
		expect(apply.mock.calls[1]).toEqual([produced, ['row-1'], { replaced: false }]);
		apply.mockRestore();
	});

	it('opens the image editor for Edit image without leaving the modal', async () => {
		const target = mount(image);
		await settle();

		await pickTool('Edit image');
		await settle();

		expect(document.body.querySelector('[data-editor-stub]')?.getAttribute('data-kind')).toBe('paint');
		expect(openMediaTool).not.toHaveBeenCalled();
	});
});

describe('LibraryItemModal closing', () => {
	it('stays open on Escape while the image editor is open, and closes once it is shut', async () => {
		const onClose = vi.fn();
		const target = mount(image, { onClose });
		await settle();
		const escape = () =>
			window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));

		await pickTool('Edit image');
		await settle();
		escape();
		await settle();
		expect(onClose).not.toHaveBeenCalled();

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		await settle();
		escape();
		await settle();
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});

async function libraryStoreLoad(items: Record<string, unknown>[]) {
	const { api } = await import('$lib/services/api/index');
	vi.mocked(api.listLibraryItems).mockResolvedValue({
		success: true,
		data: { items, total: items.length, limit: 24, offset: 0 }
	} as never);
	await libraryStore.load();
}
