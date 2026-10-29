// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
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
	createClassComponent({
		component: LibraryItemModal as any,
		target,
		props: { item, onClose: vi.fn(), onDeleteRequest: vi.fn(), ...props }
	});
	return document.body;
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

const rowButtons = (group: string) =>
	Array.from(document.body.querySelectorAll(`[data-row-group="${group}"] button`)).map((button) =>
		(button.textContent || '').trim()
	);

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
	document.body.innerHTML = '';
	libraryStore.reset();
	openMediaTool.mockReset();
	startRuleFrom.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue([...CORE_TOOLS, BOOMERANG_TOOL]);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

describe('LibraryItemModal edit row', () => {
	it('offers only the tools the medium has', async () => {
		mount(image);
		await settle();
		expect(rowButtons('edit')).toEqual(['Crop', 'Resize', 'Edit image', 'More tools']);

		document.body.innerHTML = '';
		mount(clip);
		await settle();
		expect(rowButtons('edit')).toEqual(['Trim', 'Crop', 'Resize', 'Frame', 'More tools']);

		document.body.innerHTML = '';
		mount(track);
		await settle();
		expect(rowButtons('edit')).toEqual(['Trim', 'Split', 'More tools']);
	});

	it('keeps the old row of Crop, Trim, Split, Resize, Frame and Edit buttons out of the modal', async () => {
		mount(image);
		await settle();

		const row = document.body.querySelector('[data-media-tool-row]')!;
		expect(buttonByText(row as HTMLElement, 'Split')).toBeUndefined();
		expect(buttonByText(row as HTMLElement, 'Trim')).toBeUndefined();
		expect(document.body.querySelectorAll('[data-media-tool-row]')).toHaveLength(1);
	});

	it('shows the Effects row for an image and not for a clip or a track', async () => {
		mount(image);
		await settle();
		expect(rowButtons('effects')).toEqual(['Apply a look…']);

		document.body.innerHTML = '';
		mount(clip);
		await settle();
		expect(document.body.querySelector('[data-row-group="effects"]')).toBeNull();

		document.body.innerHTML = '';
		mount(track);
		await settle();
		expect(document.body.querySelector('[data-row-group="effects"]')).toBeNull();
	});

	it('keeps the library actions in their own group, in order', async () => {
		mount(image);
		await settle();

		const page = document.body.querySelector('[data-row-group="page"]')!;
		expect(page.getAttribute('aria-label')).toBe('Library actions');
		expect(rowButtons('page')).toEqual(['Auto-file items like this…', 'Delete', 'Download']);
	});

	it('places the row between the media and the tag line', async () => {
		mount(image);
		await settle();

		const row = document.body.querySelector('[data-media-tool-row]')!;
		const media = document.body.querySelector('img[alt="portrait.png"]')!;
		const tags = Array.from(document.body.querySelectorAll('button')).find((button) =>
			(button.textContent ?? '').includes('Add tags')
		);
		expect(media.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		if (tags) expect(row.compareDocumentPosition(tags) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it('asks the page to delete the item it shows', async () => {
		const onDeleteRequest = vi.fn();
		const target = mount(image, { onDeleteRequest });
		await settle();

		buttonByText(target, 'Delete')!.click();

		expect(onDeleteRequest).toHaveBeenCalledWith(expect.objectContaining({ id: 'row-1' }));
	});

	it('hands the item to the auto-file rule', async () => {
		const target = mount(image);
		await settle();

		buttonByText(target, 'Auto-file items like this…')!.click();

		expect(startRuleFrom).toHaveBeenCalledTimes(1);
	});
});

describe('LibraryItemModal running a tool', () => {
	it('opens the tool on the row it is showing', async () => {
		const target = mount(image);
		await settle();

		buttonByText(target, 'Crop')!.click();

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
		buttonByText(target, 'Resize')!.click();

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
		buttonByText(target, 'Resize')!.click();

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

	it('opens the image editor for Edit image without leaving the modal', async () => {
		const target = mount(image);
		await settle();

		buttonByText(target, 'Edit image')!.click();
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

		buttonByText(target, 'Edit image')!.click();
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
