// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
vi.mock('$lib/components/plugins/PluginSlot.svelte', async () => ({
	default: (await import('./fixtures/PluginSlotStub.svelte')).default
}));
import { settle } from './helpers/settle';

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

const getGenerationById = vi.fn();

vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));
vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: vi.fn().mockResolvedValue([]),
	startMediaToolRun: vi.fn(),
	getMediaToolRun: vi.fn(),
	cancelMediaToolRun: vi.fn(),
	previewMediaTool: vi.fn()
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		listLibraryItems: vi.fn().mockResolvedValue({ success: false }),
		getLibraryFacets: vi.fn().mockResolvedValue({ success: false }),
		listCollections: vi.fn().mockResolvedValue({ success: false }),
		getTags: vi.fn().mockResolvedValue({ success: false }),
		getGenerationById: (...args: unknown[]) => getGenerationById(...args)
	}
}));

const { default: LibraryItemModal } = await import('../../src/lib/components/library/LibraryItemModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

const row = {
	id: 'row-1',
	filename: 'a.png',
	original_filename: 'a.png',
	media_type: 'image',
	url: '/a.png',
	width: 64,
	height: 64,
	tags: []
};

const instances: Array<{ $destroy: () => void }> = [];

function mount(item: Record<string, unknown>, props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	instances.push(
		createClassComponent({
			component: LibraryItemModal as never,
			target,
			props: { item, onClose: vi.fn(), onDeleteRequest: vi.fn(), ...props }
		})
	);
}

const back = () => document.body.querySelector('[data-back-to-generation]');

beforeEach(() => {
	getGenerationById.mockReset().mockResolvedValue({ success: true, data: { form_data: { prompt: 'a lighthouse\nat dusk' } } });
});

afterEach(() => {
	for (const instance of instances.splice(0)) instance.$destroy();
	document.body.innerHTML = '';
});

describe('Back to the generation', () => {
	it('is a link to the History details of the source generation for a generation-sourced item', async () => {
		mount({ ...row, source_generation_id: 'g-42' });
		await settle();

		const link = back()!.querySelector('a')!;
		expect(link.textContent).toContain('Back to the generation');
		expect(link.getAttribute('href')).toBe('/history?generation=g-42');
	});

	it('is a button that hands the link to the host when the host can reopen the details itself', async () => {
		const onBackToGeneration = vi.fn();
		mount(row, { generationLink: { generationId: 'g-9', fileIndex: 3 }, onBackToGeneration });
		await settle();

		expect(back()!.querySelector('a')).toBeNull();
		back()!.querySelector('button')!.click();
		expect(onBackToGeneration).toHaveBeenCalledWith({ generationId: 'g-9', fileIndex: 3 });
	});

	it('is absent for an item that did not come from a generation', async () => {
		mount(row);
		await settle();

		expect(back()).toBeNull();
		expect(getGenerationById).not.toHaveBeenCalled();
	});

	it('says the generation was deleted, with no link, when it can no longer be read', async () => {
		getGenerationById.mockResolvedValue({ success: false });
		mount({ ...row, source_generation_id: 'g-gone' });
		await settle();

		expect(back()!.textContent).toContain('Generation deleted');
		expect(back()!.querySelector('a, button')).toBeNull();
	});

	it('says the same when the lookup fails', async () => {
		getGenerationById.mockRejectedValue(new Error('404'));
		mount({ ...row, source_generation_id: 'g-gone' });
		await settle();

		expect(back()!.textContent).toContain('Generation deleted');
	});
});
