// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { A1111_TOOL, BOOMERANG_TOOL, CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const openMediaTool = vi.fn();
const openPaintEditor = vi.fn();
const listMediaTools = vi.fn();
const toastShow = vi.fn();
const goto = vi.fn();

vi.mock('$lib/components/plugins/PluginSlot.svelte', async () => ({
	default: (await import('./fixtures/PluginSlotStub.svelte')).default
}));
vi.mock('$lib/media-tools/open', () => ({
	openMediaTool: (...args: unknown[]) => openMediaTool(...args),
	openPaintEditor: (...args: unknown[]) => openPaintEditor(...args)
}));
vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: (...args: unknown[]) => listMediaTools(...args) }));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));
vi.mock('$lib/components/workbench/renderers/MeshPreview.svelte', async () => ({
	default: (await import('./fixtures/MeshPreviewStub.svelte')).default
}));
vi.mock('$app/navigation', () => ({ goto: (...args: unknown[]) => goto(...args) }));
vi.mock('$lib/stores/toast', () => ({
	toasts: {
		show: (...args: unknown[]) => toastShow(...args),
		success: vi.fn(),
		error: vi.fn(),
		info: vi.fn(),
		warning: vi.fn()
	}
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		getClient: vi.fn(() => ({
			get: vi.fn().mockRejectedValue(new Error('not mocked')),
			post: vi.fn().mockRejectedValue(new Error('not mocked')),
			put: vi.fn().mockRejectedValue(new Error('not mocked'))
		})),
		getGenerationParams: vi.fn().mockResolvedValue({ success: true, data: { parameters: {}, models: [] } }),
		getGenerationById: vi.fn().mockResolvedValue({ success: false, error: 'not mocked' }),
		getTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } }),
		getBaseURL: vi.fn(() => ''),
		getToken: vi.fn(() => null),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn()
	}
}));

const { default: GenerationDetailsModal } = await import('$lib/components/modals/GenerationDetailsModal.svelte');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { createClassComponent } = await import('svelte/legacy');
const { registerDetailsReturn, returnToDetails } = await import('$lib/tools/detailsReturn');
const { flushSync } = await import('svelte');

function generation(fileType: string, extra: Record<string, unknown> = {}) {
	const ext = fileType === 'image' ? 'png' : fileType === 'video' ? 'mp4' : fileType === 'audio' ? 'wav' : 'glb';
	return {
		id: 'gen-row',
		preset_id: 'preset-a',
		form_data: {},
		status: 'completed' as const,
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		segments: [],
		rating: 0,
		is_favorite: false,
		files: [
			{
				id: 42,
				file_path: `/outputs/gen-row/0.${ext}`,
				file_type: fileType,
				is_final: true,
				width: 512,
				height: 512,
				created_at: '2026-01-01T00:00:00Z'
			}
		],
		...extra
	};
}

let destroy: (() => void) | undefined;

function mountModal(gen: Record<string, unknown>, props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationDetailsModal as never,
		target,
		props: { generation: gen as never, isOpen: true, ...props }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	return component;
}

async function open(gen: Record<string, unknown>, props: Record<string, unknown> = {}) {
	const component = mountModal(gen, props);
	await settle();
	flushSync();
	return component;
}

const barIds = () =>
	Array.from(document.body.querySelectorAll<HTMLElement>('[data-details-action-bar] [data-bar-item]')).map(
		(node) => node.dataset.barItem
	);

const barButton = (id: string) => document.body.querySelector<HTMLElement>(`[data-bar-item="${id}"]`)!;

async function menuLabels(id: string) {
	barButton(id).click();
	await settle(3);
	flushSync();
	return Array.from(document.body.querySelectorAll<HTMLElement>('[data-bar-menu] [data-bar-row-label]')).map(
		(node) => (node.textContent ?? '').trim()
	);
}

async function pick(menuId: string, label: string) {
	await menuLabels(menuId);
	Array.from(document.body.querySelectorAll<HTMLElement>('[data-bar-menu] [role="menuitem"]'))
		.find((node) => (node.querySelector('[data-bar-row-label]')?.textContent ?? '').trim() === label)!
		.click();
	flushSync();
}

const corner = () => document.body.querySelector<HTMLElement>('[data-stage-meta]');

beforeEach(() => {
	openMediaTool.mockReset();
	openPaintEditor.mockReset();
	toastShow.mockReset();
	goto.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue([...CORE_TOOLS, BOOMERANG_TOOL, A1111_TOOL]);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
});

describe('GenerationDetailsModal action bar', () => {
	it('lists the image bar in the approved order, between the header and the media', async () => {
		await open(generation('image'));

		expect(barIds()).toEqual([
			'edit',
			'effects',
			'favorite',
			'tags',
			'reuse',
			'copy-to-library',
			'download',
			'more'
		]);
		const bar = document.body.querySelector('[data-details-action-bar]')!;
		const stage = document.body.querySelector('[data-details-stage]')!;
		expect(bar.compareDocumentPosition(stage) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(document.body.querySelector('[data-media-tool-row]')).toBeNull();
		expect(document.body.querySelector('[data-stage-toolbar]')).toBeNull();
	});

	it('lets History crop, resize and edit a generated image', async () => {
		await open(generation('image'));

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
	});

	it('lets History trim, crop, resize and take a frame from a generated clip', async () => {
		await open(generation('video'));

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
	});

	it('lets History trim and split generated audio', async () => {
		await open(generation('audio'));

		expect(await menuLabels('edit')).toEqual(['Trim', 'Speed', 'Split']);
	});

	it('shows Effects for an image only', async () => {
		await open(generation('image'));
		expect(await menuLabels('effects')).toEqual(['Apply a look…']);
		destroy?.();
		document.body.innerHTML = '';

		await open(generation('video'));
		expect(barIds()).not.toContain('effects');
	});

	it('shows nothing to edit for a mesh but keeps the generation actions', async () => {
		await open(generation('mesh'));

		expect(barIds()).toEqual(['favorite', 'tags', 'reuse', 'download', 'more']);
	});

	it('leaves the media corner with only what describes the media', async () => {
		await open(generation('image'));

		const labels = Array.from(document.body.querySelectorAll('[data-details-stage] button, [data-details-stage] [role="button"]'))
			.map((node) => node.getAttribute('aria-label') ?? (node.textContent ?? '').trim())
			.join(' | ');
		expect(labels).not.toContain('Add to favorites');
		expect(labels).not.toContain('Manage tags');
		expect(labels).not.toContain('Export');
		expect(labels).not.toContain('Publish');
		expect(labels).not.toContain('Display in new tab');
		expect(corner()).not.toBeNull();
	});
});

describe('GenerationDetailsModal Download and More', () => {
	it('lists Download file, Export bundle and the A1111 export under Download', async () => {
		await open(generation('image'));

		expect(await menuLabels('download')).toEqual([
			'Download file',
			'Export bundle',
			'Download with A1111 metadata'
		]);
	});

	it('leaves the A1111 export out of Download for a video', async () => {
		await open(generation('video'));

		expect(await menuLabels('download')).toEqual(['Download file', 'Export bundle']);
	});

	it('lists Publish and Open in new tab under More, with the plugin actions slot after them', async () => {
		await open(generation('image'));

		expect(await menuLabels('more')).toEqual(['Publish to Inspirations', 'Open in new tab']);
		const slot = document.body.querySelector('[data-bar-plugin-actions] [data-plugin-slot-stub]')!;
		expect(slot.getAttribute('data-hook')).toBe('image.actions');
		expect(slot.getAttribute('data-context')).toContain('gen-row');
	});

	it('opens the current file in a new tab', async () => {
		const open_ = vi.spyOn(window, 'open').mockImplementation(() => null);
		await open(generation('image'));

		await pick('more', 'Open in new tab');

		expect(open_).toHaveBeenCalledWith(expect.stringContaining('gen-row'), '_blank');
		open_.mockRestore();
	});

	it('opens the publish dialog from More', async () => {
		await open(generation('image'));

		await pick('more', 'Publish to Inspirations');
		await settle(2);

		expect(document.body.textContent).toContain('Publish to Inspirations');
	});

	it('shows Favorite on for a favorite generation and off otherwise', async () => {
		await open(generation('image'));
		expect(barButton('favorite').getAttribute('aria-pressed')).toBe('false');
		destroy?.();
		document.body.innerHTML = '';

		await open(generation('image', { is_favorite: true }));
		expect(barButton('favorite').getAttribute('aria-pressed')).toBe('true');
	});
});

describe('GenerationDetailsModal generation actions', () => {
	it('keeps Reuse settings and Copy to Library in the page group of the bar', async () => {
		await open(generation('image'));

		expect(barButton('reuse').textContent).toContain('Reuse settings');
		expect(barButton('copy-to-library').textContent).toContain('Copy to Library');
	});

	it('leaves out Reuse settings for a generation with no preset', async () => {
		await open(generation('image', { preset_id: null }));

		expect(barIds()).toEqual(['edit', 'effects', 'favorite', 'tags', 'copy-to-library', 'download', 'more']);
	});

	it('dispatches reuse with the generation when Reuse settings is pressed', async () => {
		const gen = generation('image');
		const component = await open(gen);
		const onReuse = vi.fn();
		component.$on('reuse', (event: CustomEvent) => onReuse(event.detail));

		barButton('reuse').click();

		expect(onReuse).toHaveBeenCalledTimes(1);
		expect(onReuse.mock.calls[0][0].id).toBe('gen-row');
	});
});

describe('GenerationDetailsModal running a tool', () => {
	it('opens a tool on the generated file with its generation file id', async () => {
		await open(generation('image'));

		await pick('edit', 'Crop');

		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('crop');
		expect(call.scope).toBe('history');
		expect(call.items).toHaveLength(1);
		expect(call.items[0]).toMatchObject({ generationId: 'gen-row', generationFileId: '42', kind: 'image' });
	});

	it('runs the A1111 export on the generation it shows', async () => {
		await open(generation('image'));

		await pick('download', 'Download with A1111 metadata');

		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe(A1111_TOOL.id);
		expect(call.context.generationIds).toEqual(['gen-row']);
	});

	it('closes the details before it opens the tool when the History page can bring them back, with quiet success', async () => {
		const unregister = registerDetailsReturn(vi.fn());
		const onClose = vi.fn();
		await open(generation('image'), { onClose });
		await pick('edit', 'Resize');

		expect(onClose).toHaveBeenCalledTimes(1);
		expect(openMediaTool.mock.calls[0][0].announce.quietSuccess).toBe(true);
		expect(toastShow).not.toHaveBeenCalled();
		unregister();
	});

	it('leaves the details open when no History page is there to bring them back', async () => {
		const onClose = vi.fn();
		await open(generation('image'), { onClose });
		await pick('edit', 'Resize');

		expect(onClose).not.toHaveBeenCalled();
		expect(openMediaTool).toHaveBeenCalledTimes(1);
	});

	it('closes the details and opens the detached editor for Edit image', async () => {
		const unregister = registerDetailsReturn(vi.fn());
		const onClose = vi.fn();
		await open(generation('image'), { onClose });

		await pick('edit', 'Edit image');
		unregister();

		expect(onClose).toHaveBeenCalledTimes(1);
		const options = openPaintEditor.mock.calls[0][0];
		expect(options.request.kind).toBe('paint');
		expect(options.request.source.url).toContain('gen-row');
		expect(options.scope).toBe('history');
	});

	it('opens the editor on the Filters tool for Apply a look', async () => {
		await open(generation('image'));

		await pick('effects', 'Apply a look…');

		expect(openPaintEditor.mock.calls[0][0].request.initialTool).toBe('filters');
	});
});

describe('GenerationDetailsModal keyboard in the bar', () => {
	it('keeps Left and Right for the bar instead of walking to the next generation', async () => {
		const onNavigate = vi.fn().mockResolvedValue(true);
		await open(generation('image'), { onNavigate, hasNext: true, hasPrevious: true });

		barButton('edit').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
		await settle(2);

		expect(onNavigate).not.toHaveBeenCalled();
	});

	it('still walks to the next generation from the media', async () => {
		const onNavigate = vi.fn().mockResolvedValue(true);
		await open(generation('image'), { onNavigate, hasNext: true, hasPrevious: true });

		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
		await settle(2);

		expect(onNavigate).toHaveBeenCalledWith(1);
	});
});

describe('GenerationDetailsModal letter shortcuts', () => {
	const press = (key: string, target: EventTarget = window, init: KeyboardEventInit = {}) => {
		target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
		flushSync();
	};

	it('E opens the image editor, as the Edit image hint says', async () => {
		const unregister = registerDetailsReturn(vi.fn());
		const onClose = vi.fn();
		await open(generation('image'), { onClose });

		press('e');
		unregister();

		expect(openPaintEditor).toHaveBeenCalledTimes(1);
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('D downloads the file', async () => {
		const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
		await open(generation('image'));

		press('d');

		expect(click).toHaveBeenCalledTimes(1);
		click.mockRestore();
	});

	it('F toggles the favorite', async () => {
		const { historyStore } = await import('$lib/stores/history');
		const toggle = vi.spyOn(historyStore, 'toggleFavorite').mockImplementation(() => undefined as never);
		await open(generation('image'));

		press('f');

		expect(toggle).toHaveBeenCalledWith('gen-row');
		toggle.mockRestore();
	});

	it('does nothing for E on a video, which has no Edit image', async () => {
		await open(generation('video'));

		press('e');

		expect(openPaintEditor).not.toHaveBeenCalled();
	});

	it('does nothing while typing in a field', async () => {
		await open(generation('image'));
		const input = document.createElement('input');
		document.body.appendChild(input);
		input.focus();

		press('e', input);

		expect(openPaintEditor).not.toHaveBeenCalled();
	});

	it('does nothing while a bar menu is open above the details', async () => {
		await open(generation('image'));
		barButton('more').click();
		flushSync();
		await settle(3);

		press('e');

		expect(openPaintEditor).not.toHaveBeenCalled();
	});

	it('does nothing while another dialog is open above the details', async () => {
		await open(generation('image'));
		await pick('more', 'Publish to Inspirations');
		await settle(3);

		press('e');

		expect(openPaintEditor).not.toHaveBeenCalled();
	});

	it('does nothing with a modifier held', async () => {
		await open(generation('image'));

		press('e', window, { ctrlKey: true });

		expect(openPaintEditor).not.toHaveBeenCalled();
	});

	it('does not let the key through to the page behind', async () => {
		const leaked = vi.fn();
		document.addEventListener('keydown', leaked);
		await open(generation('image'));

		press('e');

		expect(leaked).not.toHaveBeenCalled();
		document.removeEventListener('keydown', leaked);
	});
});

describe('GenerationDetailsModal scroll on the way back', () => {
	const stored = new WeakMap<HTMLElement, number>();
	let descriptor: PropertyDescriptor | undefined;

	beforeEach(() => {
		descriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop');
		Object.defineProperty(Element.prototype, 'scrollTop', {
			configurable: true,
			get(this: HTMLElement) {
				return stored.get(this) ?? 0;
			},
			set(this: HTMLElement, value: number) {
				stored.set(this, Math.min(value, 1000));
			}
		});
	});

	afterEach(() => {
		if (descriptor) Object.defineProperty(Element.prototype, 'scrollTop', descriptor);
		else delete (Element.prototype as { scrollTop?: number }).scrollTop;
	});

	const panel = () => document.body.querySelector<HTMLElement>('[data-details-info]')!;

	async function launchAndCome(scrolled: number, comeBack: boolean) {
		const unregister = registerDetailsReturn(vi.fn());
		await open(generation('image'));
		panel().scrollTop = scrolled;
		await pick('edit', 'Resize');
		destroy?.();
		destroy = undefined;
		document.body.innerHTML = '';
		if (comeBack) returnToDetails({ generationId: 'gen-row', fileIndex: 0 });
		await open(generation('image'));
		unregister();
	}

	it('puts the information column back where it was when the details return after a Cancel', async () => {
		await launchAndCome(420, true);

		expect(panel().scrollTop).toBe(420);
	});

	it('opens a generation at the top when it is opened afresh and not returned to', async () => {
		await launchAndCome(420, false);

		expect(panel().scrollTop).toBe(0);
	});
});
