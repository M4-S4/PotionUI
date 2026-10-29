// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BOOMERANG_TOOL, CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const openMediaTool = vi.fn();
const listMediaTools = vi.fn();
const toastShow = vi.fn();
const goto = vi.fn();

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));
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

const rowButtons = (group: string) =>
	Array.from(document.body.querySelectorAll(`[data-row-group="${group}"] button`)).map((button) =>
		(button.textContent ?? '').trim()
	);

const corner = () => document.body.querySelector<HTMLElement>('[data-stage-toolbar]')!;

const cornerLabels = () =>
	Array.from(corner().querySelectorAll('button')).map(
		(button) => button.getAttribute('aria-label') ?? (button.textContent ?? '').trim()
	);

function rowButton(text: string): HTMLButtonElement {
	return Array.from(document.body.querySelectorAll<HTMLButtonElement>('[data-media-tool-row] button')).find(
		(button) => (button.textContent ?? '').trim() === text
	)!;
}

beforeEach(() => {
	openMediaTool.mockReset();
	toastShow.mockReset();
	goto.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue([...CORE_TOOLS, BOOMERANG_TOOL]);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
});

describe('GenerationDetailsModal edit row', () => {
	it('lets History crop, resize and edit a generated image, with More tools for the rest', async () => {
		await open(generation('image'));

		expect(rowButtons('edit')).toEqual(['Crop', 'Resize', 'Edit image', 'More tools']);
	});

	it('lets History trim, crop, resize and take a frame from a generated clip', async () => {
		await open(generation('video'));

		expect(rowButtons('edit')).toEqual(['Trim', 'Crop', 'Resize', 'Frame', 'More tools']);
	});

	it('lets History trim and split generated audio', async () => {
		await open(generation('audio'));

		expect(rowButtons('edit')).toEqual(['Trim', 'Split', 'More tools']);
	});

	it('shows the Effects row for an image only', async () => {
		await open(generation('image'));
		expect(rowButtons('effects')).toEqual(['Apply a look…']);
		destroy?.();
		document.body.innerHTML = '';

		await open(generation('video'));
		expect(document.body.querySelector('[data-row-group="effects"]')).toBeNull();
	});

	it('shows nothing to edit for a mesh but keeps the generation actions', async () => {
		await open(generation('mesh'));

		expect(document.body.querySelector('[data-row-group="edit"]')).toBeNull();
		expect(rowButtons('page')).toEqual(['Reuse settings', 'Download']);
	});

	it('sits under the media, outside the corner toolbar', async () => {
		await open(generation('image'));

		const row = document.body.querySelector('[data-media-tool-row]')!;
		expect(corner().contains(row)).toBe(false);
		const stage = document.body.querySelector('[data-details-stage]')!;
		expect(stage.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});
});

describe('GenerationDetailsModal generation actions', () => {
	it('moves Reuse settings, Copy to Library and Download into the row', async () => {
		await open(generation('image'));

		const page = document.body.querySelector('[data-row-group="page"]')!;
		expect(page.getAttribute('aria-label')).toBe('Generation actions');
		expect(rowButtons('page')).toEqual(['Reuse settings', 'Copy to Library', 'Download']);
	});

	it('keeps Favorite, Tags, Export bundle, Publish and Open in a new tab in the corner', async () => {
		await open(generation('image'));

		const labels = cornerLabels();
		expect(labels).toEqual(
			expect.arrayContaining([
				'Add to favorites',
				'Export this generation as a portable bundle',
				'Publish to Inspirations',
				'Display in new tab'
			])
		);
	});

	it('no longer puts Reuse, Download, Copy to Library or Edit in the corner', async () => {
		await open(generation('image'));

		const labels = cornerLabels().join(' | ');
		expect(labels).not.toContain("Reuse this generation's settings");
		expect(labels).not.toContain('Download');
		expect(labels).not.toContain('Copy this file to your library');
		expect(labels).not.toContain('Edit this image');
	});

	it('leaves out Reuse settings for a generation with no preset', async () => {
		await open(generation('image', { preset_id: null }));

		expect(rowButtons('page')).toEqual(['Copy to Library', 'Download']);
	});

	it('dispatches reuse with the generation when Reuse settings is pressed', async () => {
		const gen = generation('image');
		const component = await open(gen);
		const onReuse = vi.fn();
		component.$on('reuse', (event: CustomEvent) => onReuse(event.detail));

		rowButton('Reuse settings').click();

		expect(onReuse).toHaveBeenCalledTimes(1);
		expect(onReuse.mock.calls[0][0].id).toBe('gen-row');
	});
});

describe('GenerationDetailsModal running a tool', () => {
	it('opens a tool on the generated file with its generation file id', async () => {
		await open(generation('image'));

		rowButton('Crop').click();

		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('crop');
		expect(call.scope).toBe('history');
		expect(call.items).toHaveLength(1);
		expect(call.items[0]).toMatchObject({ generationId: 'gen-row', generationFileId: '42', kind: 'image' });
	});

	it('asks the tool run to name the result with a link to the Library, and posts no toast of its own', async () => {
		await open(generation('image'));
		rowButton('Resize').click();

		expect(openMediaTool.mock.calls[0][0].announce).toEqual({
			title: 'Saved to Library',
			action: { label: 'Open Library', href: '/library' }
		});

		await openMediaTool.mock.calls[0][0].onResult(
			[{ id: 'lib-1', filename: 'x-uuid.png', original_filename: 'gen-resized.png', media_type: 'image', url: '/u' }],
			{ toolId: 'resize', replaced: false }
		);

		expect(toastShow).not.toHaveBeenCalled();
	});

	it('opens the image editor for Edit image and keeps the details open', async () => {
		const onClose = vi.fn();
		await open(generation('image'), { onClose });

		rowButton('Edit image').click();
		flushSync();

		const stub = document.body.querySelector('[data-editor-stub]');
		expect(stub?.getAttribute('data-kind')).toBe('paint');
		expect(stub?.getAttribute('data-url')).toContain('gen-row');
		expect(onClose).not.toHaveBeenCalled();
	});

	it('opens the image editor on the Filters tool for Apply a look', async () => {
		await open(generation('image'));

		rowButton('Apply a look…').click();
		flushSync();

		expect(document.body.querySelector('[data-editor-stub]')?.getAttribute('data-initial-tool')).toBe('filters');
	});

	it('does not close the details on Escape while the image editor is open, and does once it is shut', async () => {
		const onClose = vi.fn();
		await open(generation('image'), { onClose });
		const escape = () =>
			window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));

		escape();
		await settle(2);
		expect(onClose).toHaveBeenCalledTimes(1);
		onClose.mockClear();

		rowButton('Edit image').click();
		flushSync();
		escape();
		await settle(2);
		expect(onClose).not.toHaveBeenCalled();

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		flushSync();
		escape();
		await settle(2);
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
