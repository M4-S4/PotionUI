import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { settle } from './helpers/settle';
import { A1111_TOOL, BOOMERANG_TOOL, CORE_TOOLS, toolDto } from './fixtures/mediaToolCatalog';

const listMediaTools = vi.fn();
const listGenerationMedia = vi.fn();
const listUploads = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: (...args: unknown[]) => listMediaTools(...args),
	startMediaToolRun: vi.fn(),
	getMediaToolRun: vi.fn(),
	cancelMediaToolRun: vi.fn(),
	previewMediaTool: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({
	api: {
		listGenerationMedia: (...args: unknown[]) => listGenerationMedia(...args),
		listUploads: (...args: unknown[]) => listUploads(...args)
	}
}));

vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/OpenHostStub.svelte')).default
}));
vi.mock('$lib/media-tools/MediaToolDialog.svelte', async () => ({
	default: (await import('./fixtures/OpenHostStub.svelte')).default
}));
vi.mock('$lib/components/tools/ToolHost.svelte', async () => ({
	default: (await import('./fixtures/OpenHostStub.svelte')).default
}));

const { openMediaTool, routeFor } = await import('$lib/media-tools/open');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { toasts } = await import('$lib/stores/toast');

const image = { id: 'row-1', kind: 'image' as const, url: '/api/media/uploads/a.png', filename: 'a.png', width: 800, height: 600 };
const clip = { id: 'row-2', kind: 'video' as const, url: '/api/media/uploads/c.mp4', filename: 'c.mp4', durationSeconds: 8 };

function hosts(): Record<string, unknown>[] {
	return Array.from(document.body.querySelectorAll('[data-host-stub]')).map((el) =>
		JSON.parse(el.getAttribute('data-props') || '{}')
	);
}

function press(selector: string) {
	(document.body.querySelector(selector) as HTMLButtonElement).click();
}

function toastTexts(): string[] {
	return get(toasts).map((entry) => entry.message);
}

beforeEach(async () => {
	document.body.innerHTML = '';
	resetMediaToolCatalog();
	listMediaTools.mockReset().mockResolvedValue([...CORE_TOOLS, A1111_TOOL, BOOMERANG_TOOL]);
	listGenerationMedia.mockReset().mockResolvedValue({ success: false });
	listUploads.mockReset().mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } });
	for (const entry of get(toasts)) toasts.remove(entry.id);
});

describe('routeFor', () => {
	it('sends a tool with a built-in editor to that editor', () => {
		expect(routeFor(toolDto({ id: 'trim', editor: 'trim' }))).toBe('editor');
		expect(routeFor(toolDto({ id: 'crop', editor: 'crop' }))).toBe('editor');
		expect(routeFor(toolDto({ id: 'frame', editor: 'frame' }))).toBe('editor');
	});

	it('sends any other runnable tool to the dialog', () => {
		expect(routeFor(toolDto({ id: 'speed' }))).toBe('dialog');
		expect(routeFor(BOOMERANG_TOOL)).toBe('dialog');
	});

	it('does not take an editor name it does not know for a built-in surface', () => {
		expect(routeFor(toolDto({ id: 'x', editor: 'mask' }))).toBe('dialog');
		expect(routeFor(toolDto({ id: 'x', editor: 'paint' }))).toBe('dialog');
		expect(routeFor(toolDto({ id: 'x', editor: 'nonsense' }))).toBe('dialog');
	});

	it('sends a component-only plugin tool to its component', () => {
		expect(routeFor(A1111_TOOL)).toBe('component');
	});

	it('takes the editor first when a tool has both', () => {
		expect(routeFor(toolDto({ id: 'x', editor: 'trim', component: 'plugin:p:v' }))).toBe('editor');
	});
});

describe('opening a tool', () => {
	it('loads the catalog before it looks the tool up', async () => {
		expect(await openMediaTool({ toolId: 'speed', items: [clip], scope: 'library' })).toBe(true);
		expect(listMediaTools).toHaveBeenCalledTimes(1);
	});

	it('says so and opens nothing for a tool that is not in the catalog', async () => {
		const opened = await openMediaTool({ toolId: 'gone', items: [image], scope: 'library' });
		expect(opened).toBe(false);
		expect(hosts()).toEqual([]);
		expect(toastTexts()).toEqual(['This tool is no longer available.']);
	});

	it('gives the reason and opens nothing when the tool does not fit the selection', async () => {
		const opened = await openMediaTool({ toolId: 'speed', items: [image], scope: 'library' });
		expect(opened).toBe(false);
		expect(toastTexts()).toEqual(['Not available for image']);
		expect(hosts()).toEqual([]);
	});

	it('opens a many-to-one tool only for a selection inside its bounds', async () => {
		expect(await openMediaTool({ toolId: 'concat', items: [clip], scope: 'library' })).toBe(false);
		expect(toastTexts()).toEqual(['Select 2 to 8 items']);
		expect(await openMediaTool({ toolId: 'concat', items: [clip, { ...clip, id: 'r9' }], scope: 'library' })).toBe(true);
	});
});

describe('the dialog route', () => {
	it('mounts the dialog with the tool, the items and the scope, detached on a page', async () => {
		await openMediaTool({ toolId: 'speed', items: [clip], scope: 'library' });
		await settle();

		const [props] = hosts();
		expect((props.tool as { id: string }).id).toBe('speed');
		expect(props.scope).toBe('library');
		expect(props.detach).toBe(true);
		expect((props.items as { id: string; itemId: string }[]).map((item) => [item.id, item.itemId])).toEqual([['row-2', 'row-2']]);
	});

	it('waits for the result in a field, where there is no card to show progress on', async () => {
		await openMediaTool({ toolId: 'speed', items: [{ ...clip, id: 'field', storedPath: 'uploads/c.mp4' }], scope: 'field' });
		await settle();
		expect(hosts()[0].detach).toBe(false);
	});

	it('takes a history item as a generation, not a library row', async () => {
		await openMediaTool({
			toolId: 'speed',
			items: [{ ...clip, id: 'gen-1:0', generationId: 'gen-1', paramIndex: 0 }],
			scope: 'history'
		});
		await settle();
		const [item] = hosts()[0].items as { itemId?: string; generationId: string }[];
		expect(item.itemId).toBeUndefined();
		expect(item.generationId).toBe('gen-1');
	});

	it('hands what the dialog produced to the caller, and unmounts the dialog when it closes', async () => {
		const onResult = vi.fn();
		const onClose = vi.fn();
		await openMediaTool({ toolId: 'speed', items: [clip], scope: 'library', onResult, onClose });
		await settle();

		press('[data-stub-result-dialog]');
		expect(onResult).toHaveBeenCalledWith(
			[expect.objectContaining({ filename: 'out.png' })],
			{ toolId: 'speed', replaced: false }
		);

		press('[data-stub-close]');
		await settle();
		expect(onClose).toHaveBeenCalledTimes(1);
		expect(document.body.querySelector('[data-host-stub]')).toBeNull();
	});

	it('ignores items that are not media when it opens a dialog', async () => {
		await openMediaTool({
			toolId: 'speed',
			items: [clip, { id: 'm1', kind: 'mesh', url: '/m.glb', filename: 'm.glb' }],
			scope: 'library'
		});
		await settle();
		expect((hosts()[0].items as unknown[]).length).toBe(1);
	});

	it('opens a plugin tool with params in the same dialog', async () => {
		await openMediaTool({ toolId: 'boomerang.loop', items: [clip], scope: 'library' });
		await settle();
		expect((hosts()[0].tool as { id: string }).id).toBe('boomerang.loop');
	});
});

describe('the editor route', () => {
	it('opens the built-in editor on the item with its own library row', async () => {
		await openMediaTool({ toolId: 'trim', items: [clip], scope: 'library' });
		await settle();

		const [props] = hosts();
		expect(props.detach).toBe(true);
		expect(props.request).toEqual({
			kind: 'trim',
			itemIndex: null,
			source: {
				url: clip.url,
				kind: 'video',
				fileName: 'c.mp4',
				key: 'row-2',
				itemId: 'row-2',
				generationFileId: null,
				storedPath: null,
				width: null,
				height: null,
				durationSeconds: 8,
				fps: null
			}
		});
	});

	it('finds the generation file of a history item so the editor can edit it straight from storage', async () => {
		listGenerationMedia.mockResolvedValue({ success: true, data: { media: [{ id: 'file-9', filename: 'c.mp4' }] } });
		await openMediaTool({
			toolId: 'trim',
			items: [{ ...clip, id: 'gen-4:0', generationId: 'gen-4', url: '/api/media/generations/gen-4/c.mp4' }],
			scope: 'history'
		});
		await settle();

		const source = (hosts()[0].request as { source: { itemId: string | null; generationFileId: string | null; key: string } }).source;
		expect(source.generationFileId).toBe('file-9');
		expect(source.itemId).toBeNull();
		expect(source.key).toBe('gen-4:0');
	});

	it('says so and opens nothing when the media cannot be found', async () => {
		const opened = await openMediaTool({
			toolId: 'trim',
			items: [{ ...clip, id: 'gen-4:0', generationId: 'gen-4', url: '/api/media/generations/gen-4/c.mp4' }],
			scope: 'history'
		});
		expect(opened).toBe(false);
		expect(toastTexts()[0]).toContain('not in your library');
		expect(hosts()).toEqual([]);
	});

	it('needs exactly one item', async () => {
		const opened = await openMediaTool({ toolId: 'trim', items: [clip, { ...clip, id: 'r9' }], scope: 'library' });
		expect(opened).toBe(false);
		expect(toastTexts()).toEqual(['Select one item to use this tool.']);
	});

	it('turns an editor result into items for the caller, replaced or not', async () => {
		const onResult = vi.fn();
		await openMediaTool({ toolId: 'trim', items: [clip], scope: 'library', onResult });
		await settle();

		press('[data-stub-result-editor]');
		expect(onResult).toHaveBeenLastCalledWith([expect.objectContaining({ id: 'out-1' })], { toolId: 'trim', replaced: true });

		press('[data-stub-result-editor-items]');
		expect(onResult.mock.calls.at(-1)![0]).toHaveLength(2);
		expect(onResult.mock.calls.at(-1)![1]).toEqual({ toolId: 'trim', replaced: false });
	});

	it('does not pass a mask on as an edited item', async () => {
		const onResult = vi.fn();
		await openMediaTool({ toolId: 'trim', items: [clip], scope: 'library', onResult });
		await settle();
		press('[data-stub-result-mask]');
		expect(onResult).not.toHaveBeenCalled();
	});
});

describe('the plugin component route', () => {
	it('mounts the host with a tool that carries the component and a context built from the items', async () => {
		await openMediaTool({
			toolId: A1111_TOOL.id,
			items: [{ ...image, id: 'gen-1:0', generationId: 'gen-1', paramIndex: 0 }],
			scope: 'history'
		});
		await settle();

		const [props] = hosts();
		const tool = props.tool as { id: string; componentRef: string; source: string; scopes: string[] };
		expect(tool.id).toBe(A1111_TOOL.id);
		expect(tool.componentRef).toBe('plugin:a1111-metadata-export:A1111ExportTool.js');
		expect(tool.source).toBe('plugin');
		expect(tool.scopes).toEqual(['history']);
		const context = props.context as { scope: string; items: { id: string; generationId: string }[]; generationIds: string[]; kinds: string[] };
		expect(context.scope).toBe('history');
		expect(context.items.map((item) => item.id)).toEqual(['gen-1:0']);
		expect(context.generationIds).toEqual(['gen-1']);
		expect(context.kinds).toEqual(['image']);
	});

	it('uses the context the caller already has', async () => {
		const context = {
			scope: 'history' as const,
			items: [],
			kinds: new Set(['image' as const]),
			collectionId: 'c1',
			generations: [],
			generationIds: ['g7', 'g8'],
			files: []
		};
		await openMediaTool({ toolId: A1111_TOOL.id, items: [image], scope: 'history', context });
		await settle();
		expect((hosts()[0].context as { generationIds: string[] }).generationIds).toEqual(['g7', 'g8']);
	});

	it('closes the host and reports an empty result when the component finishes', async () => {
		const onResult = vi.fn();
		await openMediaTool({ toolId: A1111_TOOL.id, items: [image], scope: 'history', onResult });
		await settle();

		press('[data-stub-done]');
		await settle();

		expect(onResult).toHaveBeenCalledWith([], { toolId: A1111_TOOL.id, replaced: false });
		expect(document.body.querySelector('[data-host-stub]')).toBeNull();
	});

	it('tells the caller when the component is closed without finishing', async () => {
		const onClose = vi.fn();
		const onResult = vi.fn();
		await openMediaTool({ toolId: A1111_TOOL.id, items: [image], scope: 'history', onClose, onResult });
		await settle();
		press('[data-stub-close]');
		await settle();
		expect(onClose).toHaveBeenCalledTimes(1);
		expect(onResult).not.toHaveBeenCalled();
	});
});
