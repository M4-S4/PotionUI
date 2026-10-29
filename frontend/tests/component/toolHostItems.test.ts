// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { settle } from './helpers/settle';

const resolvePluginComponent = vi.fn();

vi.mock('$lib/plugin-api/componentResolver', () => ({
	resolvePluginComponent: (...args: unknown[]) => resolvePluginComponent(...args)
}));
vi.mock('$lib/stores/toast', () => ({ toasts: { error: vi.fn(), success: vi.fn(), info: vi.fn(), show: vi.fn() } }));

const { default: ToolHost } = await import('$lib/components/tools/ToolHost.svelte');
const { default: PluginToolStub } = await import('./fixtures/PluginToolStub.svelte');
const { default: CoreModalStub } = await import('./fixtures/CoreModalStub.svelte');
const { buildHistoryToolContext, buildLibraryToolContext } = await import('$lib/tools/tools');

let cleanup: (() => void) | undefined;

const pluginTool = {
	id: 'a1111-metadata-export.export-zip',
	label: 'Download with A1111 metadata',
	icon: 'download',
	category: 'plugins',
	source: 'plugin' as const,
	scopes: ['history' as const],
	componentRef: 'plugin:a1111-metadata-export:A1111ExportTool.js',
	applies: () => ({ enabled: true })
};

function render(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(ToolHost, {
		target,
		props: { tool: pluginTool, context: null, onClose: vi.fn(), onDone: vi.fn(), ...props }
	});
	flushSync();
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
}

const stub = () => document.body.querySelector<HTMLElement>('[data-plugin-stub]');
const items = () => JSON.parse(stub()!.getAttribute('data-items')!);

function generation(id: string, files: Array<Record<string, unknown>>) {
	return {
		id,
		form_data: {},
		status: 'completed',
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files,
		rating: 0,
		is_favorite: false
	} as never;
}

beforeEach(() => {
	resolvePluginComponent.mockReset();
	resolvePluginComponent.mockResolvedValue(PluginToolStub);
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

describe('ToolHost plugin components', () => {
	it('resolves the component the tool names', async () => {
		render({ context: buildHistoryToolContext([], [], null) });
		await settle();

		expect(resolvePluginComponent).toHaveBeenCalledWith('a1111-metadata-export', 'A1111ExportTool.js');
	});

	it('passes the selected items with id, kind, url, filename, width and height', async () => {
		const context = buildHistoryToolContext(
			[
				generation('g1', [
					{ id: 5, file_path: 'out/g1/a.png', file_type: 'image', is_final: true, width: 640, height: 480, created_at: 'x' }
				])
			],
			['g1'],
			null
		);
		render({ context });
		await settle();
		flushSync();

		expect(items()).toEqual([
			{
				id: 'g1:0',
				kind: 'image',
				url: '/api/media/generations/g1/a.png',
				filename: 'a.png',
				width: 640,
				height: 480
			}
		]);
	});

	it('passes only those fields, not the file ids a plugin has no use for', async () => {
		const context = buildHistoryToolContext(
			[
				generation('g1', [
					{ id: 5, file_path: 'out/g1/a.png', file_type: 'image', is_final: true, width: 8, height: 9, created_at: 'x' }
				])
			],
			['g1'],
			null
		);
		render({ context });
		await settle();
		flushSync();

		expect(Object.keys(items()[0]).sort()).toEqual(['filename', 'height', 'id', 'kind', 'url', 'width']);
	});

	it('tells a library-scoped tool what is selected, which it had no way to know before', async () => {
		const context = buildLibraryToolContext(
			[
				{ id: 'u1', filename: 'u1.mp4', original_filename: 'clip.mp4', media_type: 'video', url: '/u/u1.mp4', width: 10, height: 20, tags: [] },
				{ id: 'u2', filename: 'u2.png', media_type: 'image', url: '/u/u2.png', tags: [] }
			] as never,
			null
		);
		render({ context, tool: { ...pluginTool, scopes: ['library'] } });
		await settle();
		flushSync();

		expect(items().map((entry: { id: string; kind: string }) => [entry.id, entry.kind])).toEqual([
			['u1', 'video'],
			['u2', 'image']
		]);
		expect(items()[0].filename).toBe('clip.mp4');
	});

	it('still passes the generation ids and the generations for existing plugins', async () => {
		const context = buildHistoryToolContext(
			[generation('g1', [{ id: 5, file_path: 'out/g1/a.png', file_type: 'image', is_final: true, created_at: 'x' }])],
			['g1'],
			null
		);
		render({ context });
		await settle();
		flushSync();

		expect(JSON.parse(stub()!.getAttribute('data-generation-ids')!)).toEqual(['g1']);
		expect(stub()!.getAttribute('data-generation-count')).toBe('1');
	});

	it('passes an empty list rather than nothing when nothing is selected', async () => {
		render({ context: buildLibraryToolContext([], null) });
		await settle();
		flushSync();

		expect(items()).toEqual([]);
	});

	it('forwards what the plugin reports back', async () => {
		const onDone = vi.fn();
		const onClose = vi.fn();
		render({ context: buildLibraryToolContext([], null), onDone, onClose });
		await settle();
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-done]')!.click();
		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();

		expect(onDone).toHaveBeenCalledTimes(1);
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});

describe('ToolHost core components', () => {
	const coreTool = {
		id: 'look:noir',
		label: 'Noir',
		icon: 'palette',
		category: 'effects',
		source: 'core' as const,
		scopes: ['library' as const],
		component: CoreModalStub as never,
		applies: () => ({ enabled: true })
	};

	const modal = () => document.body.querySelector<HTMLElement>('[data-core-stub]');

	it('mounts the component with the context it was picked for', () => {
		render({ tool: coreTool, context: buildLibraryToolContext([], null) });

		expect(modal()).toBeTruthy();
		expect(modal()!.getAttribute('data-item-count')).toBe('0');
	});

	it('hands the tool its own props, which is how a recent look names itself', () => {
		render({
			tool: { ...coreTool, componentProps: { lookId: 'retro-tape:vhs' } },
			context: buildLibraryToolContext([], null)
		});

		expect(modal()!.getAttribute('data-look-id')).toBe('retro-tape:vhs');
	});

	it('passes no extra props for a tool that declares none', () => {
		render({ tool: coreTool, context: buildLibraryToolContext([], null) });

		expect(modal()!.getAttribute('data-look-id')).toBe('');
	});

	it('cannot override what the host owns with a prop of the tool', () => {
		const onClose = vi.fn();
		render({
			tool: { ...coreTool, componentProps: { onClose: () => undefined } },
			context: buildLibraryToolContext([], null),
			onClose
		});

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();

		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
