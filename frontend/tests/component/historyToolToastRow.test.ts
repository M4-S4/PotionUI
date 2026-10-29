import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { get } from 'svelte/store';
import { CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const openMediaTool = vi.fn();
const listMediaTools = vi.fn();

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));
vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: (...args: unknown[]) => listMediaTools(...args) }));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));

const { default: MediaToolRow } = await import('$lib/media-tools/MediaToolRow.svelte');
const { SAVED_TO_LIBRARY_ANNOUNCE } = await import('$lib/media-tools/announce');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { registerEntryTools } = await import('$lib/tools/entryTools');
const { toasts } = await import('$lib/stores/toast');

registerEntryTools();

const video = {
	id: 'row-1',
	kind: 'video' as const,
	url: '/api/media/uploads/clip.mp4',
	filename: 'clip.mp4',
	width: 640,
	height: 480,
	itemId: 'row-1'
};
const image = { ...video, kind: 'image' as const, url: '/api/media/uploads/a.png', filename: 'a.png' };

let cleanup: (() => void) | undefined;

async function render(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(MediaToolRow, {
		target,
		props: { items: [video], scope: 'library', pageLabel: 'Library', pageActions: [], ...props }
	});
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	await settle();
	flushSync();
	return target;
}

function button(root: ParentNode, text: string): HTMLButtonElement {
	return Array.from(root.querySelectorAll('button')).find((entry) => (entry.textContent ?? '').trim() === text)!;
}

beforeEach(() => {
	openMediaTool.mockReset();
	listMediaTools.mockReset().mockResolvedValue(CORE_TOOLS);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
	for (const entry of get(toasts)) toasts.remove(entry.id);
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

describe('the row on History', () => {
	it('opens a tool with the saved-to-Library toast and its link', async () => {
		const root = await render({ scope: 'history' });

		button(root, 'Trim').click();

		expect(openMediaTool).toHaveBeenCalledTimes(1);
		expect(openMediaTool.mock.calls[0][0]).toMatchObject({ toolId: 'trim', scope: 'history', announce: SAVED_TO_LIBRARY_ANNOUNCE });
	});

	it('opens a tool on the Library with the default toast', async () => {
		const root = await render({ scope: 'library' });

		button(root, 'Trim').click();

		expect(openMediaTool.mock.calls[0][0].scope).toBe('library');
		expect(openMediaTool.mock.calls[0][0].announce).toBeUndefined();
	});

	it('posts the one toast for an image saved from the editor, and tells the page', async () => {
		const onResult = vi.fn();
		const root = await render({ scope: 'history', items: [image], onResult });
		button(root, 'Edit image').click();
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-result]')!.click();
		flushSync();

		const saved = get(toasts).filter((entry) => entry.type === 'success');
		expect(saved).toHaveLength(1);
		expect(saved[0].action?.label).toBe('Open Library');
		expect(onResult).toHaveBeenCalledTimes(1);
	});

	it('posts no toast of its own for an image saved from the editor on the Library', async () => {
		const root = await render({ scope: 'library', items: [image] });
		button(root, 'Edit image').click();
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-result]')!.click();
		flushSync();

		expect(get(toasts)).toEqual([]);
	});
});
