// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { settle } from './helpers/settle';

vi.mock('$lib/tools/pluginTools', () => ({ retryMediaToolCatalog: vi.fn() }));

import ToolsMenu from '$lib/components/tools/ToolsMenu.svelte';
import type { MediaTool, MediaToolAvailability, MediaToolGroup } from '$lib/tools/tools';

function tool(id: string, extra: Partial<MediaTool> = {}): MediaTool {
	return {
		id,
		label: id.toUpperCase(),
		icon: 'extension',
		category: 'edit',
		source: 'core',
		scopes: ['history', 'library'],
		applies: () => ({ enabled: true }),
		...extra
	};
}

type Entry = { tool: MediaTool; availability: MediaToolAvailability };
const on = (id: string, extra: Partial<MediaTool> = {}): Entry => ({ tool: tool(id, extra), availability: { enabled: true } });
const off = (id: string, extra: Partial<MediaTool> = {}): Entry => ({
	tool: tool(id, extra),
	availability: { enabled: false, reason: 'Not now' }
});
const group = (id: string, order: number, tools: Entry[]): MediaToolGroup => ({
	category: { id, label: id, order },
	tools
});

const GROUPS = [
	group('edit', -20, [on('crop'), on('rotate', { shortcut: 'R' }), off('flip', { shortcut: 'L' })]),
	group('compose', 20, [on('edit-image', { category: 'compose', shortcut: 'E' })]),
	group('export', 30, [on('download', { category: 'export', shortcut: 'D' })]),
	group('organize', 40, [on('favorite', { category: 'organize', shortcut: 'F' })]),
	group('danger', 99, [on('delete', { category: 'danger', shortcut: 'Backspace' })])
];

let cleanup: (() => void) | undefined;

function render(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const handlers = { onToggle: vi.fn(), onClose: vi.fn(), onPick: vi.fn() };
	const instance = mount(ToolsMenu, {
		target,
		props: { open: true, scope: 'library', groups: GROUPS, ...handlers, ...props }
	});
	flushSync();
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return handlers;
}

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = window) {
	target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
	flushSync();
}

const picked = (handlers: { onPick: ReturnType<typeof vi.fn> }) =>
	handlers.onPick.mock.calls.map((call) => (call[0] as MediaTool).id);

describe('letters in an open menu', () => {
	it('trigger the row that shows them, whichever group it is in', async () => {
		for (const [key, id] of [['e', 'edit-image'], ['d', 'download'], ['f', 'favorite'], ['Backspace', 'delete']] as const) {
			cleanup?.();
			document.body.innerHTML = '';
			const handlers = render();
			await settle(2);
			press(key);
			expect(picked(handlers)).toEqual([id]);
		}
	});

	it('trigger a row inside an open submenu', async () => {
		const handlers = render();
		await settle(2);
		document.body.querySelector<HTMLElement>('[data-submenu-row="edit"]')!.click();
		flushSync();
		await settle(2);

		press('r');

		expect(picked(handlers)).toEqual(['rotate']);
	});

	it('only trigger what the open submenu shows', async () => {
		const handlers = render();
		await settle(2);
		document.body.querySelector<HTMLElement>('[data-submenu-row="edit"]')!.click();
		flushSync();
		await settle(2);

		press('e');

		expect(picked(handlers)).toEqual([]);
	});

	it('skip a disabled row', async () => {
		const handlers = render();
		await settle(2);
		document.body.querySelector<HTMLElement>('[data-submenu-row="edit"]')!.click();
		flushSync();
		await settle(2);

		press('l');

		expect(picked(handlers)).toEqual([]);
	});

	it('do nothing while the menu is closed', async () => {
		const handlers = render({ open: false });
		await settle(2);

		press('e');

		expect(picked(handlers)).toEqual([]);
	});

	it('do nothing while typing in a field', async () => {
		const handlers = render();
		await settle(2);
		const input = document.createElement('input');
		document.body.appendChild(input);
		input.focus();

		press('e', {}, input);

		expect(picked(handlers)).toEqual([]);
	});

	it('do nothing with a modifier held', async () => {
		const handlers = render();
		await settle(2);

		press('e', { ctrlKey: true });

		expect(picked(handlers)).toEqual([]);
	});

	it('work in the media field menu too', async () => {
		const handlers = render({
			variant: 'field',
			scope: 'field',
			groups: [group('edit', -20, [on('crop', { shortcut: 'C', scopes: ['field'] })])]
		});
		await settle(2);

		press('c');

		expect(picked(handlers)).toEqual(['crop']);
	});
});
