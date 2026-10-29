// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { BOOMERANG_TOOL, CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

const openMediaTool = vi.fn();
const listMediaTools = vi.fn();

vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));
vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: (...args: unknown[]) => listMediaTools(...args) }));
vi.mock('$lib/stores/toast', () => ({
	toasts: { show: vi.fn(), success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() }
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		listLibraryItems: vi.fn().mockResolvedValue({ success: false }),
		getLibraryFacets: vi.fn().mockResolvedValue({ success: false }),
		listCollections: vi.fn().mockResolvedValue({ success: false })
	}
}));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));

const { default: MediaToolRow } = await import('$lib/media-tools/MediaToolRow.svelte');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { registerEntryTools } = await import('$lib/tools/entryTools');
const { recentLooks } = await import('$lib/filters/recentLooks');

registerEntryTools();

type Kind = 'image' | 'video' | 'audio' | 'mesh';

function item(kind: Kind, extra: Record<string, unknown> = {}) {
	return {
		id: 'row-1',
		kind,
		url: `/api/media/uploads/row-1.${kind}`,
		filename: kind === 'image' ? 'photo.png' : `clip.${kind}`,
		width: 640,
		height: 480,
		itemId: 'row-1',
		...extra
	};
}

let cleanup: (() => void) | undefined;

function render(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(MediaToolRow, {
		target,
		props: { items: [item('image')], scope: 'library', pageLabel: 'Library', pageActions: [], ...props }
	});
	flushSync();
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return target;
}

const labels = (root: ParentNode) => {
	const heading = root.querySelector<HTMLElement>('span.uppercase');
	const buttons = Array.from(root.querySelectorAll('button')).map((button) => (button.textContent ?? '').trim());
	return heading ? [(heading.textContent ?? '').trim(), ...buttons] : buttons;
};

const group = (root: ParentNode, name: string) => root.querySelector(`[data-row-group="${name}"]`);

async function renderSettled(props: Record<string, unknown>) {
	const target = render(props);
	await settle();
	flushSync();
	return target;
}

beforeEach(() => {
	recentLooks.set([]);
	openMediaTool.mockReset();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue(CORE_TOOLS);
	resetMediaToolCatalog();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

describe('MediaToolRow groups', () => {
	it('lists an image Edit row with the built-in editors, Edit image and More tools', async () => {
		const root = await renderSettled({ items: [item('image')] });

		expect(labels(group(root, 'edit')!)).toEqual(['Edit', 'Crop', 'Resize', 'Edit image', 'More tools']);
	});

	it('lists a video Edit row without Edit image', async () => {
		const root = await renderSettled({ items: [item('video')] });

		expect(labels(group(root, 'edit')!)).toEqual(['Edit', 'Trim', 'Crop', 'Resize', 'Frame', 'More tools']);
	});

	it('lists an audio Edit row', async () => {
		const root = await renderSettled({ items: [item('audio')] });

		expect(labels(group(root, 'edit')!)).toEqual(['Edit', 'Trim', 'Split', 'More tools']);
	});

	it('shows the Effects row for an image only', async () => {
		const root = await renderSettled({ items: [item('image')] });
		expect(labels(group(root, 'effects')!)).toEqual(['Effects', 'Apply a look…']);
		cleanup?.();

		const video = await renderSettled({ items: [item('video')] });
		expect(group(video, 'effects')).toBeNull();
		cleanup?.();

		const audio = await renderSettled({ items: [item('audio')] });
		expect(group(audio, 'effects')).toBeNull();
	});

	it('gives both pages the same Edit and Effects rows for the same media', async () => {
		const library = await renderSettled({ scope: 'library', items: [item('video')] });
		const libraryEdit = labels(group(library, 'edit')!);
		cleanup?.();

		const history = await renderSettled({
			scope: 'history',
			pageLabel: 'Generation',
			items: [item('video', { itemId: undefined, generationId: 'g1', generationFileId: '4', paramIndex: 0 })]
		});

		expect(labels(group(history, 'edit')!)).toEqual(libraryEdit);
	});

	it('renders the page group under the label it was given, as a labelled group', async () => {
		const root = await renderSettled({
			pageLabel: 'Generation',
			pageActions: [{ id: 'download', label: 'Download', icon: 'download', onclick: vi.fn() }]
		});

		const page = group(root, 'page')!;
		expect(page.getAttribute('aria-label')).toBe('Generation actions');
		expect(labels(page)).toEqual(['Generation', 'Download']);
	});

	it('keeps the page actions in the order they were passed', async () => {
		const root = await renderSettled({
			pageActions: [
				{ id: 'auto', label: 'Auto-file items like this…', icon: 'sparkles', onclick: vi.fn() },
				{ id: 'delete', label: 'Delete', icon: 'trash', tone: 'danger', onclick: vi.fn() },
				{ id: 'download', label: 'Download', icon: 'download', onclick: vi.fn() }
			]
		});

		expect(labels(group(root, 'page')!).slice(1)).toEqual([
			'Auto-file items like this…',
			'Delete',
			'Download'
		]);
	});

	it('draws a danger action as a red button and a normal one as secondary', async () => {
		const root = await renderSettled({
			pageActions: [
				{ id: 'delete', label: 'Delete', icon: 'trash', tone: 'danger', onclick: vi.fn() },
				{ id: 'download', label: 'Download', icon: 'download', onclick: vi.fn() }
			]
		});

		const buttons = Array.from(group(root, 'page')!.querySelectorAll('button'));
		expect(buttons[0].className).toContain('bg-danger-solid');
		expect(buttons[1].className).not.toContain('bg-danger-solid');
		expect(buttons[1].className).toContain('bg-surface-3');
	});

	it('renders nothing for media nothing edits when there are no page actions', async () => {
		const root = await renderSettled({ items: [item('mesh')], pageActions: [] });

		expect(root.querySelector('[data-media-tool-row]')).toBeNull();
	});

	it('still shows the page actions for media nothing edits', async () => {
		const root = await renderSettled({
			items: [item('mesh')],
			pageActions: [{ id: 'download', label: 'Download', icon: 'download', onclick: vi.fn() }]
		});

		expect(group(root, 'edit')).toBeNull();
		expect(labels(group(root, 'page')!)).toEqual(['Library', 'Download']);
	});
});

describe('MediaToolRow More tools', () => {
	async function openMore(kind: Kind = 'image', extra: Record<string, unknown> = {}) {
		const root = await renderSettled({ items: [item(kind, extra)] });
		const trigger = Array.from(root.querySelectorAll('button')).find((button) =>
			(button.textContent ?? '').includes('More tools')
		)!;
		trigger.click();
		flushSync();
		await settle(2);
		flushSync();
		return root;
	}

	const menuRows = () =>
		Array.from(document.body.querySelectorAll<HTMLElement>('[role="menu"] [data-row-tool]'));

	it('holds the tools without a built-in editor', async () => {
		await openMore('image');

		expect(menuRows().map((row) => row.getAttribute('data-row-tool'))).toEqual([
			'rotate',
			'flip',
			'fit_canvas',
			'color_key',
			'crop_subject'
		]);
	});

	it('shows a disabled tool with the reason it is disabled', async () => {
		await openMore('image', { filename: 'photo.jpg' });

		const row = menuRows().find((entry) => entry.getAttribute('data-row-tool') === 'crop_subject')!;
		expect(row.getAttribute('aria-disabled')).toBe('true');
		expect(row.textContent).toContain('Needs a transparent image');
	});

	it('names the plugin on a second line for a plugin tool', async () => {
		listMediaTools.mockResolvedValue([...CORE_TOOLS, BOOMERANG_TOOL]);
		await openMore('video');

		const row = menuRows().find((entry) => entry.getAttribute('data-row-tool') === 'boomerang.loop')!;
		expect(row.textContent).toContain('Boomerang');
		expect(row.querySelectorAll('span.text-fg-subtle')).toHaveLength(1);
		const core = menuRows().find((entry) => entry.getAttribute('data-row-tool') === 'speed')!;
		expect(core.querySelectorAll('span.text-fg-subtle')).toHaveLength(0);
	});

	it('opens the chosen tool and closes the menu', async () => {
		await openMore('image');

		menuRows()[0].click();
		flushSync();

		expect(openMediaTool).toHaveBeenCalledTimes(1);
		expect(openMediaTool.mock.calls[0][0]).toMatchObject({ toolId: 'rotate', scope: 'library' });
		expect(document.body.querySelector('[role="menu"]')).toBeNull();
	});

	it('does nothing for a disabled tool', async () => {
		await openMore('image', { filename: 'photo.jpg' });

		menuRows().find((entry) => entry.getAttribute('data-row-tool') === 'crop_subject')!.click();
		flushSync();

		expect(openMediaTool).not.toHaveBeenCalled();
	});
});

describe('MediaToolRow running a tool', () => {
	function button(root: ParentNode, text: string) {
		return Array.from(root.querySelectorAll('button')).find((entry) => (entry.textContent ?? '').trim() === text)!;
	}

	it('opens a catalog tool on the item the row shows, in the scope of the page', async () => {
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'Crop').click();

		expect(openMediaTool).toHaveBeenCalledTimes(1);
		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('crop');
		expect(call.scope).toBe('library');
		expect(call.items).toHaveLength(1);
		expect(call.items[0]).toMatchObject({ id: 'row-1', itemId: 'row-1', kind: 'image' });
	});

	it('hands a generated file to the tool with its generation file id', async () => {
		const root = await renderSettled({
			scope: 'history',
			items: [item('image', { itemId: undefined, generationId: 'g1', generationFileId: '42', paramIndex: 0 })]
		});

		button(root, 'Crop').click();

		const call = openMediaTool.mock.calls[0][0];
		expect(call.scope).toBe('history');
		expect(call.items[0]).toMatchObject({ generationId: 'g1', generationFileId: '42' });
	});

	it('passes what the tool produced back to the page', async () => {
		const onResult = vi.fn();
		const root = await renderSettled({ items: [item('image')], onResult });
		button(root, 'Resize').click();

		const produced = [{ id: 'new-1', filename: 'photo-resized.png' }];
		const info = { toolId: 'resize', replaced: false };
		await openMediaTool.mock.calls[0][0].onResult(produced, info);

		expect(onResult).toHaveBeenCalledWith(produced, info);
	});

	it('runs a page action when it is pressed', async () => {
		const onclick = vi.fn();
		const root = await renderSettled({
			pageActions: [{ id: 'download', label: 'Download', icon: 'download', onclick }]
		});

		button(root, 'Download').click();

		expect(onclick).toHaveBeenCalledTimes(1);
	});

	it('disables a page action while it is loading', async () => {
		const onclick = vi.fn();
		const root = await renderSettled({
			pageActions: [{ id: 'copy', label: 'Copy to Library', icon: 'photo', loading: true, onclick }]
		});

		button(root, 'Copy to Library').click();

		expect(button(root, 'Copy to Library').disabled).toBe(true);
		expect(onclick).not.toHaveBeenCalled();
	});

	it('loads the catalog by itself so a page that never did still shows tools', async () => {
		await renderSettled({ items: [item('image')] });

		expect(listMediaTools).toHaveBeenCalled();
	});
});

describe('MediaToolRow image editor', () => {
	function button(root: ParentNode, text: string) {
		return Array.from(root.querySelectorAll('button')).find((entry) => (entry.textContent ?? '').trim() === text)!;
	}

	const stub = () => document.body.querySelector<HTMLElement>('[data-editor-stub]');

	it('opens the image editor on the item for Edit image', async () => {
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'Edit image').click();
		flushSync();

		expect(stub()?.getAttribute('data-kind')).toBe('paint');
		expect(stub()?.getAttribute('data-url')).toBe('/api/media/uploads/row-1.image');
		expect(stub()?.getAttribute('data-initial-tool')).toBe('');
	});

	it('opens the same editor with the Filters tool for Apply a look', async () => {
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'Apply a look…').click();
		flushSync();

		expect(stub()?.getAttribute('data-kind')).toBe('paint');
		expect(stub()?.getAttribute('data-initial-tool')).toBe('filters');
	});

	it('returns to the default tool when Edit image follows a look', async () => {
		const root = await renderSettled({ items: [item('image')] });
		button(root, 'Apply a look…').click();
		flushSync();
		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		flushSync();

		button(root, 'Edit image').click();
		flushSync();

		expect(stub()?.getAttribute('data-initial-tool')).toBe('');
	});

	it('folds an image saved on the Library into the library store and says so, without asking the page', async () => {
		const onResult = vi.fn();
		const root = await renderSettled({ items: [item('image')], onResult });
		button(root, 'Apply a look…').click();
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-result]')!.click();
		await settle();

		expect(onResult).not.toHaveBeenCalled();
		const { toasts } = await import('$lib/stores/toast');
		expect(toasts.success).toHaveBeenCalledWith('Saved to your library as a new image');
	});

	it('tells the modal it is editing while the editor is open', async () => {
		const states: boolean[] = [];
		const root = await renderSettled({ items: [item('image')], onEditingChange: (value: boolean) => states.push(value) });
		expect(states.at(-1)).toBe(false);

		button(root, 'Edit image').click();
		flushSync();
		expect(states.at(-1)).toBe(true);

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		flushSync();
		expect(states.at(-1)).toBe(false);
	});

	it('does not open an editor for Apply a look on a video', async () => {
		const root = await renderSettled({ items: [item('video')] });

		expect(Array.from(root.querySelectorAll('button')).some((entry) => entry.textContent?.includes('Apply a look'))).toBe(
			false
		);
		expect(stub()).toBeNull();
	});
});

describe('MediaToolRow recent looks', () => {
	const looks = [
		{ id: 'noir', name: 'Noir', group: 'Black and white' },
		{ id: 'retro-tape:vhs', name: 'VHS', group: 'Retro Tape' }
	];

	function button(root: ParentNode, text: string) {
		return Array.from(root.querySelectorAll('button')).find((entry) => (entry.textContent ?? '').trim() === text)!;
	}

	const stub = () => document.body.querySelector<HTMLElement>('[data-editor-stub]');

	it('sit after Apply a look as one button each', async () => {
		recentLooks.set(looks);
		const root = await renderSettled({ items: [item('image')] });

		expect(labels(group(root, 'effects')!)).toEqual(['Effects', 'Apply a look…', 'Noir', 'VHS']);
	});

	it('are not shown for a video', async () => {
		recentLooks.set(looks);
		const root = await renderSettled({ items: [item('video')] });

		expect(group(root, 'effects')).toBeNull();
	});

	it('open the editor on that look, focused on Apply', async () => {
		recentLooks.set(looks);
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'Noir').click();
		flushSync();

		expect(stub()?.getAttribute('data-kind')).toBe('paint');
		expect(stub()?.getAttribute('data-initial-tool')).toBe('filters');
		expect(stub()?.getAttribute('data-initial-filter')).toBe('noir');
	});

	it('keep a plugin look id that has a colon in it', async () => {
		recentLooks.set(looks);
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'VHS').click();
		flushSync();

		expect(stub()?.getAttribute('data-initial-tool')).toBe('filters');
		expect(stub()?.getAttribute('data-initial-filter')).toBe('retro-tape:vhs');
	});

	it('leave Apply a look opening the editor on Filters alone', async () => {
		recentLooks.set(looks);
		const root = await renderSettled({ items: [item('image')] });

		button(root, 'Apply a look…').click();
		flushSync();

		expect(stub()?.getAttribute('data-initial-tool')).toBe('filters');
		expect(stub()?.getAttribute('data-has-initial-filter')).toBe('false');
	});

	it('appear without a reload when a look is applied while the row is open', async () => {
		const root = await renderSettled({ items: [item('image')] });
		expect(labels(group(root, 'effects')!)).toEqual(['Effects', 'Apply a look…']);

		recentLooks.set([looks[0]]);
		flushSync();

		expect(labels(group(root, 'effects')!)).toEqual(['Effects', 'Apply a look…', 'Noir']);
	});
});
