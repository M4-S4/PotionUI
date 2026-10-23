// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		searchPhrasebook: vi.fn(),
		getFileURL: (fileId: string, size: string) => `/media/${size}/${fileId}`
	}
}));

const { api } = await import('$lib/services/api/index');
const { default: InlineChipEditor } = await import('$lib/components/InlineChipEditor.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { flushSync } = await import('svelte');

let target: HTMLDivElement;
let component: ReturnType<typeof createClassComponent> | undefined;

interface EditorInstance {
	insertPhrasebookTrigger: () => void;
}

function mountEditor(props: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: InlineChipEditor as never,
		target,
		props: { value: '', chips: {}, variant: 'segment-composer', borderless: true, ...props }
	});
	return component as unknown as EditorInstance;
}

async function openPhrasebookBrowseModal(props: Record<string, unknown> = {}) {
	const editor = mountEditor(props);
	editor.insertPhrasebookTrigger();
	flushSync();
	await vi.advanceTimersByTimeAsync(210);
	flushSync();

	const editorEl = document.querySelector('.inline-chip-editor')!;
	const textNode = editorEl.firstChild as Text;
	textNode.textContent = '#li';
	const sel = window.getSelection()!;
	const range = document.createRange();
	range.setStart(textNode, 3);
	range.collapse(true);
	sel.removeAllRanges();
	sel.addRange(range);
	editorEl.dispatchEvent(new Event('input', { bubbles: true }));
	flushSync();
	await vi.advanceTimersByTimeAsync(210);
	flushSync();

	editorEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
	flushSync();

	return editorEl;
}

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	document.body.innerHTML = '';
	vi.restoreAllMocks();
	vi.useRealTimers();
});

describe('PromptPickerBrowseModal', () => {
	it('shows a context strip with words around the caret and the trigger marker in place', async () => {
		vi.mocked(api.searchPhrasebook).mockResolvedValue({
			success: true,
			data: { child_categories: [], values: [] }
		} as never);

		mountEditor({ value: 'a woman standing near the harbor at dusk wrapped in #li light, low key fog' });
		flushSync();

		const editorEl = document.querySelector('.inline-chip-editor')!;
		const textNode = Array.from(editorEl.childNodes).find(
			(n) => n.nodeType === Node.TEXT_NODE && n.textContent?.includes('#li')
		) as Text;
		const hashIndex = textNode.textContent!.indexOf('#li') + 3;
		const range = document.createRange();
		range.setStart(textNode, hashIndex);
		range.collapse(true);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		editorEl.dispatchEvent(new Event('input', { bubbles: true }));
		flushSync();
		await vi.advanceTimersByTimeAsync(210);
		flushSync();

		editorEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
		flushSync();

		const context = document.querySelector('.picker-context');
		expect(context).not.toBeNull();
		expect(context!.textContent).toContain('harbor at dusk wrapped in');
		expect(context!.textContent).toContain('light, low key fog');
		expect(context!.querySelector('.k')!.textContent).toContain('#li');
	});

	it('Insert calls the same insertion path as picking the row in the dropdown, and closes the modal', async () => {
		vi.mocked(api.searchPhrasebook).mockResolvedValue({
			success: true,
			data: {
				child_categories: [],
				values: [
					{
						id: 'v1',
						category_id: 'c1',
						category_path: 'lighting',
						label: 'golden hour',
						value: 'golden hour lighting',
						sort_order: 0,
						created_at: '',
						updated_at: ''
					}
				]
			}
		} as never);

		await openPhrasebookBrowseModal();

		const row = document.querySelector<HTMLElement>('.picker-vrow');
		expect(row).not.toBeNull();
		row!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
		flushSync();

		const insertButton = Array.from(document.querySelectorAll('button')).find((b) => b.textContent?.includes('Insert'));
		expect(insertButton).toBeTruthy();
		insertButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
		flushSync();

		expect(document.querySelector('[role="dialog"]')).toBeNull();
		expect(target.querySelector('.phrase-chip')).not.toBeNull();
		expect(target.querySelector('.phrase-chip')!.textContent).toContain('golden hour');
	});

	it('Esc closes the modal without touching the typed trigger text', async () => {
		vi.mocked(api.searchPhrasebook).mockResolvedValue({
			success: true,
			data: { child_categories: [], values: [] }
		} as never);

		const editorEl = await openPhrasebookBrowseModal();

		const searchInput = document.querySelector<HTMLInputElement>('.picker-modal-search input');
		expect(searchInput).not.toBeNull();
		searchInput!.value = 'something else entirely';
		searchInput!.dispatchEvent(new Event('input', { bubbles: true }));
		flushSync();

		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
		flushSync();

		expect(document.querySelector('[role="dialog"]')).toBeNull();
		expect(editorEl.textContent).toContain('#li');
		expect(editorEl.textContent).not.toContain('something else entirely');
	});
});

describe('PromptPickerBrowseModal — categories column for every trigger, browsing fresh', () => {
	it('@ references: groups on the left, that group\'s grid on the right, no drill-down step', async () => {
		const editor = mountEditor({
			promptResources: [
				{ field: 'references', kind: 'image', label: 'Pictures', token: '<Picture @>' },
				{ field: 'reference_videos', kind: 'video', label: 'Videos', token: '<Video @>' }
			],
			resourceFieldValues: {
				references: [{ relative_path: 'a.png', name: 'a.png' }],
				reference_videos: []
			}
		}) as unknown as { insertResourceTrigger: () => void };
		editor.insertResourceTrigger();
		flushSync();

		const editorEl = document.querySelector('.inline-chip-editor')!;
		editorEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
		flushSync();

		const groups = Array.from(document.querySelectorAll<HTMLElement>('.picker-tree-row'));
		expect(groups.some((g) => g.textContent?.includes('Pictures'))).toBe(true);
		expect(groups.some((g) => g.textContent?.includes('Videos'))).toBe(true);

		const rows = Array.from(document.querySelectorAll<HTMLElement>('.picker-gitem'));
		expect(rows.length).toBe(1);
		expect(rows[0]?.textContent).toContain('a.png');
	});

	it('/ syntax: lists the tokens in a one-pane dialog, naming the group in the header', async () => {
		const editor = mountEditor({
			promptSyntax: [{ token: 'BREAK', kind: 'marker', help: 'Splits into CLIP chunks' }]
		}) as unknown as { insertSyntaxTrigger: () => void };
		editor.insertSyntaxTrigger();
		flushSync();

		const editorEl = document.querySelector('.inline-chip-editor')!;
		editorEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
		flushSync();

		expect(document.querySelectorAll('.picker-tree-row')).toHaveLength(0);
		expect(document.querySelector<HTMLElement>('[role="dialog"]')!.className).toContain('md:w-[600px]');
		expect(document.querySelector('[data-segment-modal-context]')!.textContent).toContain('Tokens');
		expect(document.querySelector('.picker-values')?.textContent).toContain('BREAK');
	});

	it('$ variables: the primary names the variable it inserts', async () => {
		const editor = mountEditor({
			variables: { mood: { type: 'text', value: 'calm' }, lens: { type: 'text', value: '85mm' } }
		}) as unknown as { insertVariableTrigger: () => void };
		editor.insertVariableTrigger();
		flushSync();
		const editorEl = document.querySelector('.inline-chip-editor')!;
		editorEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }));
		flushSync();
		const primary = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-segment-modal-end] button')).at(-1)!;
		expect(primary.textContent?.replace(/\s+/g, ' ').trim()).toMatch(/^Insert \$(mood|lens) Enter$/);
		expect(document.querySelectorAll('.picker-tree-row')).toHaveLength(0);
	});
});
