// @vitest-environment jsdom
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { flushSync, tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { conditionModal, setCaret } from './helpers/typing';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: () => ({
			get: vi.fn().mockResolvedValue({ data: { data: {} } }),
			put: vi.fn().mockResolvedValue({ data: { success: true } })
		})
	}
}));

class SilentResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

const { default: PromptSegment } = await import('$lib/components/PromptSegment.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { promptSegmentActionPins } = await import('$lib/stores/promptSegmentActionPins');

const variables: VariablesMap = {
	x: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] }
};
const DEFAULT = '<<if $x = "calm">>text<<end>>';

let cards: ReturnType<typeof createClassComponent>[] = [];

beforeEach(() => {
	(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = SilentResizeObserver;
	promptSegmentActionPins.reset();
});

afterEach(() => {
	cards.forEach((card) => card.$destroy());
	cards = [];
	document.body.innerHTML = '';
});

function mountSegment(content: string) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const segment = { id: 'seg-1', content, type: 'content', chips: {}, enabled: true };
	const card = createClassComponent({
		component: PromptSegment as never,
		target,
		props: { segment, index: 0, total: 1, variables }
	});
	cards.push(card);
	const values: string[] = [];
	const handle = card as unknown as {
		$on: (n: string, cb: (e: CustomEvent) => void) => void;
		$set: (p: Record<string, unknown>) => void;
	};
	handle.$on('change', (e) => {
		values.push(e.detail.value);
		handle.$set({ segment: { ...segment, content: e.detail.value } });
	});
	flushSync();
	return { target, values, editor: target.querySelector<HTMLElement>('.inline-chip-editor')! };
}

async function settle() {
	flushSync();
	await tick();
	await tick();
}

async function insertFromMenu(target: HTMLElement) {
	const trigger = Array.from(target.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
		(b.getAttribute('aria-label') ?? '').startsWith('More actions for')
	)!;
	(document.activeElement as HTMLElement | null)?.blur();
	trigger.click();
	await settle();
	const item = Array.from(document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')).find(
		(b) => b.textContent?.trim().endsWith('Insert a condition…')
	)!;
	item.click();
	await settle();
}

async function saveModalWithText(text: string) {
	const dialog = conditionModal()!;
	const area = dialog.querySelector<HTMLTextAreaElement>('textarea[aria-label="If text"]')!;
	area.value = text;
	area.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
	Array.from(dialog.querySelectorAll('button')).find((b) => b.textContent?.trim().startsWith('Save'))!.click();
	await settle();
}

const chipThens = (editor: HTMLElement) =>
	Array.from(editor.querySelectorAll('[data-testid="condition-chip-summary"]')).map((el) => el.textContent?.trim());

describe('inserting a condition from the segment menu', () => {
	it('lands right after a variable chip in the middle of the text', async () => {
		const { target, values, editor } = mountSegment('portrait of ${x}, soft light');
		await settle();
		setCaret(editor.querySelector('.variable-usage-container')!.nextSibling!, 0);
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`portrait of \${x}${DEFAULT}, soft light`);
	});

	it('lands inside the text that follows a chip', async () => {
		const { target, values, editor } = mountSegment('portrait of ${x}, soft light');
		await settle();
		setCaret(editor.querySelector('.variable-usage-container')!.nextSibling!, ', soft'.length);
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`portrait of \${x}, soft${DEFAULT} light`);
	});

	it('lands after a variable chip whose popover was just closed', async () => {
		const { target, values, editor } = mountSegment('portrait of ${x}, soft light');
		await settle();
		const main = editor.querySelector<HTMLElement>('.variable-usage-container .chip-main')!;
		main.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
		await settle();
		main.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
		await settle();
		document.dispatchEvent(new Event('selectionchange'));
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`portrait of \${x}${DEFAULT}, soft light`);
	});

	it('lands after a variable chip when the caret was left inside its container', async () => {
		const { target, values, editor } = mountSegment('portrait of ${x}, soft light');
		await settle();
		const container = editor.querySelector('.variable-usage-container')!;
		setCaret(container, container.childNodes.length);
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`portrait of \${x}${DEFAULT}, soft light`);
	});

	it('uses the last caret in the text when the selection moved out of the editor', async () => {
		const { target, values, editor } = mountSegment('portrait of ${x}, soft light');
		await settle();
		setCaret(editor.querySelector('.variable-usage-container')!.nextSibling!, 0);
		const outside = document.createElement('input');
		document.body.appendChild(outside);
		outside.focus();
		setCaret(document.body, 0);
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`portrait of \${x}${DEFAULT}, soft light`);
	});
});

describe('a second condition on the same variable', () => {
	it('opens and saves the new one in place when an identical condition comes first, and keeps the order on reload', async () => {
		const { target, values, editor } = mountSegment(`${DEFAULT} portrait of \${x}, soft light`);
		await settle();
		const tail = editor.lastChild!;
		setCaret(tail, (tail.textContent ?? '').length);
		await insertFromMenu(target);
		expect(values.at(-1)).toBe(`${DEFAULT} portrait of \${x}, soft light${DEFAULT}`);
		expect(conditionModal()).not.toBeNull();
		await saveModalWithText('storm');
		const saved = values.at(-1)!;
		expect(saved).toBe(`${DEFAULT} portrait of \${x}, soft light<<if $x = "calm">>storm<<end>>`);
		expect(chipThens(editor)).toEqual(['→ text', '→ storm']);

		const reloaded = mountSegment(saved);
		await settle();
		expect(chipThens(reloaded.editor)).toEqual(['→ text', '→ storm']);
	});

	it('keeps a new condition after an edited one on the same variable', async () => {
		const first = '<<if $x = "tense">>storm<<else>>calm sea<<end>>';
		const { target, values, editor } = mountSegment(`${first} portrait of \${x}, soft light`);
		await settle();
		setCaret(editor.querySelector('.variable-usage-container')!.nextSibling!, 0);
		await insertFromMenu(target);
		await saveModalWithText('soft glow');
		const saved = values.at(-1)!;
		expect(saved).toBe(`${first} portrait of \${x}<<if $x = "calm">>soft glow<<end>>, soft light`);

		const reloaded = mountSegment(saved);
		await settle();
		expect(chipThens(reloaded.editor)).toEqual(['→ storm · else calm sea', '→ soft glow']);
	});
});
