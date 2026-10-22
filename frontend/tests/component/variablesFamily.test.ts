// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableManagerModal } = await import('$lib/components/VariableManagerModal.svelte');
const { default: PasteVariablesModal } = await import('$lib/components/PasteVariablesModal.svelte');
const { default: VariableConditionChip } = await import('$lib/components/VariableConditionChip.svelte');
const { createClassComponent } = await import('svelte/legacy');

const variables: VariablesMap = {
	outfit: {
		type: 'choice',
		mode: 'shuffle',
		pinnedIndex: null,
		options: ['a tailored charcoal three-piece suit with a crisp white shirt', 'an oversized cream cable-knit sweater']
	},
	time: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['dawn', 'night'] }
};

let destroy: (() => void) | undefined;

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

function legacy(Component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: Component as never, target, props }) as unknown as {
		$destroy: () => void;
		$on: (name: string, fn: (event: CustomEvent) => void) => void;
	};
	destroy = () => component.$destroy();
	flushSync();
	return component;
}

const buttonStarting = (root: ParentNode, label: string) =>
	Array.from(root.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith(label))!;

async function escape() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
	await tick();
}

describe('Prompt variables', () => {
	it('opens at the L width and puts each option’s When under its value', () => {
		legacy(VariableManagerModal, { isOpen: true, variables });
		const dialog = document.querySelector<HTMLElement>('[role="dialog"]')!;
		expect(dialog.className).toContain('md:w-[1040px]');
		const optionRows = Array.from(document.querySelectorAll<HTMLElement>('[data-testid="variable-option-row"]'));
		expect(optionRows).toHaveLength(2);
		for (const row of optionRows) {
			const value = row.querySelector('textarea')!;
			const when = row.querySelector<HTMLElement>('[data-testid="variable-option-when"]')!;
			expect(value.compareDocumentPosition(when) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
			expect(when.className).toContain('col-start-2');
			expect(when.textContent).toContain('When');
			expect(when.contains(value)).toBe(false);
		}
	});

	it('asks before Esc throws edits away, and closes at once without edits', async () => {
		const component = legacy(VariableManagerModal, { isOpen: true, variables });
		const closes: number[] = [];
		component.$on('close', () => closes.push(1));
		await escape();
		expect(closes).toEqual([1]);
		const textarea = document.querySelector<HTMLTextAreaElement>('textarea')!;
		textarea.value = 'a navy suit';
		textarea.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		await escape();
		expect(closes).toEqual([1]);
		expect(document.querySelector('[role="alertdialog"]')!.textContent).toContain('Your edits to the variables are undone.');
	});
});

describe('Paste variables', () => {
	it('opens at the M width and imports on Ctrl+Enter', async () => {
		const component = legacy(PasteVariablesModal, { isOpen: true, existing: {} });
		const imported: unknown[] = [];
		component.$on('confirm', (event) => imported.push(event.detail));
		expect(document.querySelector<HTMLElement>('[role="dialog"]')!.className).toContain('md:w-[600px]');
		expect(buttonStarting(document, 'Import').textContent?.replace(/\s+/g, '')).toBe('ImportCtrl+Enter');
		const json = document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Variables JSON"]')!;
		json.value = '{ "mood": { "type": "text", "value": "calm" } }';
		json.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		json.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
		await tick();
		expect(imported).toEqual([]);
		json.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
		await tick();
		expect(imported).toHaveLength(1);
	});
});

describe('Only when', () => {
	function open(width: number) {
		vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(width);
		const changes: unknown[] = [];
		const instance = mount(VariableConditionChip, {
			target: document.body,
			props: {
				ownerName: 'outfit',
				optionNumber: 2,
				when: { var: 'time', values: ['night'] },
				mode: 'shuffle',
				variables,
				onChange: (when: unknown) => changes.push(when)
			}
		});
		destroy = () => unmount(instance);
		flushSync();
		document.querySelector<HTMLButtonElement>('button[aria-label="Edit condition for this option of outfit"]')!.click();
		flushSync();
		return changes;
	}

	const panel = () => document.querySelector<HTMLElement>('[role="dialog"][aria-label="Condition for outfit"]');

	it('shows the family header and closes from Done', async () => {
		open(1440);
		await tick();
		expect(panel()!.textContent).toContain('Only when');
		expect(panel()!.textContent?.replace(/\s+/g, ' ')).toContain('Option 2 of $outfit');
		buttonStarting(panel()!, 'Done').click();
		await tick();
		expect(panel()).toBeNull();
	});

	it('sets the option back to Always', async () => {
		const changes = open(1440);
		await tick();
		buttonStarting(panel()!, 'Set to Always').click();
		await tick();
		expect(changes).toEqual([undefined]);
		expect(panel()).toBeNull();
	});

	it('becomes a bottom sheet with gutters on a phone instead of a fixed 420px box', async () => {
		open(390);
		await tick();
		const style = panel()!.getAttribute('style')!;
		expect(style).toContain('left: 8px');
		expect(style).toContain('right: 8px');
		expect(style).not.toContain('width: 420px');
	});
});
