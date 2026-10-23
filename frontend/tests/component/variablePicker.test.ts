// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariablePicker } = await import('$lib/components/VariablePicker.svelte');

const variables: VariablesMap = {
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'serene', 'tense'] },
	hair: { type: 'choice', mode: 'pin', pinnedIndex: 0, options: ['auburn', 'grey'] },
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 },
	name: { type: 'text', value: 'Mara' }
};

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLDivElement | undefined;

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	target?.remove();
	document.body.innerHTML = '';
});

function setup(props: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const onselect = vi.fn();
	const oncreate = vi.fn();
	const onclose = vi.fn();
	instance = mount(VariablePicker, {
		target,
		props: { variables, onselect, onclose, ...props, ...(props.noCreate ? {} : { oncreate }) }
	});
	flushSync();
	return { onselect, oncreate, onclose };
}

const input = () => target!.querySelector<HTMLInputElement>('input[role="combobox"]')!;
const options = () => Array.from(target!.querySelectorAll<HTMLElement>('[role="option"]'));
const names = () => options().filter((o) => o.dataset.variable).map((o) => o.dataset.variable);
const activeName = () => {
	const id = input().getAttribute('aria-activedescendant');
	return id ? document.getElementById(id)?.dataset.variable ?? document.getElementById(id)?.dataset.create : undefined;
};

async function type(value: string) {
	input().value = value;
	input().dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

async function key(k: string) {
	input().dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
	await tick();
}

describe('VariablePicker', () => {
	it('focuses the search input on open', async () => {
		setup();
		await tick();
		await tick();
		expect(document.activeElement).toBe(input());
	});

	it('exposes combobox and listbox roles with a label and active descendant', () => {
		setup({ label: 'Condition variable' });
		expect(input().getAttribute('aria-label')).toBe('Condition variable');
		expect(input().getAttribute('aria-controls')).toBe(target!.querySelector('[role="listbox"]')!.id);
		expect(activeName()).toBe('mood');
	});

	it('shows type badge, mode and preview per row', () => {
		setup();
		const mood = options().find((o) => o.dataset.variable === 'mood')!;
		expect(mood.textContent).toContain('$mood');
		expect(mood.textContent).toContain('Choice');
		expect(mood.textContent).toContain('shuffle');
		expect(mood.textContent).toContain('calm · serene · tense');
		const age = options().find((o) => o.dataset.variable === 'age')!;
		expect(age.textContent).toContain('Number');
		expect(age.textContent).toContain('per image');
		expect(age.textContent).toContain('18–80');
	});

	it('filters by name and value text, case-insensitively', async () => {
		setup();
		await type('MOO');
		expect(names()).toEqual(['mood']);
		await type('auburn');
		expect(names()).toEqual(['hair']);
	});

	it('highlights the matched text in the name and the value', async () => {
		setup();
		await type('ser');
		const marks = Array.from(target!.querySelectorAll('mark')).map((m) => m.textContent);
		expect(marks).toEqual(['ser']);
		await type('ma');
		expect(Array.from(target!.querySelectorAll('mark')).map((m) => m.textContent.toLowerCase())).toContain('ma');
	});

	it('moves with arrow keys, wraps, and selects with Enter', async () => {
		const { onselect } = setup();
		expect(activeName()).toBe('mood');
		await key('ArrowDown');
		expect(activeName()).toBe('hair');
		await key('ArrowUp');
		await key('ArrowUp');
		expect(activeName()).toBe('name');
		await key('Home');
		expect(activeName()).toBe('mood');
		await key('End');
		expect(activeName()).toBe('name');
		await key('Enter');
		expect(onselect).toHaveBeenCalledWith('name');
	});

	it('selects on click', async () => {
		const { onselect } = setup();
		options().find((o) => o.dataset.variable === 'age')!.click();
		expect(onselect).toHaveBeenCalledWith('age');
	});

	it('clears the query on the first Escape and closes on the second', async () => {
		const { onclose } = setup();
		await type('mo');
		await key('Escape');
		expect(input().value).toBe('');
		expect(onclose).not.toHaveBeenCalled();
		await key('Escape');
		expect(onclose).toHaveBeenCalledWith('escape');
	});

	it('does not prevent Tab so focus moves out, and reports it', async () => {
		const { onclose } = setup();
		const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
		input().dispatchEvent(event);
		expect(event.defaultPrevented).toBe(false);
		expect(onclose).toHaveBeenCalledWith('tab');
	});

	it('offers creating the typed name when it does not exist, and Enter creates it', async () => {
		const { oncreate, onselect } = setup();
		await type('hat');
		expect(names()).toEqual([]);
		expect(target!.querySelector('[data-create="hat"]')!.textContent).toContain('Create variable');
		expect(activeName()).toBe('hat');
		await key('Enter');
		expect(oncreate).toHaveBeenCalledWith('hat');
		expect(onselect).not.toHaveBeenCalled();
	});

	it('does not offer create for an existing name, an invalid name, or without a handler', async () => {
		setup();
		await type('mood');
		expect(target!.querySelector('[data-create]')).toBeNull();
		await type('not valid');
		expect(target!.querySelector('[data-create]')).toBeNull();
		unmount(instance!);
		target!.innerHTML = '';
		setup({ noCreate: true });
		await type('hat');
		expect(target!.querySelector('[data-create]')).toBeNull();
	});

	it('shows an empty state when nothing matches and creation is unavailable', async () => {
		setup({ noCreate: true });
		await type('zzz');
		expect(target!.querySelector('[data-testid="variable-picker-empty"]')!.textContent).toContain('No variables match');
	});

	it('shows a no-variables state for an empty map', () => {
		setup({ variables: {}, noCreate: true });
		expect(target!.querySelector('[data-testid="variable-picker-empty"]')!.textContent).toContain('No variables yet');
	});

	it('limits rows to the allowed types', () => {
		setup({ allowedTypes: ['choice'] });
		expect(names()).toEqual(['mood', 'hair']);
	});

	it('skips disabled rows for navigation and selection', async () => {
		const { onselect } = setup({ disabledReasons: { mood: 'this variable' } });
		expect(activeName()).toBe('hair');
		const mood = options().find((o) => o.dataset.variable === 'mood')!;
		expect(mood.getAttribute('aria-disabled')).toBe('true');
		mood.click();
		expect(onselect).not.toHaveBeenCalled();
	});

	it('starts on the selected variable', () => {
		setup({ selected: 'age' });
		expect(activeName()).toBe('age');
	});

	it('rows are at least 36px tall and text at least 12px by class', () => {
		setup();
		const row = options()[0];
		expect(row.className).toContain('min-h-9');
		expect(target!.innerHTML).not.toMatch(/text-2xs/);
	});
});
