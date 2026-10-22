// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { conditionModal } from './helpers/typing';

const { default: ConditionChip } = await import('$lib/components/ConditionChip.svelte');

const choice = (...options: string[]) => ({ type: 'choice', mode: 'shuffle', pinnedIndex: null, options }) as const;
const number = { type: 'number', mode: 'per-image', min: 1, max: 9, step: 1, value: 1 } as const;

const PAIRS: [string, string, string][] = [
	['mood', 'mood2', 'mood2'],
	['mood', 'moody', 'mood'],
	['outfit', 'outfit_kind', 'outfit'],
	['Mood', 'mood', 'Mood'],
	['age', 'age2', 'age2'],
	['a', 'ab', 'a']
];

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLDivElement | undefined;
let errors: unknown[] = [];

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	target?.remove();
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

function variablesFor(a: string, b: string): VariablesMap {
	return { [a]: choice('calm', 'serene'), [b]: choice('calm', 'tense'), n: number };
}

async function openFor(raw: string, variables: VariablesMap) {
	errors = [];
	vi.spyOn(console, 'error').mockImplementation((...args) => {
		errors.push(args);
	});
	const onError = (e: ErrorEvent) => errors.push(e.error ?? e.message);
	window.addEventListener('error', onError);
	target = document.createElement('div');
	document.body.appendChild(target);
	const context = writable({ variables, rolls: {} });
	instance = mount(ConditionChip, { target, props: { raw, context, variant: 'segment-composer' } });
	flushSync();
	target.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	await tick();
	await tick();
	window.removeEventListener('error', onError);
	return conditionModal();
}

describe('condition editor with similar variable names', () => {
	for (const [a, b, read] of PAIRS) {
		it(`opens with ${a} and ${b} reading ${read}`, async () => {
			const raw = `<<if $${read} = "calm">>x<<else>>y<<end>>`;
			const modal = await openFor(raw, variablesFor(a, b));
			expect(errors).toEqual([]);
			expect(modal).not.toBeNull();
			expect(modal!.querySelector('[data-testid="variable-select"]')!.textContent).toContain(`$${read}`);
		});

		it(`opens the picker with ${a} and ${b} and selects the exact one`, async () => {
			const raw = `<<if $${read} = "calm">>x<<end>>`;
			const modal = await openFor(raw, variablesFor(a, b));
			modal!.querySelector<HTMLElement>('[data-testid="variable-select"]')!.click();
			await tick();
			await tick();
			const names = Array.from(document.querySelectorAll<HTMLElement>('[role="option"][data-variable]')).map((o) => o.dataset.variable);
			expect(names).toEqual(expect.arrayContaining([a, b, 'n']));
			expect(new Set(names).size).toBe(names.length);
			const other = read === a ? b : a;
			Array.from(document.querySelectorAll<HTMLElement>('[role="option"][data-variable]')).find((o) => o.dataset.variable === other)!.click();
			await tick();
			expect(errors).toEqual([]);
			expect(modal!.querySelector('[data-testid="variable-select"]')!.textContent).toContain(`$${other}`);
		});
	}

	it('survives a condition reading both similar variables across tests and branches', async () => {
		const raw = '<<if $mood = "calm" and $mood2 = "calm">>a<<else if $mood2 = "tense">>b<<else if $mood = "serene">>c<<end>>';
		const modal = await openFor(raw, variablesFor('mood', 'mood2'));
		expect(errors).toEqual([]);
		expect(modal!.querySelectorAll('[data-testid="condition-test"]').length).toBe(4);
		expect(modal!.querySelector('[data-testid="condition-last-run"]')).not.toBeNull();
	});

	it('survives duplicate option text inside one variable', async () => {
		const modal = await openFor('<<if $mood in "calm">>x<<end>>', { mood: choice('calm', 'calm', 'tense'), mood2: choice('calm') });
		expect(errors).toEqual([]);
		expect(modal).not.toBeNull();
	});

	it('keeps a variable named create apart from the Create row', async () => {
		const modal = await openFor('<<if $create = "calm">>x<<end>>', { create: choice('calm'), mood: choice('calm') });
		modal!.querySelector<HTMLElement>('[data-testid="variable-select"]')!.click();
		await tick();
		await tick();
		const ids = Array.from(document.querySelectorAll('[role="option"]')).map((o) => o.id);
		expect(new Set(ids).size).toBe(ids.length);
		expect(errors).toEqual([]);
	});

	it('opens the option-condition chip when the referenced variable repeats an option', async () => {
		const { default: VariableConditionChip } = await import('$lib/components/VariableConditionChip.svelte');
		errors = [];
		vi.spyOn(console, 'error').mockImplementation((...args) => {
			errors.push(args);
		});
		const onError = (e: ErrorEvent) => errors.push(e.error ?? e.message);
		window.addEventListener('error', onError);
		target = document.createElement('div');
		document.body.appendChild(target);
		instance = mount(VariableConditionChip, {
			target,
			props: {
				ownerName: 'look',
				when: { var: 'mood', values: [] },
				mode: 'shuffle',
				variables: { mood: choice('calm', 'calm', 'tense'), mood2: choice('x'), look: choice('a', 'b') },
				onChange: () => {}
			}
		});
		flushSync();
		target.querySelector('button')!.click();
		await tick();
		await tick();
		window.removeEventListener('error', onError);
		expect(errors).toEqual([]);
		expect(document.querySelectorAll('[aria-label^="Toggle"]').length).toBe(2);
	});
});

describe('Variables window name collisions', () => {
	async function mountManager(variables: VariablesMap) {
		const { default: VariableManagerModal } = await import('$lib/components/VariableManagerModal.svelte');
		const { createClassComponent } = await import('svelte/legacy');
		const host = document.createElement('div');
		document.body.appendChild(host);
		const changes: VariablesMap[] = [];
		const component = createClassComponent({
			component: VariableManagerModal as never,
			target: host,
			props: { isOpen: true, variables }
		});
		component.$on('change', (e: CustomEvent<VariablesMap>) => changes.push(e.detail));
		await tick();
		return { component, changes };
	}

	async function renameSecondTo(name: string) {
		const rows = Array.from(document.querySelectorAll<HTMLElement>('[role="option"]'));
		rows[1].click();
		await tick();
		const input = document.querySelector<HTMLInputElement>('input[aria-label="Variable name"]')!;
		input.value = name;
		input.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		await tick();
	}

	it.each(['mood', 'Mood', 'mood '])('blocks the name %j with an inline error and keeps the first variable', async (name) => {
		const first = choice('calm', 'serene');
		const { component, changes } = await mountManager({ mood: first, other: choice('x') });
		await renameSecondTo(name);
		expect(document.body.textContent).toContain('Another variable already uses this name');
		const last = changes.at(-1);
		if (last) {
			expect(Object.keys(last)).toEqual(['mood']);
			expect(last.mood).toEqual(first);
		}
		component.$destroy();
	});

	it('keeps a legacy case-twin stored when another row is edited, and deletes it only when its row is removed', async () => {
		const mood = choice('calm', 'serene');
		const twin = choice('x', 'y');
		const { component, changes } = await mountManager({ Mood: mood, mood: twin, tone: choice('a', 'b') });
		const rows = Array.from(document.querySelectorAll<HTMLElement>('[role="option"]'));
		rows[2].click();
		await tick();
		const input = document.querySelector<HTMLInputElement>('input[aria-label="Variable name"]')!;
		input.value = 'tone2';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		await tick();
		const afterEdit = changes.at(-1)!;
		expect(Object.keys(afterEdit).sort()).toEqual(['Mood', 'mood', 'tone2']);
		expect(afterEdit.mood).toEqual(twin);

		const rowsAgain = Array.from(document.querySelectorAll<HTMLElement>('[role="option"]'));
		rowsAgain[1].click();
		await tick();
		const remove = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
			b.getAttribute('aria-label')?.startsWith('Remove variable mood')
		)!;
		remove.click();
		await tick();
		await tick();
		expect(Object.keys(changes.at(-1)!).sort()).toEqual(['Mood', 'tone2']);
		component.$destroy();
	});
});
