// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import type { VariablesMap } from '$lib/utils/variableDefs';

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
	return document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="condition-modal-title"]');
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
