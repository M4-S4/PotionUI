// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import type { VariableRoll, VariablesMap } from '$lib/utils/variableDefs';

const { default: ConditionChip } = await import('$lib/components/ConditionChip.svelte');

const RAW = '<<if $age > 65>>grey hair<<else>>auburn hair<<end>>';

const variables: VariablesMap = {
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 },
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'serene', 'tense'] },
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

function setup(raw = RAW, vars: VariablesMap = variables, rolls: Record<string, VariableRoll> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const context = writable({ variables: vars, rolls });
	const changes: string[] = [];
	const removed: number[] = [];
	instance = mount(ConditionChip, {
		target,
		props: {
			raw,
			context,
			variant: 'segment-composer',
			onchange: (next: string) => changes.push(next),
			onremove: () => removed.push(1),
			onCreateVariable: (name: string) => changes.push(`create:${name}`)
		}
	});
	flushSync();
	return { context, changes, removed };
}

const popover = () => document.querySelector<HTMLElement>('[role="dialog"][aria-label="Edit condition"]');

async function openPopover() {
	target!.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	return popover()!;
}

async function pick(select: HTMLSelectElement, value: string) {
	select.value = value;
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await tick();
}

async function typeInto(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
	el.value = value;
	el.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

const byText = (root: ParentNode, selector: string, text: string) =>
	Array.from(root.querySelectorAll<HTMLElement>(selector)).find((el) => el.textContent?.trim() === text)!;

describe('ConditionChip chip', () => {
	it('summarizes the test, the text and the otherwise', () => {
		setup();
		expect(target!.querySelector('.chip-context')!.textContent).toBe('age > 65');
		expect(target!.querySelector('[data-testid="condition-chip-summary"]')!.textContent).toContain(
			'grey hair · else auburn hair'
		);
		expect(target!.querySelector('.condition-chip.broken')).toBeNull();
	});

	it('counts hidden else-if branches', () => {
		setup('<<if $age > 65>>a<<else if $age < 25>>b<<else if $age = 40>>c<<else>>d<<end>>');
		expect(target!.querySelector('.chip-state')!.textContent).toBe('+2');
	});

	it('turns amber when the variable is gone and offers to create it', async () => {
		const { changes } = setup('<<if $ghost > 1>>a<<end>>');
		expect(target!.querySelector('.condition-chip.broken')).not.toBeNull();
		const pop = await openPopover();
		expect(pop.querySelector('[data-testid="condition-issue"]')!.textContent).toContain("$ghost isn't a variable");
		byText(pop, 'button', 'Create $ghost').click();
		expect(changes).toEqual(['create:ghost']);
	});

	it('shows a malformed block as raw text and stays closed', async () => {
		setup('<<if $age >>oops<<end>>');
		expect(target!.querySelector('.condition-chip.broken')).not.toBeNull();
		await openPopover();
		expect(popover()).toBeNull();
	});

	it('titles the popover in plain language, not grammar', async () => {
		setup();
		const pop = await openPopover();
		expect(pop.querySelector('.popover-title span')!.textContent).toBe('When age is more than 65 +1 more');
		expect(pop.querySelector('.popover-head')!.textContent).not.toContain('<<');
	});

	it('picks up a changed variable through the shared context', async () => {
		const { context } = setup();
		expect(target!.querySelector('.condition-chip.broken')).toBeNull();
		context.set({ variables: { mood: variables.mood }, rolls: {} });
		await tick();
		expect(target!.querySelector('.condition-chip.broken')).not.toBeNull();
	});
});

describe('ConditionChip popover edits', () => {
	it('changes the operator and writes the grammar', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		await pick(pop.querySelector('select[aria-label="Operator"]')!, '<=');
		expect(changes.at(-1)).toBe('<<if $age <= 65>>grey hair<<else>>auburn hair<<end>>');
	});

	it('steps a number operand', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		pop.querySelector<HTMLElement>('button[aria-label="Increase Value"]')!.click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 66>>grey hair<<else>>auburn hair<<end>>');
	});

	it('writes a between test as a range', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		await pick(pop.querySelector('select[aria-label="Operator"]')!, 'in');
		expect(changes.at(-1)).toBe('<<if $age in 65..65>>grey hair<<else>>auburn hair<<end>>');
	});

	it('moves to another variable with that type\'s default test', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		await pick(pop.querySelector('select[aria-label="Variable"]')!, 'mood');
		expect(changes.at(-1)).toBe('<<if $mood = "calm">>grey hair<<else>>auburn hair<<end>>');
		expect(pop.querySelectorAll('[aria-label="Values"] button').length).toBe(3);
	});

	it('toggles choice values for is one of', async () => {
		const { changes } = setup('<<if $mood in "calm">>a<<end>>');
		const pop = await openPopover();
		byText(pop, '[aria-label="Values"] button', 'serene').click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $mood in "calm", "serene">>a<<end>>');
		byText(pop, '[aria-label="Values"] button', 'calm').click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $mood in "serene">>a<<end>>');
	});

	it('edits the branch text', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'white hair, kind eyes');
		expect(changes.at(-1)).toBe('<<if $age > 65>>white hair, kind eyes<<else>>auburn hair<<end>>');
	});

	it('adds a test with a joiner and removes it again', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		byText(pop, 'button', 'Add test').click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 65 and $age >= 18>>grey hair<<else>>auburn hair<<end>>');
		byText(pop, '[role="radio"]', 'any').click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 65 or $age >= 18>>grey hair<<else>>auburn hair<<end>>');
		pop.querySelectorAll<HTMLElement>('button[aria-label="Remove test"]')[1].click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 65>>grey hair<<else>>auburn hair<<end>>');
	});

	it('adds else-if before the otherwise and removes the otherwise', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		byText(pop, 'button', 'Else if').click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 65>>grey hair<<else if $age >= 18>><<else>>auburn hair<<end>>');
		pop.querySelector<HTMLElement>('button[aria-label="Remove otherwise"]')!.click();
		await tick();
		expect(changes.at(-1)).toBe('<<if $age > 65>>grey hair<<else if $age >= 18>><<end>>');
		expect(byText(pop, 'button', 'Otherwise')).toBeTruthy();
	});

	it('does not write an incomplete edit and says so', async () => {
		const { changes } = setup();
		const pop = await openPopover();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'a <<if $x > 1>>b<<end>>');
		expect(changes).toEqual([]);
		expect(pop.querySelector('[data-testid="condition-invalid"]')).not.toBeNull();
	});

	it('removes the whole condition', async () => {
		const { removed } = setup();
		const pop = await openPopover();
		byText(pop, 'button', 'Remove condition').click();
		expect(removed).toEqual([1]);
	});

	it('reports the last run from a roll, or waits', async () => {
		const mood = '<<if $mood = "calm">>x<<else>>y<<end>>';
		setup(mood, variables, { mood: { optionIndex: 0, value: 'calm', rolledAt: 1 } });
		let pop = await openPopover();
		expect(pop.querySelector('[data-testid="condition-last-run"]')!.textContent).toMatch(/\$mood = calm\s*→ x/);
		unmount(instance!);
		document.body.innerHTML = '';
		setup(mood);
		pop = await openPopover();
		expect(pop.querySelector('[data-testid="condition-last-run"]')!.textContent).toContain('Decided when you generate');
	});

	it('closes with Done', async () => {
		setup();
		const pop = await openPopover();
		byText(pop, 'button', 'Done').click();
		await tick();
		expect(popover()).toBeNull();
	});
});
