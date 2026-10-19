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

const modal = () => document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="condition-modal-title"]');
const popover = modal;

async function openModal() {
	target!.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	await tick();
	return modal()!;
}

async function pick(select: HTMLSelectElement, value: string) {
	select.value = value;
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await tick();
}

const footerButton = (label: string) =>
	Array.from(modal()!.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith(label))!;

async function save() {
	footerButton('Save').click();
	await tick();
}

const picker = () => document.querySelector<HTMLElement>('[data-testid="variable-picker"]');
const pickerInput = () => picker()!.querySelector<HTMLInputElement>('input[role="combobox"]')!;
const pickerOptions = () =>
	Array.from(picker()!.querySelectorAll<HTMLElement>('[role="option"][data-variable]')).map((el) => el.dataset.variable);

async function openVariablePicker() {
	popover()!.querySelector<HTMLElement>('[data-testid="variable-select"]')!.click();
	await tick();
	await tick();
}

async function search(value: string) {
	const input = pickerInput();
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

async function keydown(key: string) {
	pickerInput().dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
	await tick();
}

async function choose(name: string) {
	await openVariablePicker();
	document.querySelector<HTMLElement>(`[role="option"][data-variable="${name}"]`)!.click();
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
		const pop = await openModal();
		expect(pop.querySelector('[data-testid="condition-issue"]')!.textContent).toContain("$ghost isn't a variable");
		byText(pop, 'button', 'Create $ghost').click();
		expect(changes).toEqual(['create:ghost']);
	});

	it('shows a malformed block as raw text and stays closed', async () => {
		setup('<<if $age >>oops<<end>>');
		expect(target!.querySelector('.condition-chip.broken')).not.toBeNull();
		await openModal();
		expect(modal()).toBeNull();
	});

	it('titles the modal in plain language, not grammar', async () => {
		setup();
		const pop = await openModal();
		expect(pop.querySelector('[data-testid="condition-summary"]')!.textContent).toBe('When age is more than 65 +1 more');
		expect(pop.textContent).not.toContain('<<');
	});

	it('picks up a changed variable through the shared context', async () => {
		const { context } = setup();
		expect(target!.querySelector('.condition-chip.broken')).toBeNull();
		context.set({ variables: { mood: variables.mood }, rolls: {} });
		await tick();
		expect(target!.querySelector('.condition-chip.broken')).not.toBeNull();
	});
});

describe('ConditionChip editor modal', () => {
	it('opens a modal straight from the chip with a plain-language header', async () => {
		setup();
		const pop = await openModal();
		expect(pop.querySelector('#condition-modal-title')!.textContent).toBe('Condition');
		expect(document.querySelector('.fixed.z-overlay.rounded-lg[role="dialog"]')).toBeNull();
	});

	it('applies edits only on Save', async () => {
		const { changes } = setup();
		const pop = await openModal();
		await pick(pop.querySelector('select[aria-label="Operator"]')!, '<=');
		expect(changes).toEqual([]);
		await save();
		expect(changes).toEqual(['<<if $age <= 65>>grey hair<<else>>auburn hair<<end>>']);
		expect(modal()).toBeNull();
	});

	it('discards edits on Cancel', async () => {
		const { changes } = setup();
		const pop = await openModal();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'white hair');
		footerButton('Cancel').click();
		await tick();
		expect(changes).toEqual([]);
		expect(modal()).toBeNull();
		const again = await openModal();
		expect(again.querySelector<HTMLTextAreaElement>('textarea[aria-label="If text"]')!.value).toBe('grey hair');
	});

	it('cancels on Escape and saves on Ctrl+Enter', async () => {
		const { changes } = setup();
		await openModal();
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await tick();
		expect(modal()).toBeNull();
		const pop = await openModal();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'x');
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
		await tick();
		expect(changes).toEqual(['<<if $age > 65>>x<<else>>auburn hair<<end>>']);
	});

	it('steps a number operand', async () => {
		const { changes } = setup();
		const pop = await openModal();
		pop.querySelector<HTMLElement>('button[aria-label="Increase Value"]')!.click();
		await tick();
		await save();
		expect(changes.at(-1)).toBe('<<if $age > 66>>grey hair<<else>>auburn hair<<end>>');
	});

	it('moves to another variable with that type\'s default test', async () => {
		const { changes } = setup();
		const pop = await openModal();
		await choose('mood');
		expect(pop.querySelectorAll('[aria-label="Values"] button').length).toBe(3);
		await save();
		expect(changes.at(-1)).toBe('<<if $mood = "calm">>grey hair<<else>>auburn hair<<end>>');
	});

	it('searches variables by value text in the picker and picks the match', async () => {
		const { changes } = setup();
		await openModal();
		await openVariablePicker();
		await search('serene');
		expect(pickerOptions()).toEqual(['mood']);
		await keydown('Enter');
		expect(picker()).toBeNull();
		expect(modal()).not.toBeNull();
		await save();
		expect(changes.at(-1)).toBe('<<if $mood = "calm">>grey hair<<else>>auburn hair<<end>>');
	});

	it('closes only the picker on Escape, keeping the modal open', async () => {
		setup();
		await openModal();
		await openVariablePicker();
		await keydown('Escape');
		expect(picker()).toBeNull();
		expect(modal()).not.toBeNull();
	});

	it('creates a variable from the search text and retargets the test once it exists', async () => {
		const { changes, context } = setup();
		await openModal();
		await openVariablePicker();
		await search('hat');
		document.querySelector<HTMLElement>('[data-create="hat"]')!.click();
		await tick();
		expect(changes).toContain('create:hat');
		context.set({ variables: { ...variables, hat: { type: 'text', value: '' } }, rolls: {} });
		await tick();
		await tick();
		await save();
		expect(changes.at(-1)).toContain('$hat');
	});

	it('toggles choice values for is one of', async () => {
		const { changes } = setup('<<if $mood in "calm">>a<<end>>');
		const pop = await openModal();
		byText(pop, '[aria-label="Values"] button', 'serene').click();
		await tick();
		await save();
		expect(changes.at(-1)).toBe('<<if $mood in "calm", "serene">>a<<end>>');
	});

	it('edits the branch text', async () => {
		const { changes } = setup();
		const pop = await openModal();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'white hair, kind eyes');
		await save();
		expect(changes.at(-1)).toBe('<<if $age > 65>>white hair, kind eyes<<else>>auburn hair<<end>>');
	});

	it('adds a test with a joiner and removes it again', async () => {
		const { changes } = setup();
		const pop = await openModal();
		byText(pop, 'button', 'Add test').click();
		await tick();
		byText(pop, '[role="radio"]', 'any').click();
		await tick();
		expect(pop.querySelectorAll('[data-testid="condition-test"]').length).toBe(2);
		pop.querySelectorAll<HTMLElement>('button[aria-label="Remove test"]')[1].click();
		await tick();
		expect(pop.querySelectorAll('[data-testid="condition-test"]').length).toBe(1);
		await save();
		expect(changes).toEqual([]);
	});

	it('saves a two-test condition with its joiner', async () => {
		const { changes } = setup();
		const pop = await openModal();
		byText(pop, 'button', 'Add test').click();
		await tick();
		byText(pop, '[role="radio"]', 'any').click();
		await tick();
		await save();
		expect(changes.at(-1)).toBe('<<if $age > 65 or $age >= 18>>grey hair<<else>>auburn hair<<end>>');
	});

	it('adds else-if before the otherwise and removes the otherwise', async () => {
		const { changes } = setup();
		const pop = await openModal();
		byText(pop, 'button', 'Else if').click();
		await tick();
		pop.querySelector<HTMLElement>('button[aria-label="Remove otherwise"]')!.click();
		await tick();
		expect(byText(pop, 'button', 'Otherwise')).toBeTruthy();
		await save();
		expect(changes.at(-1)).toBe('<<if $age > 65>>grey hair<<else if $age >= 18>><<end>>');
	});

	it('does not save an incomplete edit and says so', async () => {
		const { changes } = setup();
		const pop = await openModal();
		await typeInto(pop.querySelector('textarea[aria-label="If text"]')!, 'a <<if $x > 1>>b<<end>>');
		expect(pop.querySelector('[data-testid="condition-invalid"]')).not.toBeNull();
		expect(footerButton('Save').disabled).toBe(true);
		expect(changes).toEqual([]);
	});

	it('removes the whole condition', async () => {
		const { removed } = setup();
		const pop = await openModal();
		byText(pop, 'button', 'Remove condition').click();
		expect(removed).toEqual([1]);
	});

	it('reports the last run from a roll, or waits', async () => {
		const mood = '<<if $mood = "calm">>x<<else>>y<<end>>';
		setup(mood, variables, { mood: { optionIndex: 0, value: 'calm', rolledAt: 1 } });
		let pop = await openModal();
		expect(pop.querySelector('[data-testid="condition-last-run"]')!.textContent).toMatch(/\$mood = calm\s*→ x/);
		unmount(instance!);
		document.body.innerHTML = '';
		setup(mood);
		pop = await openModal();
		expect(pop.querySelector('[data-testid="condition-last-run"]')!.textContent).toContain('Decided when you generate');
	});

	it('puts every value editor on its own line and wraps long values inside the body', async () => {
		const long = {
			look: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [
					'a fitted phase coat with emitters',
					'a long hooded cloak woven from silver thread and moonlight',
					'plain linen shirt',
					'ceremonial armour plated in layered bronze scales'
				]
			}
		} as VariablesMap;
		setup('<<if $look in "plain linen shirt">>x<<end>>', long);
		const pop = await openModal();
		const row = pop.querySelector('[data-testid="condition-test"]')!;
		const controls = row.children[0] as HTMLElement;
		const value = row.querySelector('[data-testid="condition-value"]') as HTMLElement;
		expect(controls.contains(value)).toBe(false);
		expect(controls.compareDocumentPosition(value) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		const buttons = value.querySelectorAll('[aria-label="Values"] button');
		expect(buttons.length).toBe(4);
		buttons.forEach((b) => {
			expect(b.className).toContain('max-w-full');
			expect(b.className).toContain('whitespace-normal');
		});
		expect(value.querySelector('[aria-label="Values"]')!.className).toContain('flex-wrap');
	});

	it('returns focus to the editor after the chip when the modal closes', async () => {
		const host = document.createElement('div');
		host.setAttribute('contenteditable', 'true');
		host.setAttribute('tabindex', '-1');
		document.body.appendChild(host);
		target = document.createElement('div');
		host.appendChild(target);
		const context = writable({ variables, rolls: {} });
		instance = mount(ConditionChip, { target, props: { raw: RAW, context, variant: 'segment-composer' } });
		flushSync();
		await openModal();
		footerButton('Cancel').click();
		await tick();
		await tick();
		expect(document.activeElement).toBe(host);
	});

	it('ignores an outside pointerdown while the modal is open', async () => {
		setup();
		await openModal();
		document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
		await tick();
		expect(modal()).not.toBeNull();
	});
});
