// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import { MAX_STRING_CHARS } from '$lib/utils/promptConditions';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: ConditionChip } = await import('$lib/components/ConditionChip.svelte');

const AT_CAP = 'a long hooded cloak woven from silver thread and moonlight'.padEnd(MAX_STRING_CHARS - 1, ', fox fur') + 's';
const LONG = 'a cloak woven from silver thread and moonlight'.padEnd(MAX_STRING_CHARS, ', pearls') + 's';

const variables: VariablesMap = {
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 },
	cloak: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [LONG, `${LONG} and lace`] },
	look: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [LONG, 'linen shirt', AT_CAP, 'silk gown'] },
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

function setup(raw: string) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const context = writable({ variables, rolls: {} });
	const changes: string[] = [];
	instance = mount(ConditionChip, {
		target,
		props: { raw, context, variant: 'segment-composer', onchange: (next: string) => changes.push(next) }
	});
	flushSync();
	return { changes };
}

const modal = () => document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="condition-modal-title"]')!;
const branches = () => Array.from(modal().querySelectorAll<HTMLElement>('[data-testid="condition-branch"]'));
const problems = () =>
	Array.from(modal().querySelectorAll<HTMLElement>('[data-testid="condition-problem"]')).map((el) => el.textContent?.trim());
const byText = (root: ParentNode, selector: string, text: string) =>
	Array.from(root.querySelectorAll<HTMLElement>(selector)).find((el) => el.textContent?.trim() === text);
const saveButton = () =>
	Array.from(modal().querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith('Save'))!;

async function openModal() {
	target!.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	await tick();
	return modal();
}

async function typeInto(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
	el.value = value;
	el.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

async function pick(select: HTMLSelectElement, value: string) {
	select.value = value;
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await tick();
}

async function click(el: HTMLElement | undefined) {
	el!.click();
	await tick();
}

async function save() {
	saveButton().click();
	await tick();
}

async function ctrlEnter() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
	await tick();
}

const operator = (branch: HTMLElement) => branch.querySelector<HTMLSelectElement>('select[aria-label="Operator"]')!;
const textValue = (branch: HTMLElement) => branch.querySelector<HTMLInputElement>('input[aria-label="Text"]')!;
const body = (branch: HTMLElement) => branch.querySelector<HTMLTextAreaElement>('textarea')!;
const saveDisabled = () => saveButton().disabled;

describe('condition editor switch on one variable', () => {
	it('saves an if and an else if on a text variable as soon as both are filled in', async () => {
		const { changes } = setup('<<if $name is empty>>text<<end>>');
		const pop = await openModal();
		await pick(operator(branches()[0]), '=');
		await typeInto(textValue(branches()[0]), 'y');
		await typeInto(body(branches()[0]), 'z');
		await click(byText(pop, 'button', 'Else if'));
		await pick(operator(branches()[1]), '=');
		expect(saveDisabled()).toBe(true);
		await typeInto(textValue(branches()[1]), 'u');
		await typeInto(body(branches()[1]), 'o');
		expect(pop.querySelector('[data-testid="condition-invalid"]')).toBeNull();
		expect(saveDisabled()).toBe(false);
		await save();
		expect(changes).toEqual(['<<if $name = "y">>z<<else if $name = "u">>o<<end>>']);
	});

	it('starts a new else if on a choice option short enough to compare', async () => {
		const { changes } = setup('<<if $look = "linen shirt">>z<<end>>');
		const pop = await openModal();
		await click(byText(pop, 'button', 'Else if'));
		expect(saveDisabled()).toBe(false);
		expect(byText(branches()[1], '[aria-label="Values"] button', 'linen shirt')!.getAttribute('aria-pressed')).toBe('true');
		await click(byText(branches()[1], '[aria-label="Values"] button', 'silk gown'));
		await typeInto(body(branches()[1]), 'o');
		await save();
		expect(changes).toEqual(['<<if $look = "linen shirt">>z<<else if $look = "silk gown">>o<<end>>']);
	});

	it('saves an else if on an option as long as the value cap', async () => {
		const { changes } = setup('<<if $look = "linen shirt">>z<<end>>');
		const pop = await openModal();
		await click(byText(pop, 'button', 'Else if'));
		await click(byText(branches()[1], '[aria-label="Values"] button', AT_CAP));
		await typeInto(body(branches()[1]), 'o');
		expect(byText(branches()[1], '[aria-label="Values"] button', AT_CAP)!.getAttribute('aria-pressed')).toBe('true');
		expect(saveDisabled()).toBe(false);
		await save();
		expect(changes).toEqual([`<<if $look = "linen shirt">>z<<else if $look = "${AT_CAP}">>o<<end>>`]);
	});

	it('never shows an option over the value cap as picked', async () => {
		setup('<<if $look = "linen shirt">>z<<end>>');
		const pop = await openModal();
		await click(byText(pop, 'button', 'Else if'));
		const long = byText(branches()[1], '[aria-label="Values"] button', LONG)!;
		await click(long);
		expect(long.getAttribute('aria-disabled')).toBe('true');
		expect(long.getAttribute('aria-pressed')).toBe('false');
		expect(byText(branches()[1], '[aria-label="Values"] button', 'linen shirt')!.getAttribute('aria-pressed')).toBe('true');
		expect(saveDisabled()).toBe(false);
	});

	it('names an option over the value cap when no other option fits, and flags its row', async () => {
		setup('<<if $look = "linen shirt">>z<<else if $cloak is empty>>o<<end>>');
		await openModal();
		await pick(operator(branches()[1]), '=');
		expect(saveDisabled()).toBe(true);
		expect(problems()).toEqual([
			`Else if: the option picked for $cloak is longer than ${MAX_STRING_CHARS} characters and can't be used in a condition.`
		]);
		expect(branches()[1].querySelectorAll('[aria-label="Values"] button[aria-pressed="true"]')).toHaveLength(0);
		expect(branches()[1].dataset.invalid).toBe('true');
		expect(branches()[1].querySelector<HTMLElement>('[data-testid="condition-test"]')!.dataset.invalid).toBe('true');
		expect(branches()[0].dataset.invalid).toBeUndefined();
	});

	it('picks the first option when a choice test leaves is empty', async () => {
		const { changes } = setup('<<if $mood is empty>>z<<end>>');
		const pop = await openModal();
		await pick(operator(branches()[0]), '=');
		expect(saveDisabled()).toBe(false);
		expect(byText(pop, '[aria-label="Values"] button', 'calm')!.getAttribute('aria-pressed')).toBe('true');
		await save();
		expect(changes).toEqual(['<<if $mood = "calm">>z<<end>>']);
	});

	it('keeps one value when a choice test goes from is one of to is', async () => {
		const { changes } = setup('<<if $mood in "calm", "tense">>z<<end>>');
		const pop = await openModal();
		await pick(operator(branches()[0]), '=');
		const pressed = Array.from(pop.querySelectorAll('[aria-label="Values"] button[aria-pressed="true"]'));
		expect(pressed.map((b) => b.textContent?.trim())).toEqual(['calm']);
		await save();
		expect(changes).toEqual(['<<if $mood = "calm">>z<<end>>']);
	});

	it('uses a typed number before the field loses focus', async () => {
		const { changes } = setup('<<if $age = 30>>z<<end>>');
		const pop = await openModal();
		await click(byText(pop, 'button', 'Else if'));
		await pick(operator(branches()[1]), '=');
		await typeInto(body(branches()[1]), 'o');
		await typeInto(branches()[1].querySelector<HTMLInputElement>('input[aria-label="Value"]')!, '40');
		await ctrlEnter();
		expect(changes).toEqual(['<<if $age = 30>>z<<else if $age = 40>>o<<end>>']);
	});

	it('updates Save while a between range is typed', async () => {
		setup('<<if $age in 20..30>>z<<end>>');
		await openModal();
		await typeInto(modal().querySelector<HTMLInputElement>('input[aria-label="From"]')!, '40');
		expect(saveDisabled()).toBe(true);
		expect(problems()).toEqual(['If: the From number for $age is above To.']);
		await typeInto(modal().querySelector<HTMLInputElement>('input[aria-label="To"]')!, '50');
		expect(saveDisabled()).toBe(false);
	});
});

describe('condition editor problems', () => {
	it('names the unfinished branch and highlights its value', async () => {
		setup('<<if $name = "a">>x<<end>>');
		const pop = await openModal();
		await click(byText(pop, 'button', 'Else if'));
		await click(byText(pop, 'button', 'Else if'));
		await pick(operator(branches()[2]), '=');
		expect(problems()).toEqual(['Else if 2: type the text to compare $name with.']);
		expect(textValue(branches()[2]).getAttribute('aria-invalid')).toBe('true');
		expect(textValue(branches()[2]).className).toContain('border-warning');
		expect(branches()[1].dataset.invalid).toBeUndefined();
		expect(pop.textContent).not.toContain('Finish the highlighted values');
	});

	it('lists every problem, one line each', async () => {
		setup('<<if $name = "a">>x<<else>>y<<end>>');
		await openModal();
		await typeInto(textValue(branches()[0]), '');
		await typeInto(body(branches()[1]), 'y <<if $age > 1>>b<<end>>');
		expect(problems()).toEqual([
			'If: type the text to compare $name with.',
			"Otherwise text contains another condition. Conditions can't be nested, so add an Else if instead."
		]);
		expect(body(branches()[1]).getAttribute('aria-invalid')).toBe('true');
	});

	it('moves a condition typed into the otherwise text into an else if', async () => {
		const { changes } = setup('<<if $age > 65>>grey<<else>>auburn<<end>>');
		const pop = await openModal();
		await typeInto(body(branches()[1]), '<<if $age < 25>>young<<else>>auburn<<end>>');
		expect(saveDisabled()).toBe(true);
		await click(byText(pop, 'button', 'Use Else if'));
		expect(branches().map((b) => b.dataset.kind)).toEqual(['if', 'elseif', 'else']);
		expect(pop.querySelector('[data-testid="condition-invalid"]')).toBeNull();
		await save();
		expect(changes).toEqual(['<<if $age > 65>>grey<<else if $age < 25>>young<<else>>auburn<<end>>']);
	});

	it('offers the conversion only on the otherwise it fixes', async () => {
		setup('<<if $age > 65>>grey<<else>>auburn<<end>>');
		const pop = await openModal();
		await typeInto(body(branches()[1]), '<<if $age < 25>>young<<end>>');
		await typeInto(body(branches()[0]), 'grey <<if $age > 70>>white<<end>>');
		expect(problems()).toHaveLength(2);
		const buttons = Array.from(pop.querySelectorAll<HTMLElement>('[data-testid="condition-problem"]')).map(
			(item) => byText(item, 'button', 'Use Else if') !== undefined
		);
		expect(buttons).toEqual([false, true]);
	});

	it('offers no conversion when the nested condition sits after other text or in another branch', async () => {
		setup('<<if $age > 65>>grey<<else>>auburn<<end>>');
		const pop = await openModal();
		await typeInto(body(branches()[1]), 'auburn <<if $age < 25>>young<<end>>');
		expect(problems()).toHaveLength(1);
		expect(byText(pop, 'button', 'Use Else if')).toBeUndefined();
		await typeInto(body(branches()[1]), 'auburn');
		await typeInto(body(branches()[0]), '<<if $age < 25>>young<<end>>');
		expect(problems()).toEqual([
			"If text contains another condition. Conditions can't be nested, so add an Else if instead."
		]);
		expect(byText(pop, 'button', 'Use Else if')).toBeUndefined();
	});
});
