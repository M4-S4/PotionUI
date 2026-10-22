// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import { MAX_STRING_CHARS } from '$lib/utils/promptConditions';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { requireConditionModal as modal } from './helpers/typing';

const { default: ConditionChip } = await import('$lib/components/ConditionChip.svelte');

const SUIT = 'a tailored charcoal three-piece suit with a crisp white shirt, slim black tie and polished oxford shoes';
const SWEATER = 'an oversized cream cable-knit sweater over faded high-waisted jeans and scuffed white canvas sneakers';
const GOWN = 'a flowing emerald silk evening gown with a low open back and long satin opera gloves';
const JACKET = 'a worn leather motorcycle jacket over a faded black band t-shirt and heavy lace-up combat boots';
const CAP = 'a technical outdoor shell jacket in signal orange'.padEnd(MAX_STRING_CHARS, ' with taped seams');
const KIMONO = 'a hand-embroidered ceremonial kimono in deep indigo silk'.padEnd(MAX_STRING_CHARS + 20, ' with golden cranes');

const variables: VariablesMap = {
	outfit: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [SUIT, SWEATER, GOWN, JACKET, CAP, KIMONO] },
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] },
	cloak: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [KIMONO] }
};

const SWITCH = `<<if $outfit = "${SUIT}">>boardroom<<else if $outfit in "${SWEATER}", "${GOWN}">>terrace<<else>>studio<<end>>`;

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
	const changes: string[] = [];
	instance = mount(ConditionChip, {
		target,
		props: { raw, context: writable({ variables, rolls: {} }), variant: 'segment-composer', onchange: (next: string) => changes.push(next) }
	});
	flushSync();
	return { changes };
}

const branches = () => Array.from(modal().querySelectorAll<HTMLElement>('[data-testid="condition-branch"]'));
const trigger = (branch: HTMLElement) => branch.querySelector<HTMLButtonElement>('[data-testid="option-select"]')!;
const picker = () => document.querySelector<HTMLElement>('[data-testid="option-picker"]');
const search = () => picker()!.querySelector<HTMLInputElement>('input[role="combobox"]')!;
const rows = () => Array.from(picker()!.querySelectorAll<HTMLElement>('[role="option"]'));
const row = (text: string) => rows().find((el) => el.dataset.option === text)!;
const used = (text: string) => row(text).querySelector('[data-testid="option-used"]')?.textContent?.trim() ?? '';
const picked = (branch: HTMLElement) =>
	Array.from(trigger(branch).querySelectorAll<HTMLElement>('[data-option-value]')).map((el) => el.dataset.optionValue);
const button = (label: string) =>
	Array.from(modal().querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith(label))!;

async function openModal() {
	target!.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	await tick();
}

async function openValues(branch: HTMLElement) {
	trigger(branch).click();
	await tick();
	await tick();
}

async function key(name: string) {
	search().dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true }));
	await tick();
	await tick();
}

async function type(value: string) {
	search().value = value;
	search().dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

async function choose(branch: HTMLElement, text: string) {
	await openValues(branch);
	row(text).click();
	await tick();
	await tick();
}

async function operator(branch: HTMLElement, op: string) {
	const select = branch.querySelector<HTMLSelectElement>('select[aria-label="Operator"]')!;
	select.value = op;
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await tick();
}

describe('condition value dropdown', () => {
	it('searches the options and names the query when nothing matches', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(rows()).toHaveLength(6);
		expect(picker()!.querySelector('[data-testid="option-count"]')!.textContent?.trim()).toBe('6');
		await type('satin');
		expect(rows().map((r) => r.dataset.option)).toEqual([GOWN]);
		expect(picker()!.querySelector('[data-testid="option-count"]')!.textContent?.trim()).toBe('1 of 6');
		await type('zzzz');
		expect(rows()).toHaveLength(0);
		expect(picker()!.querySelector('[data-testid="option-picker-empty"]')!.textContent).toContain('zzzz');
	});

	it('keeps long options readable and blocks only the ones over the cap', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(row(CAP).getAttribute('aria-disabled')).toBeNull();
		expect(row(KIMONO).getAttribute('aria-disabled')).toBe('true');
		expect(row(KIMONO).textContent).toContain(`Longer than ${MAX_STRING_CHARS} characters`);
		expect(row(CAP).querySelector('.line-clamp-3')!.className).toContain('break-words');
		row(KIMONO).click();
		await tick();
		expect(picker()).not.toBeNull();
		row(CAP).click();
		await tick();
		await tick();
		expect(picker()).toBeNull();
		expect(picked(branches()[0])).toEqual([CAP]);
		expect(trigger(branches()[0]).querySelector('.line-clamp-2')).not.toBeNull();
	});

	it('picks with the arrow keys and Enter, and Esc clears the search before closing', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(rows().find((r) => r.dataset.active === 'true')!.dataset.option).toBe(SUIT);
		await type('leather');
		await key('Escape');
		expect(search().value).toBe('');
		expect(picker()).not.toBeNull();
		await key('ArrowDown');
		await key('Enter');
		expect(picker()).toBeNull();
		expect(modal()).not.toBeNull();
		expect(picked(branches()[0])).toEqual([SWEATER]);
		await openValues(branches()[0]);
		await key('Escape');
		expect(picker()).toBeNull();
		expect(modal()).not.toBeNull();
	});
});

describe('keys under an open dropdown', () => {
	it('leaves Ctrl+Enter to the dropdown on top instead of saving underneath it', async () => {
		const { changes } = setup(SWITCH);
		await openModal();
		await choose(branches()[0], JACKET);
		await openValues(branches()[0]);
		search().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
		await tick();
		expect(changes).toEqual([]);
		expect(modal()).not.toBeNull();
		await key('Escape');
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
		await tick();
		expect(changes).toHaveLength(1);
	});
});

describe('used in this condition already', () => {
	it('marks the options other branches test, and not the branch’s own', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(used(SWEATER)).toBe('Used in Else if');
		expect(used(GOWN)).toBe('Used in Else if');
		expect(used(SUIT)).toBe('');
		expect(used(JACKET)).toBe('');
		await key('Escape');
		await openValues(branches()[1]);
		expect(used(SUIT)).toBe('Used in If');
		expect(used(SWEATER)).toBe('');
	});

	it('marks options tested with is not and is none of, naming the operator', async () => {
		setup(`<<if $outfit != "${JACKET}">>a<<else if $outfit not in "${SUIT}", "${GOWN}">>b<<else if $outfit = "${SWEATER}">>c<<end>>`);
		await openModal();
		await openValues(branches()[2]);
		expect(used(JACKET)).toBe('Used in If · is not');
		expect(used(SUIT)).toBe('Used in Else if 1 · is none of');
		expect(used(GOWN)).toBe('Used in Else if 1 · is none of');
		expect(used(SWEATER)).toBe('');
	});

	it('updates the markers after a branch picks another option', async () => {
		setup(SWITCH);
		await openModal();
		await choose(branches()[0], JACKET);
		await openValues(branches()[1]);
		expect(used(JACKET)).toBe('Used in If');
		expect(used(SUIT)).toBe('');
	});
});

describe('never runs', () => {
	it('explains when an earlier branch already catches the picked value', async () => {
		setup(SWITCH);
		await openModal();
		expect(modal().querySelector('[data-testid="condition-never-runs"]')).toBeNull();
		await choose(branches()[1], SUIT);
		const note = branches()[1].querySelector('[data-testid="condition-never-runs"]')!;
		expect(note.textContent?.replace(/\s+/g, ' ')).toContain('If already catches');
		expect(note.textContent).toContain('so Else if never runs for it.');
	});

	it('counts an earlier is not as catching everything else', async () => {
		setup(`<<if $outfit != "${GOWN}">>a<<else if $outfit = "${GOWN}">>b<<end>>`);
		await openModal();
		expect(branches()[1].querySelector('[data-testid="condition-never-runs"]')).toBeNull();
		await choose(branches()[1], JACKET);
		expect(branches()[1].querySelector('[data-testid="condition-never-runs"]')!.textContent).toContain('If already catches');
	});
});

describe('a new Else if', () => {
	it('starts on the first option no branch covers yet, before Otherwise', async () => {
		const { changes } = setup(SWITCH);
		await openModal();
		button('Else if').click();
		await tick();
		expect(branches().map((b) => b.dataset.kind)).toEqual(['if', 'elseif', 'elseif', 'else']);
		expect(picked(branches()[2])).toEqual([JACKET]);
		button('Save').click();
		await tick();
		expect(changes.at(-1)).toContain(`<<else if $outfit = "${JACKET}">><<else>>`);
	});

	it('falls back to the first usable option when every option is covered', async () => {
		setup(`<<if $mood = "calm">>a<<else if $mood = "tense">>b<<end>>`);
		await openModal();
		button('Else if').click();
		await tick();
		expect(picked(branches()[2])).toEqual(['calm']);
	});
});

describe('problems in place', () => {
	it('shows the message under the value it belongs to and counts it in the footer', async () => {
		setup(`<<if $outfit = "${SUIT}">>a<<else if $cloak is empty>>b<<end>>`);
		await openModal();
		expect(modal().querySelector('[data-testid="condition-invalid"]')).toBeNull();
		await operator(branches()[1], '=');
		const second = branches()[1];
		const message = second.querySelector('[data-testid="condition-problem"]')!;
		expect(message.textContent).toContain(`The option picked for $cloak is longer than ${MAX_STRING_CHARS} characters`);
		expect(trigger(second).compareDocumentPosition(message) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(trigger(second).getAttribute('aria-invalid')).toBe('true');
		expect(picked(second)).toEqual([]);
		expect(branches()[0].querySelector('[data-testid="condition-problem"]')).toBeNull();
		expect(modal().querySelector('[data-testid="condition-invalid"]')!.textContent).toBe('1 problem');
		expect(button('Save').disabled).toBe(true);
		button('1 problem').click();
		await tick();
		expect(document.activeElement).toBe(trigger(second));
	});
});

describe('value dropdown accessibility', () => {
	it('is a listbox and points aria-activedescendant at the active option while arrowing', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(picker()!.querySelector('[role="listbox"]')).not.toBeNull();
		expect(search().getAttribute('aria-activedescendant')).toBe(row(SUIT).id);
		await key('ArrowDown');
		expect(search().getAttribute('aria-activedescendant')).toBe(row(SWEATER).id);
		await key('ArrowDown');
		expect(search().getAttribute('aria-activedescendant')).toBe(row(GOWN).id);
		expect(row(GOWN).dataset.active).toBe('true');
	});

	it('returns focus to the trigger after Escape and after picking an option', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		expect(document.activeElement).toBe(search());
		await key('Escape');
		await tick();
		expect(picker()).toBeNull();
		expect(document.activeElement).toBe(trigger(branches()[0]));
		await openValues(branches()[0]);
		row(SWEATER).click();
		await tick();
		await tick();
		expect(picker()).toBeNull();
		expect(document.activeElement).toBe(trigger(branches()[0]));
	});

	it('closes on Tab without pulling focus back to the trigger', async () => {
		setup(SWITCH);
		await openModal();
		await openValues(branches()[0]);
		await key('Tab');
		await tick();
		expect(picker()).toBeNull();
		expect(document.activeElement).not.toBe(trigger(branches()[0]));
	});
});
