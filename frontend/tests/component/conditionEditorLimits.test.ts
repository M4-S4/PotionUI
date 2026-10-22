// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { writable } from 'svelte/store';
import { MAX_BRANCHES, MAX_TESTS } from '$lib/utils/promptConditions';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { requireConditionModal } from './helpers/typing';

const { default: ConditionChip } = await import('$lib/components/ConditionChip.svelte');

const variables: VariablesMap = {
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 }
};

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLDivElement | undefined;

function close() {
	if (instance) unmount(instance);
	instance = undefined;
	target?.remove();
	document.body.innerHTML = '';
}

afterEach(close);

async function openModal(raw: string) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const context = writable({ variables, rolls: {} });
	instance = mount(ConditionChip, {
		target,
		props: { raw, context, variant: 'segment-composer', onchange: () => {} }
	});
	flushSync();
	target.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
	await tick();
	await tick();
	return requireConditionModal();
}

const hasButton = (root: ParentNode, text: string) =>
	Array.from(root.querySelectorAll<HTMLElement>('button')).some((el) => el.textContent?.trim() === text);

const allTests = (count: number) => Array.from({ length: count }, (_, i) => `$age > ${i}`).join(' and ');

const withBranches = (count: number, otherwise: boolean) => {
	const elseIfs = Array.from({ length: count - 1 - (otherwise ? 1 : 0) }, (_, i) => `<<else if $age > ${i + 2}>>b`);
	return `<<if $age > 1>>a${elseIfs.join('')}${otherwise ? '<<else>>z' : ''}<<end>>`;
};

describe('condition editor caps', () => {
	it('offers Add test below the tests cap and stops at it', async () => {
		const below = await openModal(`<<if ${allTests(MAX_TESTS - 1)}>>a<<end>>`);
		expect(hasButton(below, 'Add test')).toBe(true);
		close();
		const at = await openModal(`<<if ${allTests(MAX_TESTS)}>>a<<end>>`);
		expect(hasButton(at, 'Add test')).toBe(false);
	});

	it('offers Else if and Otherwise below the branches cap and stops at it', async () => {
		const below = await openModal(withBranches(MAX_BRANCHES - 1, false));
		expect(hasButton(below, 'Else if')).toBe(true);
		expect(hasButton(below, 'Otherwise')).toBe(true);
		close();
		const at = await openModal(withBranches(MAX_BRANCHES, false));
		expect(hasButton(at, 'Else if')).toBe(false);
		expect(hasButton(at, 'Otherwise')).toBe(false);
	});

	it('closes the Else if button at the cap when the block already ends in Otherwise', async () => {
		const below = await openModal(withBranches(MAX_BRANCHES - 1, true));
		expect(hasButton(below, 'Else if')).toBe(true);
		close();
		const at = await openModal(withBranches(MAX_BRANCHES, true));
		expect(hasButton(at, 'Else if')).toBe(false);
	});
});
