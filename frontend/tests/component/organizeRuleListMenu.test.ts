// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import RuleList from '$lib/components/organize/RuleList.svelte';
import { catalogFixture, ruleFixture } from './organizeFixtures';

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

let instance: ReturnType<typeof mount> | null = null;
let callbacks: Record<string, ReturnType<typeof vi.fn>>;

const rules = [
	ruleFixture({ id: 'r1', name: 'First', stop_after: false }),
	ruleFixture({ id: 'r2', name: 'Middle', stop_after: true }),
	ruleFixture({ id: 'r3', name: 'Last', stop_after: false })
];

function render() {
	callbacks = {
		onOpen: vi.fn(),
		onToggle: vi.fn(),
		onToggleStop: vi.fn(),
		onDuplicate: vi.fn(),
		onDelete: vi.fn(),
		onReorder: vi.fn()
	};
	instance = mount(RuleList as never, {
		target: document.body,
		props: { rules, catalog: catalogFixture, labels: {}, ...callbacks }
	});
	flushSync();
}

async function openMenuFor(index: number) {
	document.querySelectorAll<HTMLButtonElement>('button[aria-label="Rule actions"]')[index].click();
	await flush();
	await flush();
}

const itemLabels = () => Array.from(document.querySelectorAll('[role="menuitem"]')).map((b) => b.textContent?.trim());

function choose(label: string) {
	Array.from(document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'))
		.find((b) => b.textContent?.trim() === label)!
		.click();
}

beforeEach(render);

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
});

describe('RuleList actions menu', () => {
	it('offers no Move up on the first rule', async () => {
		await openMenuFor(0);
		expect(itemLabels()).toEqual(['Stop after this rule', 'Move down', 'Duplicate', 'Delete']);
	});

	it('offers both moves in the middle and flips the stop label when stop_after is set', async () => {
		await openMenuFor(1);
		expect(itemLabels()).toEqual(['Do not stop after this rule', 'Move up', 'Move down', 'Duplicate', 'Delete']);
	});

	it('offers no Move down on the last rule', async () => {
		await openMenuFor(2);
		expect(itemLabels()).toEqual(['Stop after this rule', 'Move up', 'Duplicate', 'Delete']);
	});

	it('routes each item to its callback', async () => {
		await openMenuFor(1);
		choose('Do not stop after this rule');
		expect(callbacks.onToggleStop).toHaveBeenCalledWith(rules[1]);

		await openMenuFor(1);
		choose('Move up');
		expect(callbacks.onReorder).toHaveBeenLastCalledWith(['r2', 'r1', 'r3']);

		await openMenuFor(1);
		choose('Move down');
		expect(callbacks.onReorder).toHaveBeenLastCalledWith(['r1', 'r3', 'r2']);

		await openMenuFor(1);
		choose('Duplicate');
		expect(callbacks.onDuplicate).toHaveBeenCalledWith(rules[1]);

		await openMenuFor(1);
		choose('Delete');
		expect(callbacks.onDelete).toHaveBeenCalledWith(rules[1]);
		expect(callbacks.onDuplicate).toHaveBeenCalledTimes(1);
	});
});
