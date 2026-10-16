// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableManagerModal } = await import('../../src/lib/components/VariableManagerModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

type ModalHandle = { $destroy: () => void; $on: (name: string, cb: (e: CustomEvent<VariablesMap>) => void) => void };
let component: ModalHandle | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

const variables: VariablesMap = {
	color: { type: 'choice', options: ['red', 'green', 'blue'], mode: 'shuffle', pinnedIndex: null },
	mood: { type: 'choice', options: ['calm', 'tense'], mode: 'shuffle', pinnedIndex: null },
	style: { type: 'text', value: 'oil' }
};

function open(usageCounts: Record<string, number> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: VariableManagerModal as never,
		target,
		props: { isOpen: true, variables, usageCounts }
	}) as unknown as ModalHandle;
	let latest: VariablesMap | null = null;
	component.$on('change', (e) => (latest = e.detail));
	return () => latest;
}

const rows = () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="option"]'));

describe('VariableManagerModal master-detail', () => {
	it('lists every variable with meta and edits the selected one', () => {
		open({ color: 3 });
		expect(rows().map((r) => r.querySelector('span')?.textContent)).toEqual(['$color', '$mood', '$style']);
		expect(rows()[0].textContent).toContain('3 opt · shuffle · 3 uses');
		expect(rows()[0].getAttribute('aria-selected')).toBe('true');
		expect(document.querySelectorAll('textarea').length).toBe(3);
	});

	it('filters the list by name or option text', async () => {
		open();
		const filter = document.querySelector<HTMLInputElement>('input[aria-label="Filter variables"]')!;
		filter.value = 'tense';
		filter.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		expect(rows().map((r) => r.textContent)).toHaveLength(1);
		expect(rows()[0].textContent).toContain('$mood');
	});

	it('sets linked from the Usages control', async () => {
		const latest = open();
		const linked = Array.from(document.querySelectorAll<HTMLButtonElement>('button[role="radio"]')).find(
			(b) => b.textContent?.trim() === 'Linked'
		)!;
		linked.click();
		await tick();
		expect((latest() as any).color.linked).toBe(true);
	});

	it('pin mode shows radio rows and pins the clicked option', async () => {
		const latest = open();
		Array.from(document.querySelectorAll<HTMLButtonElement>('button[role="radio"]'))
			.find((b) => b.textContent?.trim() === 'Pin')!
			.click();
		await tick();
		expect(document.querySelector('select')).toBeNull();
		document.querySelector<HTMLButtonElement>('button[aria-label="Pin option 3"]')!.click();
		await tick();
		expect((latest() as any).color.mode).toBe('pin');
		expect((latest() as any).color.pinnedIndex).toBe(2);
	});
});
