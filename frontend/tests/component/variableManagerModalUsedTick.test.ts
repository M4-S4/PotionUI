// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableManagerModal } = await import('../../src/lib/components/VariableManagerModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

type ModalHandle = { $destroy: () => void; $set: (props: Record<string, unknown>) => void };
let component: ModalHandle | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

const variables: VariablesMap = {
	mood: { type: 'text', value: 'calm' },
	place: { type: 'text', value: 'forest' }
};

function open(usageCounts: Record<string, number> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: VariableManagerModal as never,
		target,
		props: { isOpen: true, variables, usageCounts }
	}) as unknown as ModalHandle;
}

const ticks = () => Array.from(document.querySelectorAll('[data-testid="variable-used-tick"]'));

describe('VariableManagerModal used tick', () => {
	it('marks only used variables and labels the count', () => {
		open({ mood: 3, place: 0 });
		expect(ticks()).toHaveLength(1);
		expect(ticks()[0].closest('[role="option"]')?.textContent).toContain('$mood');
		expect(ticks()[0].getAttribute('aria-label')).toBe('Used in the prompt 3 times');
	});

	it('shows no tick when nothing is used', () => {
		open({});
		expect(ticks()).toHaveLength(0);
	});

	it('appears and disappears as usageCounts changes', async () => {
		open({});
		expect(ticks()).toHaveLength(0);
		component!.$set({ usageCounts: { mood: 1, place: 2 } });
		await tick();
		expect(ticks()).toHaveLength(2);
		component!.$set({ usageCounts: { place: 2 } });
		await tick();
		expect(ticks()).toHaveLength(1);
		expect(ticks()[0].closest('[role="option"]')?.textContent).toContain('$place');
		component!.$set({ usageCounts: {} });
		await tick();
		expect(ticks()).toHaveLength(0);
	});
});
