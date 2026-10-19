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

function open(
	variables: VariablesMap,
	usageCounts: Record<string, number> = {},
	readers: Record<string, { label: string; segment?: string }[]> = {}
) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: VariableManagerModal as never,
		target,
		props: { isOpen: true, variables, usageCounts, readers }
	}) as unknown as ModalHandle;
	let latest: VariablesMap | null = null;
	component.$on('change', (e) => (latest = e.detail));
	return () => latest as any;
}

const byText = (selector: string, text: string) =>
	Array.from(document.querySelectorAll<HTMLElement>(selector)).find((el) => el.textContent?.trim() === text)!;

const click = async (el: HTMLElement) => {
	el.click();
	await tick();
};

const setInput = async (label: string, value: string) => {
	const input = document.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	input.dispatchEvent(new Event('blur'));
	await tick();
};

const age: VariablesMap = { age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 47 } };

describe('VariableManagerModal number tab', () => {
	it('shows the range, mode and type tab for a number variable', () => {
		open(age, { age: 2 });
		expect(document.querySelector('[data-testid="number-editor"]')).not.toBeNull();
		expect(document.querySelector('[role="option"]')!.textContent).toContain('18–80 · shuffle · 2 uses');
		expect(byText('[role="tab"]', 'Number').getAttribute('aria-selected')).toBe('true');
		expect(document.querySelector('[data-testid="number-preview-note"]')!.textContent!.replace(/\s+/g, ' ')).toContain('from 63 values: 18, 19, 20 \u2026 80');
	});

	it('switches a text variable to a number variable', async () => {
		const latest = open({ name: { type: 'text', value: 'x' } });
		await click(byText('[role="tab"]', 'Number'));
		expect(latest().name.type).toBe('number');
		expect(document.querySelector('[data-testid="number-editor"]')).not.toBeNull();
	});

	it('changes the mode and commits it', async () => {
		const latest = open(age);
		await click(byText('button[role="radio"]', 'Per image'));
		expect(latest().age.mode).toBe('per-image');
		expect(document.querySelectorAll('[data-testid="number-dot"]').length).toBe(4);
		await click(byText('button[role="radio"]', 'Pin'));
		expect(latest().age.mode).toBe('pin');
		expect(document.querySelector('input[aria-label="Pinned value"]')).not.toBeNull();
		expect(document.querySelectorAll('[data-testid="number-dot"]').length).toBe(1);
	});

	it('pins inside the range when max is lowered', async () => {
		const latest = open({ age: { ...(age.age as object), mode: 'pin', value: 70 } as any });
		await setInput('Maximum', '50');
		expect(latest().age.max).toBe(50);
		expect(latest().age.value).toBe(50);
	});

	it('moves max up when min is raised past it', async () => {
		const latest = open(age);
		await setInput('Minimum', '90');
		expect(latest().age.min).toBe(90);
		expect(latest().age.max).toBe(90);
	});

	it('switches Whole and Decimal', async () => {
		const latest = open(age);
		await click(byText('button[role="radio"]', 'Decimal'));
		expect(latest().age.step).toBe(0.1);
		expect(document.querySelector('[data-testid="number-grid-warning"]')!.textContent).toContain('Raise the step');
		await click(byText('button[role="radio"]', 'Whole'));
		expect(latest().age.step).toBe(1);
	});

	it('warns when the grid exceeds 200 values', () => {
		open({ age: { type: 'number', mode: 'shuffle', min: 0, max: 1000, step: 1, value: 0 } });
		expect(document.querySelector('[data-testid="number-grid-warning"]')!.textContent).toContain('1001 values');
		expect(document.querySelector('[data-testid="number-track"]')).toBeNull();
	});

	it('explains linking when a condition reads the variable', () => {
		open(age, { age: 1 }, { age: [{ label: 'age > 65', segment: 'segment 01' }] });
		expect(document.querySelector('[data-testid="number-usage-note"]')!.textContent).toContain('Linked because a condition reads it');
		expect(document.querySelectorAll('[data-testid="number-used-by-condition"]').length).toBe(1);
		expect(byText('button[role="radio"]', 'Linked').hasAttribute('disabled')).toBe(true);
	});

	it('lets a shuffle number be set to Linked', async () => {
		const latest = open(age);
		await click(byText('button[role="radio"]', 'Linked'));
		expect(latest().age.linked).toBe(true);
	});

	it('steps the minimum with the buttons', async () => {
		const latest = open(age);
		await click(document.querySelector<HTMLElement>('button[aria-label="Increase Minimum"]')!);
		expect(latest().age.min).toBe(19);
	});
});
