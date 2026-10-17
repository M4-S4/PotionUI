// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableUsageChip } = await import('$lib/components/VariableUsageChip.svelte');

let component: ReturnType<typeof mount> | undefined;

afterEach(() => {
	if (component) unmount(component);
	component = undefined;
	document.body.innerHTML = '';
});

const OPTIONS = ['red', 'green', 'blue', 'amber', 'violet', 'teal', 'coral', 'ivory'];

function mountChip(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(VariableUsageChip as never, { target, props: { variant: 'segment-composer', ...props } });
	flushSync();
	target.querySelector<HTMLButtonElement>('.chip-main')?.dispatchEvent(
		new MouseEvent('mousedown', { bubbles: true, cancelable: true })
	);
	flushSync();
	return target;
}

function choiceVars(extra: Record<string, unknown> = {}): VariablesMap {
	return {
		color: { type: 'choice', options: OPTIONS, mode: 'shuffle', pinnedIndex: null, ...extra }
	} as VariablesMap;
}

function usagesButtons(): HTMLButtonElement[] {
	const labels = Array.from(document.querySelectorAll<HTMLElement>('.variable-popover .section-label'));
	const label = labels.find((l) => l.textContent?.startsWith('Usages'));
	return Array.from(label?.parentElement?.querySelectorAll<HTMLButtonElement>('.behavior-button') ?? []);
}

describe('VariableUsageChip usages', () => {
	it('shows the n/N badge and this usage roll', () => {
		const variables = choiceVars();
		const root = mountChip({
			name: 'color',
			definition: variables.color,
			variables,
			usage: { ordinal: 2, total: 3 },
			rolls: {
				color: { value: 'red', optionIndex: 0 },
				'color-2': { value: 'blue', optionIndex: 2 }
			}
		});
		expect(root.querySelector('.chip-state')?.textContent).toBe('2/3');
		expect(root.querySelector('.chip-label')?.textContent).toBe('blue');
	});

	it('uses the shared roll and a Linked badge when linked', () => {
		const variables = choiceVars({ linked: true });
		const root = mountChip({
			name: 'color',
			definition: variables.color,
			variables,
			usage: { ordinal: 2, total: 3 },
			rolls: { color: { value: 'red', optionIndex: 0 }, 'color-2': { value: 'blue', optionIndex: 2 } }
		});
		expect(root.querySelector('.chip-state')?.textContent).toBe('Linked 3');
		expect(root.querySelector('.chip-label')?.textContent).toBe('red');
	});

	it('Usages control reports the chosen mode', () => {
		const variables = choiceVars();
		const onLinkedChange = vi.fn();
		mountChip({ name: 'color', definition: variables.color, variables, usage: { ordinal: 1, total: 2 }, onLinkedChange });
		const [independent, linked] = usagesButtons();
		expect(independent.classList.contains('active')).toBe(true);
		linked.click();
		expect(onLinkedChange).toHaveBeenCalledWith(true);
	});

	it('forces Linked and disables the control for a when-target', () => {
		const variables = {
			...choiceVars(),
			shirt: {
				type: 'choice',
				options: [{ text: 'tee', when: { var: 'color', values: ['red'] } }, 'plain'],
				mode: 'shuffle',
				pinnedIndex: null
			}
		} as VariablesMap;
		mountChip({ name: 'color', definition: variables.color, variables, usage: { ordinal: 1, total: 2 } });
		const [independent, linked] = usagesButtons();
		expect(independent.disabled).toBe(true);
		expect(linked.disabled).toBe(true);
		expect(linked.classList.contains('active')).toBe(true);
	});

	it('filters the option list and marks the rolled row for this usage', () => {
		const variables = choiceVars();
		mountChip({
			name: 'color',
			definition: variables.color,
			variables,
			usage: { ordinal: 2, total: 2 },
			rolls: { 'color-2': { value: 'green', optionIndex: 1 } }
		});
		const rows = () => document.querySelectorAll('.variable-popover .option-row');
		expect(rows().length).toBe(OPTIONS.length);
		expect(document.querySelector('.option-row.current .option-text')?.textContent).toBe('green');

		const filter = document.querySelector<HTMLInputElement>('.option-filter input');
		expect(filter).not.toBeNull();
		filter!.value = 'e';
		filter!.dispatchEvent(new Event('input', { bubbles: true }));
		flushSync();
		expect(rows().length).toBe(OPTIONS.filter((o) => o.includes('e')).length);
	});

	it('pin mode renders radio rows instead of a select', () => {
		const variables = choiceVars({ mode: 'pin', pinnedIndex: 1 });
		const onModeChange = vi.fn();
		mountChip({ name: 'color', definition: variables.color, variables, onModeChange });
		expect(document.querySelector('.variable-popover select')).toBeNull();
		const radios = document.querySelectorAll<HTMLButtonElement>('.option-row[role="radio"]');
		expect(radios[1].getAttribute('aria-checked')).toBe('true');
		radios[3].click();
		expect(onModeChange).toHaveBeenCalledWith('pin', 3);
	});
});
