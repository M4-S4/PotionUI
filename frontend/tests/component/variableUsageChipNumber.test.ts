// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';

const { default: VariableUsageChip } = await import('../../src/lib/components/VariableUsageChip.svelte');
const { createClassComponent } = await import('svelte/legacy');

let component: { $destroy: () => void } | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

function mount(definition: unknown) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: VariableUsageChip as never,
		target,
		props: { name: 'age', definition, variables: { age: definition } }
	}) as never;
	return target;
}

const label = (t: HTMLElement) => t.querySelector('[data-testid="number-chip-label"]')?.textContent;

describe('VariableUsageChip number states', () => {
	const base = { type: 'number', min: 18, max: 80, step: 1, value: 47 };

	it('shows the range for shuffle', () => {
		expect(label(mount({ ...base, mode: 'shuffle' }))).toBe('18–80');
	});

	it('shows the pinned value', () => {
		expect(label(mount({ ...base, mode: 'pin' }))).toBe('47');
	});

	it('shows per image', () => {
		expect(label(mount({ ...base, mode: 'per-image' }))).toBe('per image');
	});

	it('shows decimals from the step', () => {
		expect(label(mount({ type: 'number', mode: 'shuffle', min: 0.8, max: 1.4, step: 0.1, value: 1 }))).toBe('0.8–1.4');
	});
});


describe('VariableUsageChip number linking from condition readers', () => {
	const def = { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 47 };

	function mountComposer(readers: Record<string, unknown[]>) {
		const target = document.createElement('div');
		document.body.appendChild(target);
		component = createClassComponent({
			component: VariableUsageChip as never,
			target,
			props: {
				name: 'age',
				definition: def,
				variables: { age: def },
				readers,
				variant: 'segment-composer',
				usage: { ordinal: 1, total: 2 }
			}
		}) as never;
		return target;
	}

	it('shows Linked when a condition reads the variable', () => {
		expect(mountComposer({ age: [{ label: 'age > 65' }] }).querySelector('.chip-state')!.textContent).toBe('Linked 2');
	});

	it('shows the usage position when nothing reads it', () => {
		expect(mountComposer({}).querySelector('.chip-state')!.textContent).toBe('1/2');
	});
});
