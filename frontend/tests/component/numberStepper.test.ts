// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';

const { default: NumberStepper } = await import('$lib/components/NumberStepper.svelte');

let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	document.body.innerHTML = '';
});

function setup(commitOnInput: boolean) {
	const changes: number[] = [];
	instance = mount(NumberStepper, {
		target: document.body,
		props: { value: 5, decimals: 0, min: 0, max: 50, label: 'Value', commitOnInput, onChange: (v: number) => changes.push(v) }
	});
	flushSync();
	return { changes, input: document.querySelector<HTMLInputElement>('input[aria-label="Value"]')! };
}

async function type(input: HTMLInputElement, value: string) {
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

describe('NumberStepper commit timing', () => {
	it('commits each typed number at once when asked, keeping what was typed', async () => {
		const { changes, input } = setup(true);
		await type(input, '1');
		await type(input, '12.');
		await type(input, '-');
		await type(input, '120');
		expect(changes).toEqual([1, 12, 50]);
		expect(input.value).toBe('120');
		input.dispatchEvent(new FocusEvent('blur'));
		await tick();
		expect(changes).toEqual([1, 12, 50, 50]);
	});

	it('waits for blur by default', async () => {
		const { changes, input } = setup(false);
		await type(input, '12');
		expect(changes).toEqual([]);
		input.dispatchEvent(new FocusEvent('blur'));
		await tick();
		expect(changes).toEqual([12]);
	});
});
