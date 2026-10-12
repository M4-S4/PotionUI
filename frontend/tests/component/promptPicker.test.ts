// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import PromptPicker from '$lib/components/PromptPicker.svelte';

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

function entry(prompt: string, name?: string) {
	return { prompt, negativePrompt: '', promptSegments: [], negativePromptSegments: [], ...(name ? { name } : {}) };
}

function render(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const handlers = {
		onSelect: vi.fn(),
		onAdd: vi.fn(),
		onDuplicate: vi.fn(),
		onDelete: vi.fn(),
		onRename: vi.fn()
	};
	const instance = mount(PromptPicker, {
		target,
		props: {
			prompts: [entry('a lighthouse')],
			active: 0,
			perPrompt: 2,
			canBatch: true,
			overCap: false,
			maxPrompts: 10,
			...handlers,
			...props
		}
	});
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return { target, handlers };
}

async function openMenu(target: HTMLElement) {
	target.querySelector<HTMLButtonElement>('[data-prompt-picker]')!.click();
	flushSync();
	await tick();
	return document.body.querySelector<HTMLElement>('[role="dialog"]')!;
}

describe('PromptPicker', () => {
	it('reads "1 prompt" for a single prompt and offers no delete', async () => {
		const { target } = render();

		expect(target.querySelector('[data-prompt-picker]')!.textContent).toContain('1 prompt');
		const menu = await openMenu(target);
		expect(menu.querySelector('[aria-label="Delete prompt"]')).toBeNull();
		expect(menu.textContent).toContain('Add prompt starts empty');
	});

	it('names the current prompt and its position in the chip', () => {
		const { target } = render({
			prompts: [entry('one'), entry('red fox curled in leaves'), entry('three')],
			active: 1
		});

		const chip = target.querySelector('[data-prompt-picker]')!.textContent!;
		expect(chip).toContain('Prompt 2 of 3');
		expect(chip).toContain('red fox curled in leaves');
	});

	it('shows the prompt name only when it differs from the default', () => {
		const { target } = render({ prompts: [entry(''), entry('')], active: 1 });

		const chip = target.querySelector('[data-prompt-picker]')!.textContent!;
		expect(chip).toContain('Prompt 2 of 2');
		expect(chip.match(/Prompt 2/g)).toHaveLength(1);
	});

	it('lists a row per prompt with its image count and selects on click', async () => {
		const { target, handlers } = render({ prompts: [entry('one'), entry('two', 'Fox')], active: 0 });

		const menu = await openMenu(target);
		const rows = menu.querySelectorAll<HTMLButtonElement>('[data-row-main]');
		expect(rows).toHaveLength(2);
		expect(rows[1].textContent).toContain('Fox');
		expect(rows[1].textContent).toContain('2 images');
		rows[1].click();
		expect(handlers.onSelect).toHaveBeenCalledWith(1);
	});

	it('locks Add prompt with plain copy when the preset cannot batch', async () => {
		const { target, handlers } = render({ canBatch: false });

		const menu = await openMenu(target);
		const add = [...menu.querySelectorAll('button')].find((button) => button.textContent?.includes('Add prompt'))!;
		expect(add.disabled).toBe(true);
		add.click();
		expect(handlers.onAdd).not.toHaveBeenCalled();
		expect(menu.textContent).toContain('This preset runs one prompt at a time.');
	});

	it('strikes out the prompts that will not run when over the cap', async () => {
		const prompts = Array.from({ length: 11 }, (_, i) => entry(`prompt ${i + 1}`));
		const { target } = render({ prompts, active: 10, overCap: true, maxPrompts: 10 });

		const menu = await openMenu(target);
		const rows = menu.querySelectorAll<HTMLButtonElement>('[data-row-main]');
		expect(rows[9].textContent).not.toContain('will not run');
		expect(rows[10].textContent).toContain('will not run');
		expect(target.querySelector('[data-prompt-picker]')!.className).toContain('text-danger');
	});

	it('adds a prompt from the menu', async () => {
		const { target, handlers } = render();

		const menu = await openMenu(target);
		[...menu.querySelectorAll('button')].find((button) => button.textContent?.includes('Add prompt'))!.click();

		expect(handlers.onAdd).toHaveBeenCalledTimes(1);
	});
});
