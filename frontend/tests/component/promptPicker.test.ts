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
	it('is a labelled list trigger that opens the prompts dialog', async () => {
		const { target } = render({ prompts: [entry('one'), entry('two')], active: 1 });

		const trigger = target.querySelector<HTMLButtonElement>('[data-prompt-picker]')!;
		expect(trigger.getAttribute('aria-label')).toBe('All prompts');
		expect(trigger.getAttribute('aria-haspopup')).toBe('dialog');
		expect(trigger.getAttribute('aria-expanded')).toBe('false');
		expect(trigger.textContent!.trim()).toBe('');

		const menu = await openMenu(target);
		expect(trigger.getAttribute('aria-expanded')).toBe('true');
		expect(menu.getAttribute('aria-label')).toBe('Prompts');
	});

	it('opens a 24rem list with 14px rows, add button and rename field', async () => {
		const { target } = render({ prompts: [entry('one'), entry('two')], active: 0 });

		const menu = await openMenu(target);
		expect(menu.getAttribute('style')).toContain('width: 384px');
		const rows = menu.querySelectorAll<HTMLElement>('[data-row-main]');
		for (const row of rows) {
			expect(row.querySelector('span.truncate')!.className).toContain('text-md');
			expect(row.className).toContain('px-3');
			expect(row.className).toContain('py-2');
		}
		const add = [...menu.querySelectorAll('button')].find((button) => button.textContent?.includes('Add prompt'))!;
		expect(add.className).toContain('text-md');

		menu.querySelector<HTMLButtonElement>('[aria-label="Rename prompt"]')!.click();
		flushSync();
		await tick();
		expect(menu.querySelector('input[data-rename]')!.className).toContain('text-md');
	});

	it('counts a single prompt in the list header and offers no delete', async () => {
		const { target } = render();

		const menu = await openMenu(target);
		expect(menu.textContent).toContain('1 prompt');
		expect(menu.querySelector('[aria-label="Delete prompt"]')).toBeNull();
		expect(menu.textContent).toContain('Add prompt starts empty');
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

	it('marks the current prompt and offers rename, duplicate and delete on its row', async () => {
		const { target, handlers } = render({ prompts: [entry('one'), entry('two')], active: 1 });

		const menu = await openMenu(target);
		const rows = menu.querySelectorAll<HTMLElement>('li');
		expect(rows[1].querySelector('[data-row-main]')!.getAttribute('aria-current')).toBe('true');
		expect(rows[0].querySelector('[data-row-main]')!.getAttribute('aria-current')).toBeNull();
		rows[0].querySelector<HTMLButtonElement>('[aria-label="Duplicate prompt"]')!.click();
		expect(handlers.onDuplicate).toHaveBeenCalledWith(0);
		rows[1].querySelector<HTMLButtonElement>('[aria-label="Delete prompt"]')!.click();
		expect(handlers.onDelete).toHaveBeenCalledWith(1);
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

	it('strikes out the prompts that will not run and tints the trigger when over the cap', async () => {
		const prompts = Array.from({ length: 11 }, (_, i) => entry(`prompt ${i + 1}`));
		const { target } = render({ prompts, active: 10, overCap: true, maxPrompts: 10 });

		const menu = await openMenu(target);
		const rows = menu.querySelectorAll<HTMLButtonElement>('[data-row-main]');
		expect(rows[9].textContent).not.toContain('will not run');
		expect(rows[10].textContent).toContain('will not run');
		expect(target.querySelector('[data-prompt-picker]')!.className).toContain('text-danger');
	});

	it('leaves the trigger untinted while every prompt will run', () => {
		const { target } = render({ prompts: [entry('one'), entry('two')], overCap: false });

		expect(target.querySelector('[data-prompt-picker]')!.className).not.toContain('text-danger');
	});

	it('adds a prompt from the menu', async () => {
		const { target, handlers } = render();

		const menu = await openMenu(target);
		[...menu.querySelectorAll('button')].find((button) => button.textContent?.includes('Add prompt'))!.click();

		expect(handlers.onAdd).toHaveBeenCalledTimes(1);
	});
});
