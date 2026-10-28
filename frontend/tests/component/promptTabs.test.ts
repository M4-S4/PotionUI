import { describe, it, expect, vi } from 'vitest';
import { tick } from 'svelte';
import PromptTabs from '$lib/components/PromptTabs.svelte';
import { mountComponent } from './helpers/mount';
import { settle } from './helpers/settle';

function entry(name?: string) {
	return { prompt: '', negativePrompt: '', promptSegments: [], negativePromptSegments: [], ...(name ? { name } : {}) };
}

function render(props: Record<string, unknown> = {}) {
	let view: ReturnType<typeof mountComponent>;
	const handlers = {
		onSelect: vi.fn((index: number) => view.rerender({ active: index })),
		onAdd: vi.fn(),
		onDuplicate: vi.fn(),
		onDelete: vi.fn(),
		onRename: vi.fn()
	};
	view = mountComponent(PromptTabs as never, {
		prompts: [entry(), entry('Rainy street'), entry()],
		active: 0,
		perPrompt: 2,
		canBatch: true,
		overCap: false,
		maxPrompts: 10,
		...handlers,
		...props
	});
	return { target: view.target, handlers, view };
}

const tabs = (target: HTMLElement) => [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
const labels = (target: HTMLElement) => tabs(target).map((tab) => tab.textContent!.trim());
const selectedIndex = (target: HTMLElement) => tabs(target).findIndex((tab) => tab.getAttribute('aria-selected') === 'true');
const caret = (target: HTMLElement) => target.querySelector<HTMLButtonElement>('button[aria-label="Prompt actions"]')!;
const menu = () => document.body.querySelector<HTMLElement>('[role="menu"]');
const menuLabels = () =>
	[...document.body.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitem"]')].map((item) =>
		item.querySelector('span span')?.textContent?.trim() ?? item.textContent!.trim()
	);
const menuItem = (label: string) =>
	[...document.body.querySelectorAll<HTMLButtonElement>('[role="menu"] [role="menuitem"]')].find((item) =>
		item.textContent!.trim().startsWith(label)
	)!;
const keydown = (el: Element, key: string) =>
	el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));

function stubRect(el: Element, rect: { left: number; bottom: number }) {
	el.getBoundingClientRect = () =>
		({
			x: rect.left,
			y: rect.bottom - 30,
			left: rect.left,
			right: rect.left + 80,
			top: rect.bottom - 30,
			bottom: rect.bottom,
			width: 80,
			height: 30,
			toJSON: () => ({})
		}) as DOMRect;
}

function menuOrigin() {
	const style = menu()!.getAttribute('style') ?? '';
	return {
		left: parseFloat(style.match(/left:\s*([\d.]+)px/)![1]),
		top: parseFloat(style.match(/top:\s*([\d.]+)px/)![1])
	};
}

async function showTooltip(button: Element) {
	button.parentElement!.dispatchEvent(new MouseEvent('mouseenter'));
	await new Promise((r) => setTimeout(r, 260));
	await settle();
	return document.body.querySelector('[role="tooltip"]')?.textContent?.trim();
}

async function openMenu(target: HTMLElement) {
	caret(target).click();
	await settle();
}

describe('PromptTabs strip', () => {
	it('renders one tab per prompt, named by the prompt or by its position', () => {
		const { target } = render();

		expect(labels(target)).toEqual(['Prompt 1', 'Rainy street', 'Prompt 3']);
		expect(target.querySelector('[role="tablist"]')!.getAttribute('aria-label')).toBe('Prompts');
	});

	it('names a lone prompt plainly and honours its name', () => {
		expect(labels(render({ prompts: [entry()] }).target)).toEqual(['Prompt']);
		expect(labels(render({ prompts: [entry('Night market')] }).target)).toEqual(['Night market']);
	});

	it('falls back to the position when the name is only whitespace', () => {
		const { target } = render({ prompts: [entry(), { ...entry(), name: '   ' }] });

		expect(labels(target)).toEqual(['Prompt 1', 'Prompt 2']);
	});

	it('marks the active tab selected and makes only it tabbable', () => {
		const { target } = render({ active: 1 });

		expect(selectedIndex(target)).toBe(1);
		expect(tabs(target).map((tab) => tab.tabIndex)).toEqual([-1, 0, -1]);
	});

	it('selects a tab on click but ignores a click on the active one', () => {
		const { target, handlers } = render({ active: 1 });

		tabs(target)[2].click();
		expect(handlers.onSelect).toHaveBeenCalledWith(2);
		handlers.onSelect.mockClear();
		tabs(target)[2].click();
		expect(handlers.onSelect).not.toHaveBeenCalled();
	});

	it('moves between tabs with the arrow keys, wrapping at both ends', async () => {
		const { target, handlers } = render();

		keydown(tabs(target)[0], 'ArrowLeft');
		expect(handlers.onSelect).toHaveBeenLastCalledWith(2);
		await tick();
		keydown(tabs(target)[2], 'ArrowRight');
		expect(handlers.onSelect).toHaveBeenLastCalledWith(0);
		await tick();
		keydown(tabs(target)[0], 'End');
		expect(handlers.onSelect).toHaveBeenLastCalledWith(2);
		await tick();
		keydown(tabs(target)[2], 'Home');
		expect(handlers.onSelect).toHaveBeenLastCalledWith(0);
	});

	it('moves focus to the tab it selects', async () => {
		const { target } = render();

		keydown(tabs(target)[0], 'ArrowRight');
		await settle();

		expect(document.activeElement).toBe(tabs(target)[1]);
	});

	it('offers a list of every prompt only once there are two or more', () => {
		expect(render({ prompts: [entry()] }).target.querySelector('[aria-label="All prompts"]')).toBeNull();
		expect(render().target.querySelector('[aria-label="All prompts"][data-prompt-picker]')).not.toBeNull();
	});

	it('adds a prompt from the plus button', () => {
		const { target, handlers } = render();

		target.querySelector<HTMLButtonElement>('button[aria-label="Add prompt"]')!.click();

		expect(handlers.onAdd).toHaveBeenCalledTimes(1);
	});

	it('disables the plus button when the preset cannot batch or the cap is reached', () => {
		const locked = render({ canBatch: false }).target.querySelector<HTMLButtonElement>('button[aria-label="Add prompt"]')!;
		expect(locked.disabled).toBe(true);

		const full = render({ maxPrompts: 3 }).target.querySelector<HTMLButtonElement>('button[aria-label="Add prompt"]')!;
		expect(full.disabled).toBe(true);
	});

	it('explains the plus button on hover: plain while there is room, the lock for a preset that cannot batch, the cap when full', async () => {
		const tipFor = async (props: Record<string, unknown>) => {
			const { target, view } = render(props);
			const text = await showTooltip(target.querySelector('button[aria-label="Add prompt"]')!);
			view.destroy();
			return text;
		};

		expect(await tipFor({})).toBe('Add prompt');
		expect(await tipFor({ canBatch: false })).toBe('This preset runs one prompt at a time.');
		expect(await tipFor({ maxPrompts: 3, perPrompt: 2 })).toBe(
			'A run can make up to 20 images. At 2 images each, that is 3 prompts.'
		);
	});

	it('strikes out the tabs that will not run and says so to screen readers', () => {
		const { target } = render({ prompts: [entry(), entry(), entry()], overCap: true, maxPrompts: 2 });

		const [first, , third] = tabs(target);
		expect(first.className).not.toContain('line-through');
		expect(third.className).toContain('line-through');
		expect(third.textContent).toContain('will not run');
		expect(first.textContent).not.toContain('will not run');
	});

	it('dims the kept prompts of a preset that runs only the first', () => {
		const { target } = render({ canBatch: false });

		const wrappers = [...target.querySelectorAll<HTMLElement>('[data-tab]')];
		expect(wrappers[0].className).not.toContain('opacity-50');
		expect(wrappers[1].className).toContain('opacity-50');
	});
});

describe('PromptTabs menu', () => {
	it('opens from the active tab caret with rename, duplicate and delete', async () => {
		const { target } = render({ active: 1 });

		await openMenu(target);

		expect(menu()!.getAttribute('aria-label')).toBe('Actions for Rainy street');
		expect(menuLabels()).toEqual(['Rename', 'Duplicate', 'Delete']);
		expect(caret(target).getAttribute('aria-expanded')).toBe('true');
	});

	it('only shows the caret on the active tab', () => {
		const { target } = render({ active: 2 });

		expect(target.querySelectorAll('button[aria-label="Prompt actions"]')).toHaveLength(1);
		expect(target.querySelector('[data-tab][data-active="true"] button[aria-label="Prompt actions"]')).not.toBeNull();
	});

	it('hides delete for a single prompt', async () => {
		const { target } = render({ prompts: [entry()] });

		await openMenu(target);

		expect(menuLabels()).toEqual(['Rename', 'Duplicate']);
	});

	it('duplicates the active prompt and closes', async () => {
		const { target, handlers } = render({ active: 1 });

		await openMenu(target);
		menuItem('Duplicate').click();
		await settle();

		expect(handlers.onDuplicate).toHaveBeenCalledWith(1);
		expect(menu()).toBeNull();
	});

	it('deletes the active prompt', async () => {
		const { target, handlers } = render({ active: 2 });

		await openMenu(target);
		menuItem('Delete').click();

		expect(handlers.onDelete).toHaveBeenCalledWith(2);
	});

	it('disables duplicate and says why when the preset cannot batch', async () => {
		const { target, handlers } = render({ canBatch: false });

		await openMenu(target);
		const duplicate = menuItem('Duplicate');
		expect(duplicate.disabled).toBe(true);
		expect(duplicate.textContent).toContain('This preset runs one prompt at a time.');
		duplicate.click();
		expect(handlers.onDuplicate).not.toHaveBeenCalled();
	});

	it('disables duplicate at the prompt cap and quotes the cap', async () => {
		const { target, handlers } = render({ maxPrompts: 3 });

		await openMenu(target);
		const duplicate = menuItem('Duplicate');
		expect(duplicate.disabled).toBe(true);
		expect(duplicate.textContent).toContain('that is 3 prompts');
		duplicate.click();
		expect(handlers.onDuplicate).not.toHaveBeenCalled();
	});

	it('leaves duplicate on, with no explanation, while there is room', async () => {
		const { target } = render({ maxPrompts: 10 });

		await openMenu(target);
		const duplicate = menuItem('Duplicate');
		expect(duplicate.disabled).toBe(false);
		expect(duplicate.textContent!.trim()).toBe('Duplicate');
	});

	it('closes on Escape', async () => {
		const { target } = render();

		await openMenu(target);
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await settle();

		expect(menu()).toBeNull();
	});

	it('opens from a right click on any tab and selects that tab first', async () => {
		const { target, handlers } = render({ active: 0 });

		tabs(target)[2].dispatchEvent(
			new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 30 })
		);
		await settle();

		expect(handlers.onSelect).toHaveBeenCalledWith(2);
		expect(selectedIndex(target)).toBe(2);
		expect(menu()!.getAttribute('aria-label')).toBe('Actions for Prompt 3');
	});

	it('opens a right click at the pointer, not at the tab', async () => {
		const { target } = render({ active: 0 });
		stubRect(target.querySelectorAll('[data-tab]')[2], { left: 10, bottom: 40 });

		tabs(target)[2].dispatchEvent(
			new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 120 })
		);
		await settle();

		expect(menuOrigin()).toEqual({ left: 300, top: 124 });
	});

	it('opens from the context-menu key under the left edge of the tab, since it reports no pointer', async () => {
		const { target } = render({ active: 1 });
		stubRect(target.querySelectorAll('[data-tab]')[1], { left: 200, bottom: 90 });

		tabs(target)[1].dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 0, clientY: 0 }));
		await settle();

		expect(menuLabels()).toContain('Rename');
		expect(menuOrigin()).toEqual({ left: 200, top: 94 });
	});

	it('does not swallow the native menu of the rename field', async () => {
		const { target } = render({ active: 1 });

		await openMenu(target);
		menuItem('Rename').click();
		await settle();
		const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
		target.querySelector('input[data-rename]')!.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(false);
	});
});

describe('PromptTabs rename', () => {
	async function startRename(target: HTMLElement) {
		await openMenu(target);
		menuItem('Rename').click();
		await settle();
		return target.querySelector<HTMLInputElement>('input[data-rename]')!;
	}

	function type(input: HTMLInputElement, value: string) {
		input.value = value;
		input.dispatchEvent(new Event('input', { bubbles: true }));
	}

	it('swaps the tab for a focused input holding the current name', async () => {
		const { target } = render({ active: 1 });

		const input = await startRename(target);

		expect(input.value).toBe('Rainy street');
		expect(document.activeElement).toBe(input);
		expect(input.getAttribute('aria-label')).toBe('Prompt name');
		expect(labels(target)).toEqual(['Prompt 1', 'Prompt 3']);
	});

	it('starts empty for a prompt that has no name yet', async () => {
		const { target } = render({ active: 0 });

		expect((await startRename(target)).value).toBe('');
	});

	it('commits on Enter and returns focus to the tab', async () => {
		const { target, handlers } = render({ active: 1 });

		const input = await startRename(target);
		type(input, 'Neon alley');
		keydown(input, 'Enter');
		input.dispatchEvent(new FocusEvent('blur'));
		await settle();

		expect(handlers.onRename).toHaveBeenCalledTimes(1);
		expect(handlers.onRename).toHaveBeenCalledWith(1, 'Neon alley');
		expect(target.querySelector('input[data-rename]')).toBeNull();
		expect(document.activeElement).toBe(tabs(target)[1]);
	});

	it('cancels on Escape without renaming', async () => {
		const { target, handlers } = render({ active: 1 });

		const input = await startRename(target);
		type(input, 'Neon alley');
		keydown(input, 'Escape');
		input.dispatchEvent(new FocusEvent('blur'));
		await settle();

		expect(handlers.onRename).not.toHaveBeenCalled();
		expect(target.querySelector('input[data-rename]')).toBeNull();
		expect(labels(target)).toEqual(['Prompt 1', 'Rainy street', 'Prompt 3']);
	});

	it('commits when the field loses focus, exactly once', async () => {
		const { target, handlers } = render({ active: 1 });

		const input = await startRename(target);
		type(input, 'Blurred');
		input.dispatchEvent(new FocusEvent('blur'));
		await settle();

		expect(handlers.onRename).toHaveBeenCalledTimes(1);
		expect(handlers.onRename).toHaveBeenCalledWith(1, 'Blurred');
	});

	it('starts from the keyboard with F2', async () => {
		const { target } = render({ active: 1 });

		keydown(tabs(target)[1], 'F2');
		await settle();

		expect(target.querySelector('input[data-rename]')).not.toBeNull();
	});
});
