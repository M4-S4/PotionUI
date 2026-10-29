// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ToolsMenu from '$lib/components/tools/ToolsMenu.svelte';
import type { MediaTool, MediaToolGroup } from '$lib/tools/tools';

function tool(id: string, extra: Partial<MediaTool> = {}): MediaTool {
	return {
		id,
		label: id.toUpperCase(),
		icon: 'extension',
		category: 'edit',
		source: 'core',
		scopes: ['history', 'field'],
		applies: () => ({ enabled: true }),
		...extra
	};
}

function groups(): MediaToolGroup[] {
	return [
		{
			category: { id: 'edit', label: 'Edit', order: 10 },
			tools: [
				{ tool: tool('crop', { shortcut: 'C', description: 'Crop it' }), availability: { enabled: true } },
				{ tool: tool('clear'), availability: { enabled: false, reason: 'No mask' } }
			]
		}
	];
}

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

function render(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(ToolsMenu, {
		target,
		props: {
			open: true,
			scope: 'history',
			groups: groups(),
			onToggle: vi.fn(),
			onClose: vi.fn(),
			onPick: vi.fn(),
			...props
		}
	});
	flushSync();
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return target;
}

function menu(): HTMLElement {
	return document.body.querySelector('[data-tools-menu]') as HTMLElement;
}

function anchorAt(top: number, bottom: number) {
	vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
		const isTrigger = this.hasAttribute('data-tools-menu-scope') && !this.hasAttribute('data-tools-menu');
		return {
			top: isTrigger ? top : 0,
			bottom: isTrigger ? bottom : 0,
			left: 100,
			right: 180,
			width: 80,
			height: isTrigger ? bottom - top : 0,
			x: 100,
			y: top,
			toJSON: () => ({})
		} as DOMRect;
	});
}

describe('ToolsMenu', () => {
	it('draws no menu while closed and a trigger that reports its state', () => {
		const target = render({ open: false });
		expect(menu()).toBeNull();
		expect(target.querySelector('[data-tools-trigger]')!.getAttribute('aria-expanded')).toBe('false');
	});

	it('lists each group under its category with the tools in it', () => {
		render({});
		const group = menu().querySelector('[role="group"]')!;
		expect(group.getAttribute('aria-label')).toBe('Edit');
		expect(Array.from(menu().querySelectorAll('[data-tool]')).map((el) => el.getAttribute('data-tool'))).toEqual([
			'crop',
			'clear'
		]);
	});

	it('closes and reports the pick for an available tool', () => {
		const props = { onClose: vi.fn(), onPick: vi.fn() };
		render(props);
		(menu().querySelector('[data-tool="crop"]') as HTMLButtonElement).click();
		expect(props.onClose).toHaveBeenCalledTimes(1);
		expect(props.onPick).toHaveBeenCalledWith(expect.objectContaining({ id: 'crop' }));
	});

	it('keeps a greyed tool focusable and does nothing when it is pressed', () => {
		const props = { onClose: vi.fn(), onPick: vi.fn() };
		render(props);
		const clear = menu().querySelector('[data-tool="clear"]') as HTMLButtonElement;
		expect(clear.getAttribute('aria-disabled')).toBe('true');
		expect(clear.disabled).toBe(false);
		clear.click();
		expect(props.onPick).not.toHaveBeenCalled();
		expect(props.onClose).not.toHaveBeenCalled();
	});

	it('shows the shortcut of a field tool and the reason a greyed one is unavailable', () => {
		render({ scope: 'field' });
		expect(menu().querySelector('[data-tool="crop"] kbd')?.textContent?.trim()).toBe('C');
		expect(menu().querySelector('[data-tool="clear"]')!.textContent).toContain('No mask');
	});

	it('never puts a native title on a menu item', () => {
		render({});
		expect(menu().querySelectorAll('[title]')).toHaveLength(0);
	});

	it('closes on Escape and on a pointer press outside it', () => {
		const props = { onClose: vi.fn() };
		render(props);
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
		expect(props.onClose).toHaveBeenCalledTimes(1);
		document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		expect(props.onClose).toHaveBeenCalledTimes(2);
	});

	it('moves focus through the items with the arrow keys', () => {
		render({});
		const [first, second] = Array.from(menu().querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));
		first.focus();
		document.activeElement!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
		expect(document.activeElement).toBe(second);
		document.activeElement!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
		expect(document.activeElement).toBe(first);
	});

	it('ignores a tool shortcut typed into a text field and runs it from the page body', () => {
		const onPick = vi.fn();
		render({ onPick });
		const field = document.body.appendChild(document.createElement('textarea'));
		field.focus();
		const typed = new KeyboardEvent('keydown', { key: 'c', bubbles: true, cancelable: true });
		field.dispatchEvent(typed);
		expect(typed.defaultPrevented).toBe(false);
		expect(onPick).not.toHaveBeenCalled();

		field.blur();
		document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'c', bubbles: true, cancelable: true }));
		expect(onPick).toHaveBeenCalledWith(expect.objectContaining({ id: 'crop' }));
	});

	it('says there is nothing to offer when no group applies', () => {
		render({ groups: [] });
		expect(menu().textContent).toContain('No tools for this selection');
	});

	it('opens upward by default, as it does above the History bar', () => {
		anchorAt(400, 430);
		render({});
		expect(menu().style.bottom).toBe(`${window.innerHeight - 400 + 4}px`);
		expect(menu().style.top).toBe('');
	});

	it('opens downward when asked and there is room', () => {
		anchorAt(100, 130);
		render({ placement: 'down' });
		expect(menu().style.top).toBe('134px');
		expect(menu().style.bottom).toBe('');
	});

	it('flips upward when it was asked to open downward and the room is below too small', () => {
		anchorAt(700, 730);
		render({ placement: 'down' });
		expect(menu().style.bottom).not.toBe('');
	});

	it('shows an icon-only trigger with an accessible name when compact', () => {
		const target = render({ open: false, compact: true });
		const trigger = target.querySelector('[data-tools-trigger]')!;
		expect(trigger.getAttribute('aria-label')).toBe('Tools');
		expect(trigger.textContent?.trim()).toBe('');
	});
});

describe('ToolsMenu second line', () => {
	function noted(): MediaToolGroup[] {
		return [
			{
				category: { id: 'edit', label: 'Edit', order: -20 },
				tools: [
					{ tool: tool('frame', { note: 'Saves one image' }), availability: { enabled: true } },
					{ tool: tool('plain'), availability: { enabled: true } },
					{ tool: tool('concat'), availability: { enabled: false, reason: 'Select 2 to 8 items' } },
					{ tool: tool('upscaler.upscale', { source: 'plugin', note: 'Upscaler' }), availability: { enabled: true } }
				]
			}
		];
	}

	const sub = (id: string) => menu().querySelector(`[data-tool="${id}"] [data-tool-sub]`);

	it('says what an enabled tool saves under its label', () => {
		render({ groups: noted() });
		expect(sub('frame')!.textContent).toBe('Saves one image');
	});

	it('gives a plain tool one line only', () => {
		render({ groups: noted() });
		expect(sub('plain')).toBeNull();
	});

	it('puts the reason a tool is unavailable under its label on a page menu', () => {
		render({ groups: noted() });
		expect(sub('concat')!.textContent).toBe('Select 2 to 8 items');
		expect(menu().querySelector('[data-tool="concat"]')!.getAttribute('aria-disabled')).toBe('true');
	});

	it('does the same on the card menu, which used to show it at the end of the row', () => {
		render({ groups: noted(), variant: 'entry', scope: 'library' });
		expect(sub('concat')!.textContent).toBe('Select 2 to 8 items');
	});

	it('keeps the reason at the end of the row for a field menu', () => {
		render({ groups: noted(), scope: 'field' });
		expect(sub('concat')).toBeNull();
		expect(menu().querySelector('[data-tool="concat"]')!.textContent).toContain('Select 2 to 8 items');
	});

	it('names the plugin on the second line instead of a Plugin tag', () => {
		render({ groups: noted(), variant: 'entry', scope: 'history' });
		expect(sub('upscaler.upscale')!.textContent).toBe('Upscaler');
		expect(menu().querySelector('[data-tool="upscaler.upscale"]')!.textContent?.toLowerCase()).not.toContain('plugin');
	});

	it('draws the second line at 12px or larger in the subtle text colour', () => {
		render({ groups: noted() });
		const line = sub('frame')!;
		expect(line.className).toContain('text-[12px]');
		expect(line.className).toContain('text-fg-subtle');
	});

	it('truncates a long label and a long second line rather than wrapping the row', () => {
		render({ groups: noted() });
		expect(menu().querySelector('[data-tool="frame"] span.truncate')).toBeTruthy();
		expect(sub('frame')!.className).toContain('truncate');
	});

	it('shows the second line in the phone sheet too', () => {
		render({ groups: noted(), sheet: true, title: 'clip.mp4' });
		expect(sub('frame')!.textContent).toBe('Saves one image');
	});
});
