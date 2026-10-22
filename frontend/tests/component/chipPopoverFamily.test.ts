// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';

const { default: ChoiceGroupChip } = await import('$lib/components/ChoiceGroupChip.svelte');
const { default: VariableUsageChip } = await import('$lib/components/VariableUsageChip.svelte');

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | undefined;

afterEach(() => {
	if (component) unmount(component);
	component = undefined;
	target?.remove();
	document.body.innerHTML = '';
	Object.defineProperty(window, 'innerWidth', { value: 1024, configurable: true });
	vi.clearAllMocks();
});

function open(Component: unknown, props: Record<string, unknown>) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(Component as never, { target, props: { variant: 'segment-composer', ...props } });
	flushSync();
	target.querySelector<HTMLButtonElement>('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
	flushSync();
	return document.querySelector<HTMLElement>('[role="dialog"]')!;
}

const footerLabels = (popover: HTMLElement) =>
	Array.from(popover.querySelectorAll<HTMLButtonElement>('.popover-actions > button')).map((b) => b.textContent?.trim());

describe('composer chip popovers', () => {
	it('lead the choice group footer with Remove group and end it with Done, which closes', () => {
		const popover = open(ChoiceGroupChip, { raw: '{a|b}' });
		expect(footerLabels(popover)).toEqual(['Remove group', 'Done']);
		popover.querySelector<HTMLButtonElement>('.popover-done')!.click();
		flushSync();
		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});

	it('put Edit in Variables in the usage footer between Remove usage and Done', () => {
		const onOpenManager = vi.fn();
		const popover = open(VariableUsageChip, {
			name: 'mood',
			definition: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] },
			onOpenManager
		});
		expect(footerLabels(popover)).toEqual(['Remove usage', 'Edit in Variables…', 'Done']);
		expect(popover.querySelector('.popover-body')!.textContent).not.toContain('Edit in Variables');
		popover.querySelectorAll<HTMLButtonElement>('.popover-actions > button')[1].click();
		flushSync();
		expect(onOpenManager).toHaveBeenCalledTimes(1);
	});

	it('become a bottom sheet below 768px, the same breakpoint as the modals', () => {
		Object.defineProperty(window, 'innerWidth', { value: 700, configurable: true });
		const popover = open(ChoiceGroupChip, { raw: '{a|b}' });
		expect(popover.style.left).toBe('8px');
		expect(popover.style.bottom).toBe('8px');
		document.body.innerHTML = '';
		unmount(component!);
		Object.defineProperty(window, 'innerWidth', { value: 768, configurable: true });
		const desktop = open(ChoiceGroupChip, { raw: '{a|b}' });
		expect(desktop.style.width).toBe('420px');
	});
});
