// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync, tick, createRawSnippet } from 'svelte';
import SegmentModal from '$lib/components/modals/SegmentModal.svelte';

let instance: ReturnType<typeof mount> | null = null;

const body = createRawSnippet(() => ({
	render: () =>
		'<div><input data-testid="field" /><textarea data-testid="area"></textarea><button data-testid="inner">Inner</button></div>'
}));
const start = createRawSnippet(() => ({
	render: () => '<span><button data-testid="destroy">Remove group</button><button data-testid="tool">Shuffle now</button></span>'
}));

function open(props: Record<string, unknown> = {}) {
	const onConfirm = vi.fn();
	const onCancel = vi.fn();
	instance = mount(SegmentModal, {
		target: document.body,
		props: { tier: 'M', title: 'Save as Segment', primaryLabel: 'Save', children: body, onConfirm, onCancel, ...props }
	});
	flushSync();
	return { onConfirm, onCancel };
}

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]')!;
const alert = () => document.querySelector<HTMLElement>('[role="alertdialog"]');
const buttonStarting = (root: ParentNode, label: string) =>
	Array.from(root.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith(label))!;

async function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = window) {
	target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
	await tick();
}

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
	document.body.style.overflow = '';
});

describe('SegmentModal tiers', () => {
	it.each([
		['S', 'md:w-[420px]'],
		['M', 'md:w-[600px]'],
		['L', 'md:w-[1040px]']
	])('gives tier %s its width', (tier, width) => {
		open({ tier, role: tier === 'S' ? 'alertdialog' : 'dialog' });
		const node = document.querySelector<HTMLElement>(tier === 'S' ? '[role="alertdialog"]' : '[role="dialog"]')!;
		expect(node.className).toContain(width);
		expect(node.className).toContain('md:max-w-[calc(100vw-3rem)]');
		expect(node.className.includes('md:h-[min(760px,calc(100dvh-4rem))]')).toBe(tier === 'L');
		expect(node.className.includes('rounded-t-xl')).toBe(tier === 'S');
	});
});

describe('SegmentModal header and footer', () => {
	it('shows the mark, the title, the context line and close', () => {
		open({ mark: '{ }', tone: 'choice', context: 'Segment 1 · Subject', titleId: 'shell-title' });
		const node = dialog();
		expect(node.getAttribute('aria-labelledby')).toBe('shell-title');
		expect(node.querySelector('#shell-title')!.textContent).toBe('Save as Segment');
		expect(node.querySelector('[data-segment-modal-mark]')!.textContent?.trim()).toBe('{ }');
		expect(node.querySelector('[data-segment-modal-mark]')!.className).toContain('text-warning');
		expect(node.querySelector('[data-segment-modal-context]')!.textContent).toBe('Segment 1 · Subject');
		expect(node.querySelector('button[aria-label="Close modal"]')).not.toBeNull();
	});

	it('puts the start actions first and ends with Cancel then the primary', () => {
		open({ footerStart: start });
		const footer = dialog().querySelector<HTMLElement>('[data-segment-modal-footer]')!;
		const startArea = footer.querySelector('[data-segment-modal-start]')!;
		const end = footer.querySelector<HTMLElement>('[data-segment-modal-end]')!;
		expect(startArea.querySelector('[data-testid="destroy"]')).not.toBeNull();
		expect(startArea.compareDocumentPosition(end) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		const labels = Array.from(end.querySelectorAll('button')).map((b) => b.textContent?.replace(/\s+/g, ' ').trim());
		expect(labels).toEqual(['Cancel Esc', 'Save Enter']);
		expect(end.lastElementChild!.className).toContain('bg-accent');
	});

	it('shows the Ctrl+Enter key on editors', () => {
		open({ submit: 'ctrl', primaryLabel: 'Done' });
		const primary = buttonStarting(dialog(), 'Done');
		expect(primary.textContent?.replace(/\s+/g, '')).toBe('DoneCtrl+Enter');
	});
});

describe('SegmentModal keys', () => {
	it('confirms on Ctrl+Enter from a textarea, and not while the primary is disabled', async () => {
		const first = open();
		await press('Enter', { ctrlKey: true }, document.querySelector('[data-testid="area"]')!);
		expect(first.onConfirm).toHaveBeenCalledTimes(1);
		unmount(instance!);
		instance = null;
		document.body.innerHTML = '';
		const second = open({ primaryDisabled: true });
		await press('Enter', { ctrlKey: true });
		expect(second.onConfirm).not.toHaveBeenCalled();
	});

	it('confirms on Enter from a single-line field but never from a textarea or a button', async () => {
		const { onConfirm } = open();
		await press('Enter', {}, document.querySelector('[data-testid="area"]')!);
		await press('Enter', {}, document.querySelector('[data-testid="inner"]')!);
		expect(onConfirm).not.toHaveBeenCalled();
		await press('Enter', {}, document.querySelector('[data-testid="field"]')!);
		expect(onConfirm).toHaveBeenCalledTimes(1);
	});

	it.each([
		['an input that reports an open popup', () => {
			const input = document.createElement('input');
			input.setAttribute('aria-expanded', 'true');
			return { target: input, init: {} };
		}],
		['an option row', () => {
			const option = document.createElement('div');
			option.setAttribute('role', 'option');
			return { target: option, init: {} };
		}],
		['a plain input with Shift held', () => ({ target: document.createElement('input'), init: { shiftKey: true } })],
		['a plain input with Alt held', () => ({ target: document.createElement('input'), init: { altKey: true } })]
	])('does not confirm on Enter from %s', async (_name, build) => {
		const { onConfirm } = open();
		const { target, init } = build();
		dialog().appendChild(target);
		await press('Enter', init, target);
		expect(onConfirm).not.toHaveBeenCalled();
	});

	it('ignores plain Enter in editors', async () => {
		const { onConfirm } = open({ submit: 'ctrl' });
		await press('Enter', {}, document.querySelector('[data-testid="field"]')!);
		expect(onConfirm).not.toHaveBeenCalled();
	});

	it('confirms again after the confirm handler keeps the dialog open', async () => {
		const onConfirm = vi.fn(() => false);
		open({ onConfirm });
		await press('Enter', { ctrlKey: true });
		await press('Enter', { ctrlKey: true });
		expect(onConfirm).toHaveBeenCalledTimes(2);
	});

	it('closes at once on Esc when nothing changed', async () => {
		const { onCancel } = open();
		await press('Escape');
		expect(onCancel).toHaveBeenCalledTimes(1);
		expect(alert()).toBeNull();
	});

	it('asks before discarding edits and Ctrl+Enter on the question discards', async () => {
		const { onCancel, onConfirm } = open({ dirty: true, discardMessage: 'Your edits to this condition are lost.' });
		await press('Escape');
		expect(onCancel).not.toHaveBeenCalled();
		expect(alert()!.textContent).toContain('Discard changes?');
		expect(alert()!.textContent).toContain('Your edits to this condition are lost.');
		await new Promise((resolve) => requestAnimationFrame(resolve));
		expect(document.activeElement?.textContent?.trim().startsWith('Keep editing')).toBe(true);
		await press('Enter', { ctrlKey: true }, alert()!);
		expect(onConfirm).not.toHaveBeenCalled();
		expect(onCancel).toHaveBeenCalledTimes(1);
	});

	it('goes back to the editor from Keep editing and discards from Discard', async () => {
		const { onCancel } = open({ dirty: true });
		buttonStarting(dialog(), 'Cancel').click();
		await tick();
		await press('Escape');
		expect(alert()).toBeNull();
		expect(onCancel).not.toHaveBeenCalled();
		buttonStarting(dialog(), 'Cancel').click();
		await tick();
		buttonStarting(alert()!, 'Keep editing').click();
		await tick();
		expect(alert()).toBeNull();
		buttonStarting(dialog(), 'Cancel').click();
		await tick();
		buttonStarting(alert()!, 'Discard').click();
		await tick();
		expect(onCancel).toHaveBeenCalledTimes(1);
	});
});
