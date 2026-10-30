import { describe, it, expect } from 'vitest';
import { flushSync } from 'svelte';
import { mountComponent } from './helpers/mount';

const { default: PreviewPane } = await import('$lib/media-tools/PreviewPane.svelte');

function render(overrides: Record<string, unknown> = {}) {
	return mountComponent(PreviewPane, {
		beforeUrl: '/before.png',
		beforeKind: 'image',
		afterUrl: '/after.png',
		...overrides
	});
}

const split = (target: HTMLElement) => target.querySelector('[data-split-input]') as HTMLInputElement;
const line = (target: HTMLElement) => target.querySelector('[data-split-line]') as HTMLElement;
const handle = (target: HTMLElement) => target.querySelector('[data-split-handle]') as HTMLElement;

describe('PreviewPane split control', () => {
	it('is a labelled range input that can take focus', () => {
		const { target } = render();
		expect(split(target).getAttribute('aria-label')).toBe('Before and after split');
		expect(split(target).type).toBe('range');
		split(target).focus();
		expect(document.activeElement).toBe(split(target));
	});

	it('shows focus on the line and the handle, which the invisible input cannot show itself', () => {
		const { target } = render();

		expect(split(target).className).toContain('peer');
		expect(split(target).className).toContain('opacity-0');
		expect(line(target).className).toContain('peer-focus-visible:bg-signal');
		expect(line(target).className).toContain('peer-focus-visible:w-1');
		expect(handle(target).className).toContain('peer-focus-visible:ring-2');
		expect(handle(target).className).toContain('peer-focus-visible:ring-signal');
	});

	it('puts the input before the line and the handle, so the peer styles can reach them', () => {
		const { target } = render();
		const input = split(target);
		expect(input.compareDocumentPosition(line(target)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(input.compareDocumentPosition(handle(target)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(input.parentElement).toBe(line(target).parentElement);
		expect(input.parentElement).toBe(handle(target).parentElement);
	});

	it('uses tokens for the focus colours and nothing hard-coded', () => {
		const { target } = render();
		const classes = `${line(target).className} ${handle(target).className}`;
		expect(classes).not.toMatch(/#[0-9a-fA-F]{3,6}/);
		expect(classes).not.toMatch(/\b(zinc|gray|blue)-\d/);
		expect(classes).not.toContain('outline-none');
	});

	it('moves the line, the handle and the cut together with the value', () => {
		const { target } = render();
		const input = split(target);

		input.value = '30';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		flushSync();

		expect(line(target).getAttribute('style')).toContain('left: 30%');
		expect(handle(target).getAttribute('style')).toContain('left: 30%');
		expect(target.querySelector('.checkerboard')?.getAttribute('style')).toContain('inset(0 0 0 30%)');
	});

	it('labels both sides and shows the image it was given on the right', () => {
		const { target } = render();
		expect(target.textContent).toContain('Before');
		expect(target.textContent).toContain('After');
		expect(target.querySelector('img[alt="After"]')?.getAttribute('src')).toBe('/after.png');
	});

	it('says so when the preview could not be rendered, and shows the busy state', () => {
		const { target } = render({ afterUrl: null, error: 'The preview could not be rendered', loading: true });
		expect(target.textContent).toContain('The preview could not be rendered');
		expect(target.querySelector('[data-preview-loading]')).not.toBeNull();
	});
});
