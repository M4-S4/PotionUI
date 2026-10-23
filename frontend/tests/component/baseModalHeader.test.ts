import { describe, it, expect, afterEach } from 'vitest';
import { createClassComponent } from 'svelte/legacy';
import { mount, unmount, flushSync, createRawSnippet } from 'svelte';
import BaseModal from '$lib/components/modals/BaseModal.svelte';
import SegmentModal from '$lib/components/modals/SegmentModal.svelte';

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
	document.body.style.overflow = '';
});

function openBase(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: BaseModal as never,
		target,
		props: { isOpen: true, ...props }
	});
	cleanup = () => component.$destroy();
}

function openSegment() {
	const children = createRawSnippet(() => ({ render: () => '<div>Body</div>' }));
	const instance = mount(SegmentModal, {
		target: document.body,
		props: {
			tier: 'M',
			title: 'Save as Segment',
			context: 'Segment 1 · Subject',
			primaryLabel: 'Save',
			children,
			onConfirm: () => {},
			onCancel: () => {}
		}
	});
	flushSync();
	cleanup = () => unmount(instance);
}

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]')!;
const header = () => dialog().firstElementChild as HTMLElement;
const classes = (el: Element) => el.className.split(/\s+/);

describe('modal header height', () => {
	it('is 56px on desktop and 52px on a phone for a title-only modal', () => {
		openBase({ title: 'Details' });
		expect(classes(header())).toEqual(expect.arrayContaining(['min-h-[52px]', 'md:min-h-14']));
		expect(classes(header())).toEqual(expect.arrayContaining(['md:py-[9px]', 'md:pl-5', 'md:pr-2.5', 'py-[7px]', 'pl-4', 'pr-2']));
		expect(classes(header())).not.toEqual(expect.arrayContaining(['md:py-4']));
		expect(classes(header())).not.toEqual(expect.arrayContaining(['md:px-6']));
	});

	it('gives the segment modal the same header, with a 36px title and context block that fits it', () => {
		openSegment();
		expect(classes(header())).toEqual(expect.arrayContaining(['min-h-[52px]', 'md:min-h-14', 'md:py-[9px]', 'py-[7px]']));
		expect(header().querySelector('[data-segment-modal-context]')!.className).toContain('text-sm');
		expect(header().querySelector('strong')!.className).toContain('text-md');
	});

	it('keeps the close control at 32px so it fits inside the header', () => {
		openBase({ title: 'Details' });
		const close = header().querySelector<HTMLElement>('button[aria-label="Close modal"]')!;
		expect(classes(close)).toEqual(expect.arrayContaining(['min-w-8', 'min-h-8']));
		expect(classes(close)).not.toEqual(expect.arrayContaining(['min-h-10']));
	});
});

describe('modal subtitle', () => {
	it('is set at 12px', () => {
		openBase({ title: 'Details', subtitle: 'One line of context' });
		const subtitle = Array.from(dialog().querySelectorAll('p')).find((p) => p.textContent === 'One line of context')!;
		expect(classes(subtitle)).toContain('text-sm');
		expect(classes(subtitle)).not.toContain('text-xs');
	});
});
