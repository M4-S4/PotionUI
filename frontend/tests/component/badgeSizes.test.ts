// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, createRawSnippet } from 'svelte';
import Badge from '$lib/components/ui/Badge.svelte';
import Button from '$lib/components/ui/Button.svelte';
import CopyButton from '$lib/components/ui/CopyButton.svelte';

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
});

function mountInto(component: any, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const children = createRawSnippet(() => ({ render: () => '<span>Done</span>' }));
	const instance = mount(component, { target, props: { ...props, children } });
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return target.firstElementChild as HTMLElement;
}

function renderBadge(size?: 'sm' | 'md' | 'lg') {
	return mountInto(Badge, { size, dot: true, variant: 'success' });
}

function renderButton(size: 'xs' | 'sm') {
	return mountInto(Button, { size });
}

function renderCopy(size: 'xs' | 'sm') {
	return mountInto(CopyButton, { size, text: 'x', label: 'Copy' });
}

const SUB_FLOOR = /\btext-(2xs|xs)\b/;

describe('Badge sizes', () => {
	it('renders every size at 12 px text', () => {
		for (const size of [undefined, 'sm', 'md', 'lg'] as const) {
			const badge = renderBadge(size);
			expect(badge.className).toContain('text-sm');
			expect(badge.className).not.toMatch(SUB_FLOOR);
			expect(badge.textContent?.trim()).toBe('Done');
			cleanup?.();
		}
	});

	it('differs between sizes by padding only', () => {
		const pad = (size: 'sm' | 'md' | 'lg') => {
			const cls = renderBadge(size).className;
			cleanup?.();
			return cls.match(/\bpx-[\d.]+ py-[\d.]+\b/)?.[0];
		};
		expect(new Set([pad('sm'), pad('md'), pad('lg')]).size).toBe(3);
	});
});

describe('Button and CopyButton small sizes', () => {
	it('keeps xs Button at 12 px text with tighter padding than sm', () => {
		const xs = renderButton('xs').className;
		cleanup?.();
		const sm = renderButton('sm').className;
		expect(xs).toContain('text-sm');
		expect(xs).not.toMatch(SUB_FLOOR);
		expect(xs).toContain('py-1 ');
		expect(sm).toContain('py-1.5');
		expect(xs).toContain('px-2 ');
		expect(sm).toContain('px-3');
	});

	it('keeps labeled xs CopyButton at 12 px text with tighter padding than sm', () => {
		const xs = renderCopy('xs').className;
		cleanup?.();
		const sm = renderCopy('sm').className;
		expect(xs).toContain('text-sm');
		expect(xs).not.toMatch(SUB_FLOOR);
		expect(xs).toContain('py-1 ');
		expect(sm).toContain('py-1.5');
	});
});
