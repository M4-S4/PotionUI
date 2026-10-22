import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import Kbd from '$lib/components/ui/Kbd.svelte';

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
});

function render(keys: string | string[]) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(Kbd, { target, props: { keys } });
	flushSync();
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return target.firstElementChild as HTMLElement;
}

describe('Kbd', () => {
	it('renders every key cap at 12px', () => {
		const root = render(['Ctrl', 'Enter']);
		const caps = Array.from(root.querySelectorAll('kbd'));

		expect(caps.map((cap) => cap.textContent?.trim())).toEqual(['Ctrl', 'Enter']);
		for (const node of [root, ...caps]) {
			expect(node.className).toContain('text-sm');
			expect(node.className).not.toMatch(/\btext-(2xs|xs)\b/);
		}
	});

	it('keeps every key cap 20px tall so a Button holding one keeps its height', () => {
		const caps = Array.from(render(['Ctrl', 'K']).querySelectorAll('kbd'));

		expect(caps).toHaveLength(2);
		for (const cap of caps) {
			const classes = cap.className.split(/\s+/);
			expect(classes).toEqual(expect.arrayContaining(['text-sm', 'py-px', 'border']));
			expect(classes).not.toContain('py-0.5');
			expect(classes).not.toContain('py-1');
		}
	});

	it('puts a plus between the caps of a chord and none for a single key', () => {
		expect(render(['Ctrl', 'K']).textContent?.replace(/\s+/g, '')).toBe('Ctrl+K');
		cleanup?.();
		expect(render('Esc').textContent?.replace(/\s+/g, '')).toBe('Esc');
	});
});
