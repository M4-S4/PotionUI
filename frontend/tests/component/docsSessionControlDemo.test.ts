import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';

const { default: Reference } = await import('../../src/routes/docs/components/live/InteractiveComponentsReference.svelte');

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement;

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	target?.remove();
});

describe('docs SessionControl demo', () => {
	it('sits in a labelled step the way the toolbar renders it, so the stacked select lines up', () => {
		target = document.createElement('div');
		document.body.appendChild(target);
		instance = mount(Reference as never, { target, props: {} });
		flushSync();
		const trigger = target.querySelector<HTMLElement>('[data-testid="session-trigger"]')!;
		expect(trigger).not.toBeNull();
		const step = trigger.closest('.ctx-step')!;
		expect(step).not.toBeNull();
		const label = step.querySelector<HTMLElement>('.ctx-label')!;
		expect(label.textContent).toBe('Session');
		expect(step.firstElementChild).toBe(label);
		expect(trigger.getAttribute('aria-labelledby')!.split(' ')[0]).toBe(label.id);
	});
});
