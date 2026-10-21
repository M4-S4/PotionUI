// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount } from 'svelte';
import ResolutionField from '$lib/components/form-fields/ResolutionField.svelte';

const flush = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

let instance: ReturnType<typeof mount> | null = null;

async function openPanel() {
	instance = mount(ResolutionField as never, {
		target: document.body,
		props: {
			name: 'resolution',
			config: { title: 'Resolution', options: ['512x512', '768x768'] },
			value: '512x512',
			onChange: vi.fn()
		}
	});
	await flush(5);
	const trigger = document.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!;
	trigger.click();
	await flush(20);
	return trigger;
}

const panel = () => document.body.querySelector('[role="listbox"]');
const customInput = () => document.body.querySelector('input[aria-label="Custom width"]');

function pressEscape() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
}

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
});

describe('ResolutionField Escape', () => {
	it('leaves custom mode first, then closes the panel and focuses the trigger', async () => {
		const trigger = await openPanel();
		expect(panel()).not.toBeNull();

		Array.from(document.body.querySelectorAll('button'))
			.find((b) => b.textContent?.includes('Custom size'))!
			.click();
		await flush(20);
		expect(customInput()).not.toBeNull();

		pressEscape();
		await flush(20);
		expect(panel()).not.toBeNull();
		expect(customInput()).toBeNull();

		pressEscape();
		await flush(200);
		expect(panel()).toBeNull();
		expect(document.activeElement).toBe(trigger);
	});
});
