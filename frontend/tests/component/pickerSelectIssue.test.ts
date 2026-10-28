import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { settle } from './helpers/settle';

const { default: PickerSelect } = await import('../../src/lib/components/preset/PickerSelect.svelte');

let instance: ReturnType<typeof mount> | null = null;

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
});

describe('PickerSelect option issue', () => {
	it('shows the description and the missing-file note together', async () => {
		const target = document.body.appendChild(document.createElement('div'));
		instance = mount(PickerSelect, {
			target,
			props: {
				label: 'Mode',
				value: 'video',
				options: [
					{ value: 'video', label: 'Video' },
					{
						value: 'remove_object',
						label: 'Remove object',
						description: 'Erase objects from a video.',
						issue: 'A required file is missing.'
					}
				],
				onchange: () => {}
			}
		});
		flushSync();
		target.querySelector('button')!.click();
		flushSync();
		await settle();

		const row = Array.from(document.body.querySelectorAll('button')).find((b) => b.textContent?.includes('Remove object'))!;
		expect(row.textContent).toContain('Erase objects from a video.');
		expect(row.querySelector('[data-option-issue]')?.textContent).toContain('A required file is missing.');
	});
});
