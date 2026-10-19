// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import type { PresetInfo } from '$lib/types/api';

const { default: PresetPicker } = await import('$lib/components/preset/PresetPicker.svelte');

const presets = [
	{ id: 'alpha-render', name: 'Alpha Render', engine: 'native', version: '1', category: 'image' },
	{ id: 'beta-toolkit', name: 'Beta Toolkit', engine: 'native', version: '1', category: 'utility' }
] as unknown as PresetInfo[];

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLDivElement | undefined;

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	target?.remove();
	target = undefined;
	document.body.innerHTML = '';
});

async function openPicker() {
	target = document.createElement('div');
	document.body.appendChild(target);
	instance = mount(PresetPicker, { target, props: { presets, selectedPreset: '' } });
	flushSync();
	(document.querySelector('[data-testid="preset-picker-trigger"]') as HTMLButtonElement).click();
	await tick();
	flushSync();
}

function chipButton(label: string): HTMLButtonElement | undefined {
	return Array.from(document.querySelectorAll<HTMLButtonElement>('button[aria-pressed]')).find((button) =>
		button.textContent?.includes(label)
	);
}

function rowTitles(): string[] {
	return Array.from(document.querySelectorAll('[data-pane-row]')).map((row) => row.textContent ?? '');
}

describe('PresetPicker Tools chip', () => {
	it('renders a Tools chip after the media chips', async () => {
		await openPicker();
		const labels = Array.from(document.querySelectorAll<HTMLButtonElement>('button[aria-pressed]')).map((button) =>
			button.textContent?.trim() ?? ''
		);
		const toolsIndex = labels.findIndex((label) => label.startsWith('Tools'));
		const threeDIndex = labels.findIndex((label) => label.startsWith('3D'));
		expect(toolsIndex).toBeGreaterThan(-1);
		expect(toolsIndex).toBeGreaterThan(threeDIndex);
	});

	it('filters the list to utility presets when Tools is selected', async () => {
		await openPicker();
		expect(rowTitles()).toHaveLength(2);

		chipButton('Tools')?.click();
		await tick();
		flushSync();

		const rows = rowTitles();
		expect(rows).toHaveLength(1);
		expect(rows[0]).toContain('Beta Toolkit');
		expect(chipButton('Tools')?.getAttribute('aria-pressed')).toBe('true');
	});

	it('labels utility presets as Tools and never as Utility', async () => {
		await openPicker();
		const text = document.body.textContent ?? '';
		expect(text).toContain('Tools');
		expect(text).not.toContain('Utility');
	});
});
