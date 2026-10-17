import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync, tick } from 'svelte';
import { tabsStore } from '$lib/stores/tabs';
import { registerLoraSelectionSource } from '$lib/stores/loraPickerSelections';
import LoraEditor from '$lib/components/compare/editors/LoraEditor.svelte';
import type { AxisCandidate, CompareAxisValue } from '$lib/generation/compare/types';

const candidate = { field: 'loras', label: 'LoRA', type: 'lora_picker', editor: 'lora', options: [] } as unknown as AxisCandidate;

let cleanup: (() => void) | null = null;
afterEach(() => cleanup?.());

async function mountEditor() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const tabId = tabsStore.addTabWithData('Compare LoRA', {
		formData: { loras: [{ model: 'model:abc123', strength: 0.5 }] }
	} as never);
	const onChange = vi.fn<(values: CompareAxisValue[]) => void>();
	const editor = mount(LoraEditor, { target, props: { candidate, axis: null, tabId, onChange } });
	const source = registerLoraSelectionSource(tabId, 'loras');
	cleanup = () => {
		unmount(editor);
		source.unregister();
		tabsStore.removeTab(tabId);
		target.remove();
	};
	flushSync();
	await tick();
	return { target, onChange, source };
}

describe('compare LoRA editor', () => {
	it('labels form LoRAs by display name once the picker resolves them', async () => {
		const { target, onChange, source } = await mountEditor();
		expect(target.querySelector('option')?.textContent).toBe('abc123');

		source.set([{ id: 'abc123', name: 'Film Grain', strength: 0.5 }]);
		flushSync();
		await tick();

		expect(target.querySelector('option')?.textContent).toBe('Film Grain');
		const labels = onChange.mock.calls.at(-1)![0].map((value) => value.label);
		expect(labels.length).toBeGreaterThan(0);
		expect(labels.every((label) => label.startsWith('Film Grain · '))).toBe(true);
	});
});
