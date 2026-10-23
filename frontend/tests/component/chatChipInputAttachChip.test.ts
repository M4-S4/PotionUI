import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const suggestChatResources = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: { suggestChatResources }
}));

const { createClassComponent } = await import('svelte/legacy');
const { default: ChatChipInput } = await import('$lib/components/chat/ChatChipInput.svelte');

let cleanup: (() => void) | undefined;

beforeEach(() => {
	vi.useFakeTimers();
	suggestChatResources.mockResolvedValue({ success: true, data: { suggestions: [] } });
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
	suggestChatResources.mockReset();
	vi.useRealTimers();
});

function typeInto(editor: HTMLElement, text: string) {
	editor.textContent = text;
	const range = document.createRange();
	range.setStart(editor.firstChild as Text, text.length);
	range.collapse(true);
	const selection = window.getSelection()!;
	selection.removeAllRanges();
	selection.addRange(range);
	editor.dispatchEvent(new Event('input', { bubbles: true }));
}

describe('ChatChipInput attaching a resource', () => {
	it('lists @form. suggestions in a named listbox and mounts a chip for the one picked with Enter', async () => {
		const target = document.body.appendChild(document.createElement('div'));
		const component = createClassComponent({
			component: ChatChipInput as never,
			target,
			props: { value: '', resources: {}, mode: 'generation', formData: { prompt: 'a cat', steps: 20 }, loraSelections: {} }
		});
		cleanup = () => component.$destroy();
		const editor = target.querySelector('.chat-chip-input') as HTMLElement;

		typeInto(editor, '@form.');
		await vi.advanceTimersByTimeAsync(260);
		const options = document.querySelectorAll('[role="listbox"][aria-label="Suggestions"] [role="option"]');
		expect(options.length).toBeGreaterThan(0);

		editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
		await vi.advanceTimersByTimeAsync(0);

		expect(editor.querySelectorAll('.inline-chip-container')).toHaveLength(1);
		expect(editor.querySelector('.inline-chip button')).not.toBeNull();
	});

	it('ignores Enter while @ suggestions are still loading, then attaches once they arrive', async () => {
		suggestChatResources.mockResolvedValue({
			success: true,
			data: {
				suggestions: [
					{ uri: 'presets.portrait', label: 'Portrait', kind: 'preset', has_children: false, icon: 'box', badge: 'preset' }
				]
			}
		});
		const target = document.body.appendChild(document.createElement('div'));
		const component = createClassComponent({
			component: ChatChipInput as never,
			target,
			props: { value: '', resources: {}, mode: 'generation', formData: {}, loraSelections: {} }
		});
		const submits: unknown[] = [];
		component.$on('submit', (e: CustomEvent) => submits.push(e));
		cleanup = () => component.$destroy();
		const editor = target.querySelector('.chat-chip-input') as HTMLElement;
		const errors: unknown[] = [];
		const onError = (event: ErrorEvent) => {
			errors.push(event.error);
			event.preventDefault();
		};
		window.addEventListener('error', onError);

		typeInto(editor, '@pro');
		const early = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
		editor.dispatchEvent(early);
		await vi.advanceTimersByTimeAsync(0);

		expect(errors).toEqual([]);
		expect(early.defaultPrevented).toBe(true);
		expect(submits).toEqual([]);
		expect(editor.querySelectorAll('.inline-chip-container')).toHaveLength(0);
		expect(suggestChatResources).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(260);
		expect(suggestChatResources).toHaveBeenCalledWith('pro', 'generation');
		expect(document.querySelectorAll('[role="listbox"][aria-label="Suggestions"] [role="option"]')).toHaveLength(1);

		editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
		await vi.advanceTimersByTimeAsync(0);
		window.removeEventListener('error', onError);

		expect(errors).toEqual([]);
		expect(editor.querySelectorAll('.inline-chip-container')).toHaveLength(1);
		expect(editor.querySelector('.inline-chip')?.textContent).toContain('Portrait');
	});

	for (const key of ['ArrowDown', 'ArrowUp']) {
		it(`keeps the selection on the first row when ${key} is pressed before @ suggestions load`, async () => {
			suggestChatResources.mockResolvedValue({
				success: true,
				data: {
					suggestions: [
						{ uri: 'presets.portrait', label: 'Portrait', kind: 'preset', has_children: false, icon: 'box', badge: 'preset' },
						{ uri: 'presets.product', label: 'Product', kind: 'preset', has_children: false, icon: 'box', badge: 'preset' }
					]
				}
			});
			const target = document.body.appendChild(document.createElement('div'));
			const component = createClassComponent({
				component: ChatChipInput as never,
				target,
				props: { value: '', resources: {}, mode: 'generation', formData: {}, loraSelections: {} }
			});
			cleanup = () => component.$destroy();
			const editor = target.querySelector('.chat-chip-input') as HTMLElement;

			typeInto(editor, '@pro');
			const arrow = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
			editor.dispatchEvent(arrow);
			expect(arrow.defaultPrevented).toBe(true);
			await vi.advanceTimersByTimeAsync(260);

			const options = document.querySelectorAll('[role="listbox"][aria-label="Suggestions"] [role="option"]');
			expect(options).toHaveLength(2);
			expect(options[0].getAttribute('aria-selected')).toBe('true');

			editor.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
			await vi.advanceTimersByTimeAsync(0);

			expect(editor.querySelectorAll('.inline-chip-container')).toHaveLength(1);
			expect(editor.querySelector('.inline-chip')?.textContent).toContain('Portrait');
		});
	}
});
