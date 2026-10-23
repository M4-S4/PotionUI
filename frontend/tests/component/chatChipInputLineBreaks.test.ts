// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		suggestChatResources: vi.fn().mockResolvedValue({ success: true, data: { suggestions: [] } })
	}
}));

const { createClassComponent } = await import('svelte/legacy');
const { default: ChatChipInput } = await import('$lib/components/chat/ChatChipInput.svelte');

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
});

function typedValue(html: string): string {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: ChatChipInput as never,
		target,
		props: { value: '', resources: {}, mode: '', formData: {}, loraSelections: {} }
	});
	cleanup = () => {
		component.$destroy();
		target.remove();
	};
	let emitted = '';
	component.$on('change', (e: CustomEvent<{ value: string }>) => {
		emitted = e.detail.value;
	});
	const editor = target.querySelector<HTMLElement>('[contenteditable]')!;
	editor.innerHTML = html;
	editor.dispatchEvent(new Event('input', { bubbles: true }));
	return emitted;
}

describe('ChatChipInput line breaks', () => {
	it.each([
		['Chromium wraps later lines in blocks', 'a<div>b</div>', 'a\nb'],
		['Chromium empty line between two lines', 'a<div><br></div><div>b</div>', 'a\n\nb'],
		['Firefox uses bare breaks', 'a<br>b', 'a\nb'],
		['single line stays untouched', 'ab', 'ab']
	])('%s', (_name, html, expected) => {
		expect(typedValue(html)).toBe(expected);
	});
});
