import { describe, it, expect, vi, afterEach } from 'vitest';

const { default: ChatInput } = await import('$lib/components/chat/ChatInput.svelte');
const { createClassComponent } = await import('svelte/legacy');

let mounted: { $destroy: () => void } | null = null;

function mount(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	mounted = createClassComponent({ component: ChatInput as never, target, props }) as never;
	return target;
}

afterEach(() => {
	mounted?.$destroy();
	mounted = null;
	document.body.innerHTML = '';
});

describe('ChatInput stop button', () => {
	it('has an accessible name and calls onStop while generating', () => {
		const onStop = vi.fn();
		const target = mount({ isGenerating: true, onStop });
		const stop = target.querySelector<HTMLButtonElement>('button[aria-label="Stop generating"]');
		expect(stop).not.toBeNull();
		stop?.click();
		expect(onStop).toHaveBeenCalledOnce();
	});

	it('names the send button when idle', () => {
		const target = mount({ isGenerating: false, onStop: vi.fn() });
		expect(target.querySelector('button[aria-label="Send message"]')).not.toBeNull();
		expect(target.querySelector('button[aria-label="Stop generating"]')).toBeNull();
	});
});
