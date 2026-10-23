import { describe, it, expect, afterEach } from 'vitest';

const { default: tooltip } = await import('$lib/actions/tooltip');

let cleanup: Array<() => void> = [];

async function openBubble(param: Parameters<typeof tooltip>[1]): Promise<HTMLElement> {
	const node = document.createElement('button');
	node.textContent = 'x';
	document.body.appendChild(node);
	const action = tooltip(node, param);
	cleanup.push(() => {
		action.destroy();
		node.remove();
	});
	node.dispatchEvent(new FocusEvent('focusin'));
	await Promise.resolve();
	return document.body.querySelector('[role="tooltip"]') as HTMLElement;
}

afterEach(() => {
	cleanup.forEach((fn) => fn());
	cleanup = [];
});

describe('tooltip bubble type size', () => {
	it('sets the text at 12px', async () => {
		const bubble = await openBubble('Copy');
		expect(bubble.className).toContain('text-sm');
		expect(bubble.className).not.toMatch(/\btext-(2xs|xs)\b/);
	});

	it('sets the shortcut chip at 12px too', async () => {
		const bubble = await openBubble({ text: 'Save', kbd: 'Ctrl+S' });
		const cap = bubble.querySelector('kbd') as HTMLElement;
		expect(cap.textContent?.trim()).toBe('Ctrl+S');
		expect(cap.className).toContain('text-sm');
		expect(cap.className).not.toMatch(/\btext-(2xs|xs)\b/);
	});
});
