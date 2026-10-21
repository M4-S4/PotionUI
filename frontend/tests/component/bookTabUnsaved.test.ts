import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';

const { default: BookTab } = await import('$lib/components/BookTab.svelte');
const { createClassComponent } = await import('svelte/legacy');

let component: { $destroy: () => void } | undefined;

function render(unsaved: boolean) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: BookTab as never,
		target,
		props: { id: 't1', name: 'Generation 1', isActive: true, canDelete: true, unsaved }
	});
	flushSync();
	return target.querySelector<HTMLButtonElement>('button.book-tab')!;
}

afterEach(() => {
	component?.$destroy();
	component = undefined;
	vi.useRealTimers();
	document.body.innerHTML = '';
});

describe('BookTab unsaved marker', () => {
	it('shows a dot that says Unsaved changes on hover and in the tab name', async () => {
		vi.useFakeTimers();
		const tab = render(true);
		const dot = tab.querySelector('[data-testid="tab-unsaved"]');
		expect(dot).not.toBeNull();
		expect(tab.textContent?.replace(/\s+/g, ' ')).toContain('Generation 1 , unsaved changes');
		expect(tab.querySelector('.tab-name')?.textContent).toBe('Generation 1');
		dot!.parentElement!.dispatchEvent(new MouseEvent('mouseenter'));
		await vi.advanceTimersByTimeAsync(200);
		flushSync();
		expect(document.body.textContent).toContain('Unsaved changes');
		expect(document.body.querySelectorAll('[title]').length).toBe(0);
	});

	it('shows nothing for a clean tab', () => {
		const tab = render(false);
		expect(tab.querySelector('[data-testid="tab-unsaved"]')).toBeNull();
		expect(tab.textContent).not.toContain('unsaved');
	});
});
