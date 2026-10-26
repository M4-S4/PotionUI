import { afterEach, describe, expect, it } from 'vitest';
import { flushSync } from 'svelte';
import type { ChatNotice } from '../../src/lib/types/chat';

const { default: ChatNotices } = await import('../../src/lib/components/chat/ChatNotices.svelte');
const { createClassComponent } = await import('svelte/legacy');

function mount(notices: ChatNotice[]) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: ChatNotices as never, target, props: { notices } });
	flushSync();
	return target;
}

const items = (target: HTMLElement) => [...target.querySelectorAll<HTMLElement>('[data-testid="chat-notice"]')];

afterEach(() => {
	document.body.innerHTML = '';
});

describe('ChatNotices', () => {
	it('renders nothing for an empty list', () => {
		expect(mount([]).querySelector('[data-testid="chat-notices"]')).toBeNull();
	});

	it('shows the server message of each notice, in order, tagged with its kind', () => {
		const target = mount([
			{ kind: 'retrying', message: 'Rate limited; retrying in 4 s.' },
			{ kind: 'reasoning_dropped', message: 'Reasoning made with a different API key was left out.', step_seq: 12 },
			{ kind: 'approval_expired', message: 'This approval expired.' }
		]);

		expect(items(target).map((el) => [el.dataset.kind, el.textContent?.trim()])).toEqual([
			['retrying', 'Rate limited; retrying in 4 s.'],
			['reasoning_dropped', 'Reasoning made with a different API key was left out.'],
			['approval_expired', 'This approval expired.']
		]);
	});

	it('announces itself politely as a status region', () => {
		const region = mount([{ kind: 'retrying', message: 'Retrying.' }]).querySelector('[data-testid="chat-notices"]');

		expect(region?.getAttribute('role')).toBe('status');
		expect(region?.getAttribute('aria-live')).toBe('polite');
	});

	it('gives each kind its own tone and still shows a kind it does not know', () => {
		const target = mount([
			{ kind: 'retrying', message: 'a' },
			{ kind: 'reasoning_dropped', message: 'b' },
			{ kind: 'something_new' as never, message: 'c' }
		]);
		const icons = items(target).map((el) => el.querySelector('svg')?.getAttribute('class') ?? '');

		expect(icons[0]).toContain('text-warning');
		expect(icons[1]).toContain('text-info');
		expect(icons[2]).toContain('text-fg-subtle');
		expect(items(target)[2].textContent).toContain('c');
	});

	it('uses no native title tooltip', () => {
		expect(mount([{ kind: 'retrying', message: 'a' }]).querySelector('[title]')).toBeNull();
	});
});
