// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get, writable } from 'svelte/store';

const hoisted = vi.hoisted(() => ({
	goto: vi.fn(),
	setAuthHeader: vi.fn(),
	getCurrentUser: vi.fn(),
	clearAuth: vi.fn(),
	authActions: {
		logout: vi.fn(),
		logoutAccount: vi.fn(),
		logoutAll: vi.fn(),
		finishSignIn: vi.fn()
	}
}));

vi.mock('$app/navigation', () => ({ goto: hoisted.goto }));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: () => ({ get: vi.fn().mockResolvedValue({ data: { data: {} } }), put: vi.fn() }),
		setAuthHeader: hoisted.setAuthHeader,
		clearAuth: hoisted.clearAuth,
		getCurrentUser: hoisted.getCurrentUser,
		getToken: () => null,
		getSetupStatus: vi.fn().mockResolvedValue(null),
		getLoginProviders: vi.fn().mockResolvedValue([])
	}
}));

const authState = writable<Record<string, unknown>>({
	isAuthenticated: true,
	loading: false,
	error: null,
	token: 't',
	user: { id: 'alice', username: 'alice', account_type: 'ADMIN', avatar_url: null, content_restricted: false }
});

vi.mock('$lib/stores/auth', () => ({
	authStore: Object.assign({ subscribe: authState.subscribe }, hoisted.authActions)
}));

const accounts = await import('$lib/stores/accounts');
const { default: UserMenu } = await import('$lib/components/UserMenu.svelte');
const { createClassComponent } = await import('svelte/legacy');

const { COMMUNITY_LINKS } = await import('$lib/community');

const tick = async (n = 6) => {
	for (let i = 0; i < n; i++) await vi.advanceTimersByTimeAsync(0);
};

function mount(component: unknown) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({ component: component as never, target, props: {} });
	return {
		target,
		destroy: () => {
			instance.$destroy();
			target.remove();
		}
	};
}

beforeEach(() => {
	vi.useFakeTimers();
	localStorage.clear();
	accounts.resetAccountsForTests();
});

afterEach(() => {
	vi.useRealTimers();
	document.body.innerHTML = '';
});

describe('account menu community links', () => {
	it('lists the four links in one labelled row that opens in a new tab', async () => {
		const view = mount(UserMenu);
		view.target.querySelector<HTMLButtonElement>('button[aria-label="Account menu"]')!.click();
		await tick();
		const row = document.querySelector('[data-testid="user-menu-community"]')!;
		const links = Array.from(row.querySelectorAll<HTMLAnchorElement>('a'));
		expect(links.map((a) => a.getAttribute('aria-label'))).toEqual([
			'Website',
			'GitHub',
			'Reddit',
			'Discord'
		]);
		expect(links.map((a) => a.getAttribute('href'))).toEqual(COMMUNITY_LINKS.map((l) => l.href));
		for (const a of links) {
			expect(a.getAttribute('target')).toBe('_blank');
			expect(a.getAttribute('rel')).toBe('noopener noreferrer');
			expect(a.hasAttribute('title')).toBe(false);
		}
		view.destroy();
	});
});
