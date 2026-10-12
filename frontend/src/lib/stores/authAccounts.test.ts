import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import { TABS_STORAGE_KEY } from '$lib/types/tabs';
import { ACCOUNTS_STORAGE_KEY } from './accountRegistry';

const hoisted = vi.hoisted(() => ({
	goto: vi.fn(),
	expiredHandler: { current: null as null | ((token: string | null) => void) },
	header: { current: null as string | null },
	userById: {} as Record<string, unknown>,
	meOutcomes: [] as unknown[],
	refreshedHandler: { current: null as null | ((previous: string, next: string) => void) }
}));

vi.mock('$app/environment', () => ({ browser: true }));
vi.mock('$app/navigation', () => ({ goto: hoisted.goto }));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getCurrentUser: vi.fn(async () => {
			if (hoisted.meOutcomes.length > 0) {
				const outcome = hoisted.meOutcomes.shift();
				if (outcome instanceof Error || (outcome as any)?.response) {
					if ((outcome as any).response?.status === 401) {
						hoisted.expiredHandler.current?.(hoisted.header.current);
					}
					throw outcome;
				}
				return outcome;
			}
			return {
				success: true,
				data: hoisted.userById[(hoisted.header.current ?? '').replace('tok-', '')]
			};
		}),
		setAuthHeader: vi.fn((token: string) => {
			hoisted.header.current = token;
			localStorage.setItem('auth_token', token);
		}),
		clearAuth: vi.fn(() => {
			hoisted.header.current = null;
			localStorage.removeItem('auth_token');
		}),
		setOnAuthExpired: vi.fn((cb: (token: string | null) => void) => {
			hoisted.expiredHandler.current = cb;
		}),
		setOnTokenRefreshed: vi.fn((cb: (previous: string, next: string) => void) => {
			hoisted.refreshedHandler.current = cb;
		}),
		endMediaSession: vi.fn(async () => undefined),
		getToken: vi.fn(() => hoisted.header.current),
		login: vi.fn(),
		register: vi.fn()
	}
}));

const location = { reload: vi.fn(), assign: vi.fn(), pathname: '/generate' };

function stubLocalStorage() {
	const store = new Map<string, string>();
	(globalThis as any).localStorage = {
		get length() {
			return store.size;
		},
		key: (i: number) => Array.from(store.keys())[i] ?? null,
		getItem: (key: string) => store.get(key) ?? null,
		setItem: (key: string, value: string) => void store.set(key, value),
		removeItem: (key: string) => void store.delete(key),
		clear: () => store.clear()
	};
}

function userRow(id: string) {
	return {
		id,
		username: id,
		email: '',
		account_type: 'USER',
		created_at: null,
		last_login: null,
		avatar_url: null,
		has_local_password: true
	};
}

function seed(activeId: string | null, ids: string[]) {
	localStorage.setItem(
		ACCOUNTS_STORAGE_KEY,
		JSON.stringify({
			v: 1,
			activeId,
			accounts: ids.map((id) => ({
				userId: id,
				username: id,
				role: 'USER',
				avatar: null,
				token: `tok-${id}`,
				signedInAt: Date.now(),
				expired: false
			}))
		})
	);
	for (const id of ids) localStorage.setItem(`${TABS_STORAGE_KEY}::${id}`, `${id}-tabs`);
	if (activeId) localStorage.setItem('auth_token', `tok-${activeId}`);
}

async function loadAuth() {
	vi.resetModules();
	const auth = await import('./auth');
	const accounts = await import('./accounts');
	const { api } = await import('$lib/services/api/index');
	await new Promise((r) => setTimeout(r, 0));
	return { ...auth, accounts, api };
}

beforeEach(() => {
	stubLocalStorage();
	hoisted.goto.mockClear();
	hoisted.header.current = null;
	hoisted.meOutcomes = [];
	hoisted.userById = { alice: userRow('alice'), bob: userRow('bob') };
	location.assign.mockClear();
	location.reload.mockClear();
	vi.stubGlobal('location', location);
});

afterEach(() => {
	vi.unstubAllGlobals();
	delete (globalThis as any).localStorage;
});

describe('authStore with several accounts', () => {
	it('boots authenticated from the registry active account', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore } = await loadAuth();

		const state = get(authStore);
		expect(state.isAuthenticated).toBe(true);
		expect(state.token).toBe('tok-alice');
		expect(state.user?.id).toBe('alice');
	});

	it('logout removes only the active account, keeps the others, and goes to the login page', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, api, accounts } = await loadAuth();

		authStore.logout();

		expect(api.endMediaSession).toHaveBeenCalledTimes(1);
		expect(accounts.currentRegistry().accounts.map((a) => a.userId)).toEqual(['bob']);
		expect(accounts.currentRegistry().activeId).toBeNull();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)).toBeNull();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::bob`)).toBe('bob-tabs');
		expect(location.assign).toHaveBeenCalledWith('/login');
		expect(get(authStore).isAuthenticated).toBe(false);
	});

	it('logoutAccount on a non-active account touches neither the session nor the cookie', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, api, accounts } = await loadAuth();
		vi.mocked(api.endMediaSession).mockClear();
		vi.mocked(api.clearAuth).mockClear();

		authStore.logoutAccount('bob');

		expect(accounts.currentRegistry().accounts.map((a) => a.userId)).toEqual(['alice']);
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::bob`)).toBeNull();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)).toBe('alice-tabs');
		expect(api.endMediaSession).not.toHaveBeenCalled();
		expect(api.clearAuth).not.toHaveBeenCalled();
		expect(get(authStore).isAuthenticated).toBe(true);
		expect(location.assign).not.toHaveBeenCalled();
	});

	it('logoutAccount on the active account behaves like logout', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, accounts } = await loadAuth();

		authStore.logoutAccount('alice');

		expect(accounts.currentRegistry().accounts.map((a) => a.userId)).toEqual(['bob']);
		expect(get(authStore).isAuthenticated).toBe(false);
	});

	it('logoutAll wipes every account, every namespaced key and the token', async () => {
		seed('alice', ['alice', 'bob']);
		localStorage.setItem('theme', 'dark');
		const { authStore, accounts } = await loadAuth();

		authStore.logoutAll();

		expect(accounts.currentRegistry().accounts).toEqual([]);
		expect(localStorage.getItem(ACCOUNTS_STORAGE_KEY)).toBeNull();
		expect(localStorage.getItem('auth_token')).toBeNull();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)).toBeNull();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::bob`)).toBeNull();
		expect(localStorage.getItem('theme')).toBe('dark');
		expect(location.assign).toHaveBeenCalledWith('/login');
	});

	it('a 401 on the active account goes to login with expired=1, keeps its tabs and does not switch', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, accounts, api } = await loadAuth();
		vi.mocked(api.setAuthHeader).mockClear();

		hoisted.expiredHandler.current!('tok-alice');

		expect(hoisted.goto).toHaveBeenCalledWith('/login?expired=1');
		const registry = accounts.currentRegistry();
		expect(registry.accounts.map((a) => [a.userId, a.expired])).toEqual([
			['alice', true],
			['bob', false]
		]);
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)).toBe('alice-tabs');
		expect(get(authStore).isAuthenticated).toBe(false);
		expect(localStorage.getItem('auth_token')).toBeNull();
		expect(api.setAuthHeader).not.toHaveBeenCalled();
	});

	it('a 401 on a non-active account marks only that account and leaves the session alone', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, accounts } = await loadAuth();

		hoisted.expiredHandler.current!('tok-bob');

		expect(hoisted.goto).not.toHaveBeenCalled();
		expect(accounts.currentRegistry().accounts.map((a) => [a.userId, a.expired])).toEqual([
			['alice', false],
			['bob', true]
		]);
		expect(get(authStore).isAuthenticated).toBe(true);
	});

	it('ignores 401s while an account switch is verifying its token', async () => {
		seed('alice', ['alice', 'bob']);
		const { accounts } = await loadAuth();
		let during: Promise<unknown> | null = null;
		const { api } = await import('$lib/services/api/index');
		vi.mocked(api.getCurrentUser).mockImplementationOnce(async () => {
			hoisted.expiredHandler.current!('tok-bob');
			return { success: true, data: userRow('bob') } as never;
		});

		during = accounts.switchTo('bob');
		await during;

		expect(hoisted.goto).not.toHaveBeenCalled();
		expect(accounts.currentRegistry().accounts.map((a) => a.expired)).toEqual([false, false]);
		expect(accounts.currentRegistry().activeId).toBe('bob');
	});

	it('upgrades a legacy single-token install into a one-account registry after /me', async () => {
		localStorage.setItem('auth_token', 'tok-alice');
		localStorage.setItem('auth_last_user_id', 'alice');
		const { accounts } = await loadAuth();

		expect(accounts.currentRegistry().activeId).toBe('alice');
		expect(accounts.currentRegistry().accounts.map((a) => a.userId)).toEqual(['alice']);
	});

	it('login of a sixth user is rejected and leaves the session unauthenticated', async () => {
		seed(null, ['u1', 'u2', 'u3', 'u4', 'u5']);
		hoisted.userById.u6 = userRow('u6');
		const { authStore, api } = await loadAuth();
		vi.mocked(api.login).mockImplementation(async () => {
			hoisted.header.current = 'tok-u6';
			return { access_token: 'tok-u6', token_type: 'bearer' } as never;
		});

		const result = await authStore.login('u6', 'pw');

		expect(result.success).toBe(false);
		expect(get(authStore).isAuthenticated).toBe(false);
	});

	it('finishSignIn navigates softly on a first login in a clean browser', async () => {
		const { authStore, accounts } = await loadAuth();
		accounts.syncActive(userRow('alice') as never, 'tok-alice');

		authStore.finishSignIn('/generate');

		expect(location.assign).not.toHaveBeenCalled();
		expect(hoisted.goto).toHaveBeenCalledWith('/generate', { replaceState: true });
	});

	it('a soft first login saves the open tabs under the new account right away', async () => {
		const { authStore, accounts } = await loadAuth();
		expect(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)).toBeNull();
		accounts.syncActive(userRow('alice') as never, 'tok-alice');

		authStore.finishSignIn('/generate');

		const saved = JSON.parse(localStorage.getItem(`${TABS_STORAGE_KEY}::alice`)!);
		expect(saved.tabs).toHaveLength(1);
		expect(localStorage.getItem(TABS_STORAGE_KEY)).toBeNull();
	});

	it('finishSignIn reloads when stored state is waiting for the new identity', async () => {
		localStorage.setItem(TABS_STORAGE_KEY, 'seeded-before-login');
		const { authStore, accounts } = await loadAuth();
		accounts.syncActive(userRow('alice') as never, 'tok-alice');

		authStore.finishSignIn('/generate');

		expect(location.assign).toHaveBeenCalledWith('/generate');
		expect(hoisted.goto).not.toHaveBeenCalled();
	});

	it('finishSignIn reloads when a different account signs in than the tab booted with', async () => {
		seed('alice', ['alice']);
		const { authStore, accounts } = await loadAuth();
		accounts.syncActive(userRow('bob') as never, 'tok-bob');

		authStore.finishSignIn('/generate');

		expect(location.assign).toHaveBeenCalledWith('/generate');
	});

	it('finishSignIn stays soft for the same account signing in again', async () => {
		seed('alice', ['alice']);
		const { authStore, accounts } = await loadAuth();
		accounts.syncActive(userRow('alice') as never, 'tok-alice-new');

		authStore.finishSignIn('/generate');

		expect(location.assign).not.toHaveBeenCalled();
		expect(hoisted.goto).toHaveBeenCalled();
	});

	it('a token boot with legacy keys and no registry reloads once so the keys reach the user', async () => {
		localStorage.setItem('auth_token', 'tok-alice');
		localStorage.setItem(TABS_STORAGE_KEY, 'legacy-tabs');
		location.pathname = '/generate';
		(location as any).search = '?x=1';
		await loadAuth();

		expect(location.assign).toHaveBeenCalledWith('/generate?x=1');
	});
});

describe('authStore session bootstrap', () => {
	async function loadWithFakeTimers() {
		vi.useFakeTimers();
		vi.resetModules();
		const auth = await import('./auth');
		const accounts = await import('./accounts');
		const { api } = await import('$lib/services/api/index');
		await vi.advanceTimersByTimeAsync(0);
		return { ...auth, accounts, api };
	}

	beforeEach(async () => {
		const { api } = await import('$lib/services/api/index');
		vi.mocked(api.getCurrentUser).mockClear();
		vi.mocked(api.clearAuth).mockClear();
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it('keeps the token and retries /me with backoff while the backend is unreachable', async () => {
		seed('alice', ['alice']);
		hoisted.meOutcomes = [new Error('Network Error'), { response: { status: 502 } }];
		const { authStore, api, accounts } = await loadWithFakeTimers();

		let state = get(authStore);
		expect(state.isAuthenticated).toBe(true);
		expect(state.loading).toBe(true);
		expect(state.token).toBe('tok-alice');
		expect(api.getCurrentUser).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(1000);
		expect(api.getCurrentUser).toHaveBeenCalledTimes(2);
		expect(get(authStore).loading).toBe(true);

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getCurrentUser).toHaveBeenCalledTimes(3);
		state = get(authStore);
		expect(state.isAuthenticated).toBe(true);
		expect(state.loading).toBe(false);
		expect(state.user?.id).toBe('alice');
		expect(api.clearAuth).not.toHaveBeenCalled();
		expect(localStorage.getItem('auth_token')).toBe('tok-alice');
		expect(accounts.currentRegistry().accounts[0].expired).toBe(false);
		expect(hoisted.goto).not.toHaveBeenCalled();
	});

	it('keeps retrying when /me answers without a user instead of dropping the session', async () => {
		seed('alice', ['alice']);
		hoisted.meOutcomes = [{ success: false }];
		const { authStore, api } = await loadWithFakeTimers();

		expect(get(authStore).isAuthenticated).toBe(true);
		expect(get(authStore).loading).toBe(true);
		expect(api.clearAuth).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(1000);
		expect(get(authStore).loading).toBe(false);
		expect(get(authStore).user?.id).toBe('alice');
	});

	it('ends the session on a 401 and does not retry', async () => {
		seed('alice', ['alice']);
		hoisted.meOutcomes = [{ response: { status: 401 } }];
		const { authStore, api } = await loadWithFakeTimers();

		const state = get(authStore);
		expect(state.isAuthenticated).toBe(false);
		expect(state.loading).toBe(false);
		expect(hoisted.goto).toHaveBeenCalledWith('/login?expired=1');

		await vi.advanceTimersByTimeAsync(60000);
		expect(api.getCurrentUser).toHaveBeenCalledTimes(1);
	});

	it('stops retrying once the session was signed out meanwhile', async () => {
		seed('alice', ['alice']);
		hoisted.meOutcomes = [new Error('Network Error')];
		const { authStore, api } = await loadWithFakeTimers();

		authStore.logout();
		await vi.advanceTimersByTimeAsync(60000);

		expect(api.getCurrentUser).toHaveBeenCalledTimes(1);
	});
});

describe('authStore token renewal', () => {
	it('moves the active account, the stored token and the store state to the renewed token', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, accounts } = await loadAuth();

		hoisted.refreshedHandler.current!('tok-alice', 'tok-alice-renewed');

		expect(get(authStore).token).toBe('tok-alice-renewed');
		const registry = accounts.currentRegistry();
		expect(registry.accounts.find((a) => a.userId === 'alice')?.token).toBe('tok-alice-renewed');
		expect(registry.accounts.find((a) => a.userId === 'alice')?.expired).toBe(false);
		expect(registry.accounts.find((a) => a.userId === 'bob')?.token).toBe('tok-bob');
		expect(localStorage.getItem('auth_token')).toBe('tok-alice-renewed');
	});

	it('leaves the store token alone when the renewed token belongs to another session', async () => {
		seed('alice', ['alice', 'bob']);
		const { authStore, accounts } = await loadAuth();

		hoisted.refreshedHandler.current!('tok-bob', 'tok-bob-renewed');

		expect(get(authStore).token).toBe('tok-alice');
		expect(accounts.currentRegistry().accounts.find((a) => a.userId === 'bob')?.token).toBe('tok-bob-renewed');
	});
});
