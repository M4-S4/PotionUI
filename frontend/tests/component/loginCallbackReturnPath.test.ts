import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Writable } from 'svelte/store';
import { settle } from './helpers/settle';

type PageStore = Writable<{ url: URL }>;

vi.mock('$lib/services/api/index', () => ({
	api: {
		exchangeExternalLoginCode: vi.fn()
	}
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: {
		adoptToken: vi.fn(),
		finishSignIn: vi.fn()
	}
}));

const { api } = await import('$lib/services/api/index');
const { authStore } = await import('$lib/stores/auth');
const page = (await import('$app/stores')).page as unknown as PageStore;
const { default: CallbackPage } = await import('../../src/routes/login/callback/+page.svelte');
const { createClassComponent } = await import('svelte/legacy');

let destroy: (() => void) | undefined;

function mountCallback() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: CallbackPage as never, target, props: {} });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
}

beforeEach(() => {
	sessionStorage.clear();
	page.set({ url: new URL('http://localhost/login/callback?code=abc') });
	vi.mocked(api.exchangeExternalLoginCode).mockResolvedValue({ access_token: 't', token_type: 'bearer' } as never);
	vi.mocked(authStore.adoptToken).mockResolvedValue({ ok: true } as never);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	vi.clearAllMocks();
});

describe('sign-in callback return path', () => {
	it('lands on the saved path', async () => {
		sessionStorage.setItem('post_login_path', '/history?id=7');
		mountCallback();
		await settle(10);

		expect(authStore.finishSignIn).toHaveBeenCalledWith('/history?id=7');
	});

	it('lands on the root when nothing was saved', async () => {
		mountCallback();
		await settle(10);

		expect(authStore.finishSignIn).toHaveBeenCalledWith('/');
	});

	it('ignores a tampered saved path', async () => {
		sessionStorage.setItem('post_login_path', '//evil.example');
		mountCallback();
		await settle(10);

		expect(authStore.finishSignIn).toHaveBeenCalledWith('/');
	});
});
