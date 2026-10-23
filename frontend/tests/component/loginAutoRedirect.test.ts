import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Writable } from 'svelte/store';
import { settle } from './helpers/settle';

type PageStore = Writable<{ url: URL }>;

vi.mock('$lib/services/api/index', () => ({
	api: {
		getSetupStatus: vi.fn(),
		getLoginProviders: vi.fn()
	}
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: {
		finishSignIn: vi.fn(),
		login: vi.fn(),
		subscribe: (fn: (v: unknown) => void) => {
			fn({ isAuthenticated: false, loading: false, error: null, user: null, token: null });
			return () => {};
		}
	}
}));

const { api } = await import('$lib/services/api/index');
const { authStore } = await import('$lib/stores/auth');
const page = (await import('$app/stores')).page as unknown as PageStore;
const { default: LoginPage } = await import('../../src/routes/login/+page.svelte');
const { createClassComponent } = await import('svelte/legacy');

const SSO = [
	{ id: 'acme-sso', label: 'Acme SSO', start_path: '/api/plugins/acme-sso/start', auto_redirect: true }
];
const PLAIN = [{ ...SSO[0], auto_redirect: false }];

let mounted: { target: HTMLElement; destroy: () => void } | undefined;
let assigned: string[];
let originalLocation: PropertyDescriptor | undefined;

function visit(search: string) {
	page.set({ url: new URL(`http://localhost/login${search}`) });
}

function mountPage() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: LoginPage as never, target, props: {} });
	mounted = {
		target,
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
	return mounted;
}

function hasLocalForm(target: HTMLElement) {
	return target.querySelector('#remember-me') !== null && target.querySelector('form') !== null;
}

beforeEach(() => {
	sessionStorage.clear();
	assigned = [];
	originalLocation = Object.getOwnPropertyDescriptor(window, 'location');
	Object.defineProperty(window, 'location', {
		configurable: true,
		value: {
			get href() {
				return 'http://localhost/login';
			},
			set href(value: string) {
				assigned.push(value);
			}
		}
	});
	vi.mocked(api.getSetupStatus).mockResolvedValue(null as never);
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	if (originalLocation) Object.defineProperty(window, 'location', originalLocation);
	vi.clearAllMocks();
});

describe('login page auto-redirect', () => {
	it('goes straight to the provider and never shows the local form when on', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('');

		mountPage();
		await settle(10);

		expect(assigned).toEqual(['/api/plugins/acme-sso/start']);
		expect(hasLocalForm(mounted!.target)).toBe(false);
		expect(mounted!.target.textContent).toContain('Redirecting to Acme SSO');
	});

	it('shows the local form and the button when the flag is off', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(PLAIN as never);
		visit('');

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
		expect(hasLocalForm(mounted!.target)).toBe(true);
	});

	it('local=1 shows the local form and does not redirect', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?local=1');

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
		expect(hasLocalForm(mounted!.target)).toBe(true);
	});

	it('does not bounce back into the provider after an SSO error', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?error=external_login');

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
		expect(mounted!.target.textContent).toContain('Single sign-on did not complete');
		expect(mounted!.target.textContent).toContain('Continue with Acme SSO');
	});

	it('shows a signed-out state with a sign-in button instead of redirecting after logout', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?signedout=1');

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
		expect(mounted!.target.textContent).toContain('You have been signed out');
		const button = Array.from(mounted!.target.querySelectorAll('button')).find((b) =>
			b.textContent?.includes('Continue with Acme SSO')
		);
		expect(button).toBeTruthy();
		button!.click();
		await settle(5);
		expect(assigned).toEqual(['/api/plugins/acme-sso/start']);
	});

	it('does not redirect twice within the guard window', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('');

		mountPage();
		await settle(10);
		mounted!.destroy();
		assigned.length = 0;

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
	});

	it('does not redirect an expired session', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?expired=1');

		mountPage();
		await settle(10);

		expect(assigned).toEqual([]);
	});

	it('keeps a safe return path for the callback and drops an unsafe one', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?next=%2Fhistory%3Fid%3D7');

		mountPage();
		await settle(10);

		expect(sessionStorage.getItem('post_login_path')).toBe('/history?id=7');
		expect(assigned).toEqual(['/api/plugins/acme-sso/start']);
	});

	it('never stores an open-redirect return path', async () => {
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('?next=%2F%2Fevil.example');

		mountPage();
		await settle(10);

		expect(sessionStorage.getItem('post_login_path')).toBeNull();
	});

	it('clears a stale saved path when the page is opened without next', async () => {
		sessionStorage.setItem('post_login_path', '/a');
		vi.mocked(api.getLoginProviders).mockResolvedValue(SSO as never);
		visit('');

		mountPage();
		await settle(10);

		expect(sessionStorage.getItem('post_login_path')).toBeNull();
	});
});

describe('login page password sign-in return path', () => {
	async function signIn(search: string) {
		vi.mocked(api.getLoginProviders).mockResolvedValue(PLAIN as never);
		vi.mocked(authStore.login).mockResolvedValue({ success: true } as never);
		visit(search);
		mountPage();
		await settle(10);
		const target = mounted!.target;
		const username = target.querySelector('#username') as HTMLInputElement;
		const password = target.querySelector('#password') as HTMLInputElement;
		username.value = 'someuser';
		username.dispatchEvent(new Event('input', { bubbles: true }));
		password.value = 'secret';
		password.dispatchEvent(new Event('input', { bubbles: true }));
		target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
		await settle(10);
	}

	it('lands on the sanitised next', async () => {
		await signIn('?next=%2Fhistory%3Fgeneration%3Dabc');
		expect(authStore.finishSignIn).toHaveBeenCalledWith('/history?generation=abc');
	});

	it('lands on /generate without next', async () => {
		await signIn('');
		expect(authStore.finishSignIn).toHaveBeenCalledWith('/generate');
	});

	it.each(['%2F%2Fevil.com', '%2F.%2F%2Fevil.com', 'https%3A%2F%2Fevil.com'])(
		'lands on /generate for unsafe next %s',
		async (next) => {
			await signIn(`?next=${next}`);
			expect(authStore.finishSignIn).toHaveBeenCalledWith('/generate');
		}
	);
});
