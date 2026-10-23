import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';
import { writable, type Writable } from 'svelte/store';

const USER = { id: 'u1', username: 'Ember', email: 'ember@example.test', account_type: 'USER' };

vi.mock('$lib/plugin-api/extensionRefresh', () => ({
	refreshPluginExtensions: vi.fn(() => Promise.resolve())
}));

vi.mock('$lib/stores/auth', async () => {
	const { writable: store } = await import('svelte/store');
	return {
		authStore: store({
			isAuthenticated: true,
			token: null,
			user: { id: 'u1', username: 'Ember', email: 'ember@example.test', account_type: 'USER' },
			loading: false,
			error: null
		})
	};
});

vi.mock('$lib/stores/keybindings', () => ({
	keybindingsStore: {
		subscribe: writable({ bindings: [], helpPanelOpen: false, loaded: false }).subscribe,
		registerHandler: vi.fn(),
		unregisterHandler: vi.fn(),
		openHelp: vi.fn(),
		toggleHelp: vi.fn()
	},
	shortcutLabels: writable<Record<string, string | undefined>>({})
}));

vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: () => ({ get: vi.fn().mockResolvedValue({ data: { success: true, data: {} } }) }),
		getBaseURL: () => '',
		getToken: () => null,
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn(),
		getReadiness: vi.fn().mockResolvedValue({ overall: 'ready' }),
		getKeybindings: vi.fn().mockResolvedValue({ success: true, data: { keybindings: [] } })
	}
}));

vi.mock('$lib/services/admin-api', () => ({
	getBackends: vi.fn().mockResolvedValue({ success: true, data: [] }),
	invokeBackendQuickAction: vi.fn()
}));

const { authStore } = await import('$lib/stores/auth');
const auth = authStore as unknown as Writable<{ user: Record<string, unknown> | null }>;
const { default: Sidebar } = await import('$lib/components/Sidebar.svelte');
const { createClassComponent } = await import('svelte/legacy');

let component: ReturnType<typeof createClassComponent> | null = null;
let target: HTMLDivElement;

function mount() {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: Sidebar as never, target, props: {} });
	flushSync();
	return target.querySelector('aside')!;
}

afterEach(() => {
	component?.$destroy();
	component = null;
	target?.remove();
	auth.update((state) => ({ ...state, user: USER }));
});

describe('Sidebar user band', () => {
	it('closes the rail with a deck-height band holding the avatar', () => {
		const aside = mount();
		const band = aside.querySelector<HTMLElement>('[data-rail-user]')!;
		expect(band).not.toBeNull();
		expect(band).toBe(aside.lastElementChild);
		expect(band.className.split(/\s+/)).toEqual(
			expect.arrayContaining(['h-[var(--dock-height,92px)]', 'flex-shrink-0', 'border-t', 'border-line', 'items-center', 'justify-center'])
		);
		expect(band.querySelector('button')).not.toBeNull();
	});

	it('keeps the other rail actions in the section above the band', () => {
		const aside = mount();
		const band = aside.querySelector<HTMLElement>('[data-rail-user]')!;
		const above = band.previousElementSibling as HTMLElement;
		expect(above.contains(band)).toBe(false);
		expect(above.querySelector('[aria-label="Keyboard shortcuts"]')).not.toBeNull();
		expect(above.querySelector('[data-rail-user]')).toBeNull();
	});

	it('renders no band without a signed-in user', () => {
		auth.update((state) => ({ ...state, user: null }));
		const aside = mount();
		expect(aside.querySelector('[data-rail-user]')).toBeNull();
	});
});
