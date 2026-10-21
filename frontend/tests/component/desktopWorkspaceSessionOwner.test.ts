import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import { flushSync } from 'svelte';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getSessionsForPreset: vi.fn(),
		getSessionById: vi.fn(),
		getSessionVersions: vi.fn(async () => ({ success: true, data: [] })),
		getPresetFormSchema: vi.fn(async () => ({ success: false, error: 'none' })),
		getFormulaDeclaration: vi.fn(async () => ({ success: true, data: null }))
	}
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: { subscribe: (fn: (v: unknown) => void) => (fn({ user: { account_type: 'USER' } }), () => {}) }
}));
vi.mock('../../src/routes/generate/components/GenerationFormPane.svelte', async () => ({
	default: (await import('./stubs/BlankStub.svelte')).default
}));
vi.mock('../../src/routes/generate/components/GenerationWorkbenchPane.svelte', async () => ({
	default: (await import('./stubs/BlankStub.svelte')).default
}));
vi.mock('../../src/routes/generate/components/PromptSection.svelte', async () => ({
	default: (await import('./stubs/BlankStub.svelte')).default
}));

const { api } = await import('$lib/services/api/index');
const { tabsStore } = await import('$lib/stores/tabs');
const { tabSessionDirty } = await import('$lib/stores/tabSessionDirty');
const { queryTabDirty } = await import('$lib/stores/workspaceDirtyQuery');
const { keybindingsStore } = await import('$lib/stores/keybindings');
const { collectTabSessionData } = await import('$lib/utils/sessionTabState');
const { default: DesktopWorkspaceFixture } = await import('./fixtures/DesktopWorkspaceFixture.svelte');
const { default: SessionScopeTabsFixture } = await import('./fixtures/SessionScopeTabsFixture.svelte');
const { createClassComponent } = await import('svelte/legacy');

const PRESET_ID = 'preset-owner';
const MODE = 'image';
const SESSION_ID = 'session-owner';
const ORIGINAL_SEGMENTS = [{ id: 'a', content: 'original segment' }];
const EDITED_SEGMENTS = [{ id: 'a', content: 'EDITED segment' }];

const session = {
	id: SESSION_ID,
	preset_id: PRESET_ID,
	name: 'Owner session',
	data: { [MODE]: { promptSegments: ORIGINAL_SEGMENTS } },
	created_at: '2026-01-01T00:00:00Z',
	updated_at: '2026-01-01T00:00:00Z'
};

let tabId: string;
let extraTabIds: string[] = [];
let mounted: { target: HTMLElement; component: { $set: (p: object) => void; $destroy: () => void } } | undefined;

function tab() {
	return get(tabsStore).tabs.find((t) => t.id === tabId)!;
}

function mountWorkspace() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: DesktopWorkspaceFixture as never,
		target,
		props: {
			tab: tab(),
			presets: [{ id: PRESET_ID, name: 'Owner preset', version: '1' }],
			availableModes: [{ id: MODE, label: 'Image' }]
		}
	}) as unknown as { $set: (p: object) => void; $destroy: () => void };
	flushSync();
	return { target, component };
}

function sessionSegment(target: HTMLElement) {
	return target.querySelector('[data-testid="context-bar"] button[aria-label="Session"]');
}

beforeEach(() => {
	const savedSessionSignature = JSON.stringify(
		collectTabSessionData({ promptSegments: ORIGINAL_SEGMENTS } as never, MODE, session.data)
	);
	tabId = tabsStore.addTabWithData('Owner tab', {
		selectedPreset: PRESET_ID,
		selectedMode: MODE,
		selectedSessionId: SESSION_ID,
		savedSessionSignature,
		promptSegments: EDITED_SEGMENTS,
		layoutMode: 'two'
	});
	vi.mocked(api.getSessionsForPreset).mockResolvedValue({ success: true, data: [session] } as never);
	vi.mocked(api.getSessionById).mockResolvedValue({ success: true, data: session } as never);
});

afterEach(() => {
	mounted?.component.$destroy();
	mounted?.target.remove();
	mounted = undefined;
	tabsStore.removeTab(tabId);
	for (const id of extraTabIds) tabsStore.removeTab(id);
	extraTabIds = [];
	vi.restoreAllMocks();
	document.body.innerHTML = '';
});

describe('Desktop workspace session owner', () => {
	it('puts the session in the context bar above the panes and publishes the unsaved state', async () => {
		mounted = mountWorkspace();
		await settle(10);
		flushSync();
		expect(sessionSegment(mounted.target)?.textContent).toContain('Owner session');
		expect(mounted.target.querySelector('[data-testid="context-bar"] button[aria-label="Save session"]')).not.toBeNull();
		expect(mounted.target.querySelector('[data-testid="generation-panels-root"] button[aria-label="Session"]')).toBeNull();
		expect(get(tabSessionDirty)[tabId]).toBe(true);
	});

	it('keeps the session in the bar while the form is folded and forgets the tab on unmount', async () => {
		mounted = mountWorkspace();
		await settle(10);
		tabsStore.updateTab(tabId, { leftPanelCollapsed: true });
		mounted.component.$set({ tab: tab() });
		flushSync();
		await settle(10);
		expect(mounted.target.querySelector('button[aria-label="Expand generation settings"]')).not.toBeNull();
		expect(sessionSegment(mounted.target)?.textContent).toContain('Owner session');
		expect(mounted.target.querySelector('[data-testid="context-bar"] button[aria-label="Save session"]')).not.toBeNull();
		expect(get(tabSessionDirty)[tabId]).toBe(true);
		expect(await queryTabDirty(tabId, 200)).toBe(true);
		mounted.component.$destroy();
		mounted.target.remove();
		mounted = undefined;
		expect(tabId in get(tabSessionDirty)).toBe(false);
	});
});

describe('Ctrl+S across a tab switch', () => {
	it('stays bound when the newly active tab sits before the old one', async () => {
		const first = tabsStore.addTabWithData('First tab', { selectedPreset: PRESET_ID, selectedMode: MODE });
		const second = tabsStore.addTabWithData('Second tab', { selectedPreset: PRESET_ID, selectedMode: MODE });
		extraTabIds = [first, second];
		const calls: string[] = [];
		const register = keybindingsStore.registerHandler.bind(keybindingsStore);
		const unregister = keybindingsStore.unregisterHandler.bind(keybindingsStore);
		vi.spyOn(keybindingsStore, 'registerHandler').mockImplementation((id, handler) => {
			if (id === 'save_session') calls.push('register');
			register(id, handler);
		});
		vi.spyOn(keybindingsStore, 'unregisterHandler').mockImplementation((id) => {
			if (id === 'save_session') calls.push('unregister');
			unregister(id);
		});
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: SessionScopeTabsFixture as never,
			target,
			props: { tabIds: [first, second], activeTabId: second, presetId: PRESET_ID, currentMode: MODE }
		}) as unknown as { $set: (p: object) => void; $destroy: () => void };
		mounted = { target, component };
		flushSync();
		await settle(5);
		expect(calls).toEqual(['register']);
		component.$set({ activeTabId: first });
		flushSync();
		await settle(5);
		expect(calls.at(-1)).toBe('register');
		expect(calls).toEqual(['register', 'unregister', 'register']);
	});
});
