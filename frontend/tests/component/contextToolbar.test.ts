import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { mount, unmount, flushSync, tick } from 'svelte';
import { get } from 'svelte/store';
import { settle } from './helpers/settle';

const { authState, getPreset, declaration } = vi.hoisted(() => ({
	authState: { user: { account_type: 'USER' } as { account_type: string } | null },
	getPreset: vi.fn(),
	declaration: { current: { fields: ['seed'] } as unknown }
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: { subscribe: (fn: (v: unknown) => void) => (fn(authState), () => {}) }
}));

vi.mock('$lib/services/api/index', async (orig) => {
	const real: any = await orig();
	return { ...real, api: { ...real.api, getPreset, getClient: () => ({}) } };
});

vi.mock('$lib/stores/formulas', async (orig) => {
	const real: any = await orig();
	const { readable } = await import('svelte/store');
	return {
		...real,
		formulaLists: readable({ 'krea2 txt2img': { items: [{}, {}, {}], loading: false, loaded: true, error: null } }),
		getFormulaDeclaration: vi.fn(async () => declaration.current),
		loadFormulas: vi.fn(async () => {})
	};
});

const { default: ContextToolbar } = await import('../../src/routes/generate/components/ContextToolbar.svelte');
const { tabsStore } = await import('$lib/stores/tabs');
const { formAudienceStore } = await import('$lib/stores/formAudience');
const { reactiveProps } = await import('./stubs/reactiveProps.svelte');

const PRESETS = [{ id: 'krea2', name: 'Krea-2', version: '1.1.0', engine: 'native', category: 'image', tags: [], description: '' }];
const MODES = [
	{ id: 'txt2img', label: 'Text to Image', description: 'Generate an image from a text prompt.' },
	{ id: 'edit', label: 'Edit' }
];
const VARIANT_MODES = [
	{
		id: 'txt2img',
		label: 'Text to Image',
		variants: [
			{ name: 'image', label: 'Image', order: 0, default: true },
			{ name: 'video', label: 'Video', order: 1 }
		]
	},
	{ id: 'edit', label: 'Edit' }
];

function sessionState(overrides: Record<string, unknown> = {}) {
	const stamp = new Date(Date.now() - 120_000).toISOString();
	const current = { id: 'a', preset_id: 'krea2', name: 'Neon alley', data: {}, created_at: stamp, updated_at: stamp };
	return {
		sessions: [current],
		currentSession: current,
		selectedSessionId: 'a',
		isSessionLoading: false,
		isQuickSaving: false,
		hasUnsavedChanges: false,
		lastSavedTime: new Date(),
		sessionControlsEnabled: true,
		autoSaveEnabled: false,
		autoSaveInterval: 10000,
		historySessionId: null,
		historyVersions: [],
		isHistoryLoading: false,
		historyError: null,
		isRestoringVersion: false,
		...overrides
	};
}

function sessionActions() {
	return {
		select: vi.fn(),
		save: vi.fn(),
		saveAs: vi.fn(),
		startNew: vi.fn(),
		rename: vi.fn(),
		remove: vi.fn(),
		togglePin: vi.fn(),
		toggleAutosave: vi.fn(),
		setAutosaveInterval: vi.fn(),
		openHistory: vi.fn(),
		closeHistory: vi.fn(),
		restoreVersion: vi.fn()
	};
}

class FakeResizeObserver {
	cb: () => void;
	constructor(cb: () => void) {
		this.cb = cb;
	}
	observe() {
		queueMicrotask(() => this.cb());
	}
	unobserve() {}
	disconnect() {}
}

const originalRO = (globalThis as any).ResizeObserver;
const clientDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
const scrollDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollWidth');

type Box = { client: number; scroll: (el: HTMLElement) => number };
let boxes: Record<string, Box> = {};

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;
let tabId = '';
let stopTabSync: (() => void) | null = null;

function tab() {
	return get(tabsStore).tabs.find((t) => t.id === tabId)!;
}

function measure(next: Record<string, Box>) {
	boxes = next;
}

function installMeasure() {
	Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
		configurable: true,
		get() {
			return boxes[this.dataset?.testid ?? '']?.client ?? 0;
		}
	});
	Object.defineProperty(Element.prototype, 'scrollWidth', {
		configurable: true,
		get() {
			const box = boxes[(this as HTMLElement).dataset?.testid ?? ''];
			return box ? box.scroll(this as HTMLElement) : 0;
		}
	});
}

function count(el: HTMLElement, selector: string): number {
	return el.querySelectorAll(selector).length;
}

function barContent(bar: HTMLElement): number {
	return 400 + 100 * count(bar, '[data-testid="formulas-button"]') + 100 * count(bar, '[data-context-tool]') + 40 * count(bar, '[data-testid="context-tools-more"]');
}

function stagedContent(bar: HTMLElement): number {
	const atMinimum = count(bar, '[data-testid="preset-picker-trigger"].max-w-\\[96px\\]') > 0;
	const capped = count(bar, '[data-testid="preset-picker-trigger"].max-w-\\[160px\\]') > 0;
	const labelsHidden = count(bar, '[data-context-label].sr-only') > 0;
	return 300 + 100 * count(bar, '[data-testid="formulas-button"]') + 100 * count(bar, '[data-context-tool]') + 40 * count(bar, '[data-testid="context-tools-more"]') + (atMinimum ? 0 : capped ? 75 : 150) + (labelsHidden ? 0 : 120);
}

beforeEach(() => {
	authState.user = { account_type: 'USER' };
	declaration.current = { fields: ['seed'] };
	getPreset.mockResolvedValue({ success: true, data: { media: { gallery: [] } } });
	formAudienceStore.set('simple');
	(globalThis as any).ResizeObserver = FakeResizeObserver;
	boxes = {};
	installMeasure();
	tabId = tabsStore.addTabWithData('Toolbar tab', { selectedPreset: 'krea2', selectedMode: 'txt2img', layoutMode: 'two' });
});

afterEach(() => {
	stopTabSync?.();
	stopTabSync = null;
	if (component) unmount(component);
	component = null;
	target?.remove();
	tabsStore.removeTab(tabId);
	document.body.innerHTML = '';
	(globalThis as any).ResizeObserver = originalRO;
	if (clientDescriptor) Object.defineProperty(HTMLElement.prototype, 'clientWidth', clientDescriptor);
	if (scrollDescriptor) Object.defineProperty(Element.prototype, 'scrollWidth', scrollDescriptor);
	else delete (Element.prototype as any).scrollWidth;
});

async function render(overrides: Record<string, unknown> = {}) {
	const handlers = {
		onPresetChange: vi.fn(),
		onModeChange: vi.fn(),
		onVariantChange: vi.fn(),
		onReload: vi.fn()
	};
	const actions = sessionActions();
	target = document.createElement('div');
	document.body.appendChild(target);
	const live = reactiveProps({
		tab: tab(),
		presets: PRESETS,
		availableModes: MODES,
		session: sessionState(),
		actions,
		...handlers,
		...overrides
	} as Record<string, unknown>);
	stopTabSync?.();
	stopTabSync = tabsStore.subscribe(() => {
		const current = tab();
		if (current) live.set({ tab: current });
	});
	component = mount(ContextToolbar as never, { target, props: live.props } as never);
	flushSync();
	await settle(8);
	flushSync();
	return { ...handlers, actions };
}

function bar() {
	return target.querySelector('[data-testid="context-bar"]') as HTMLElement;
}

function labels() {
	return Array.from(bar().querySelectorAll<HTMLElement>('[data-context-label]')).map((el) => el.textContent?.trim());
}

function toolOrder() {
	return Array.from(
		target.querySelectorAll<HTMLElement>('[data-testid="context-tools"] [data-testid="formulas-button"], [data-testid="context-tools"] [data-context-tool], [data-testid="context-tools-more"]')
	).map((el) => el.dataset.contextTool ?? (el.dataset.testid === 'formulas-button' ? 'formulas' : 'more'));
}

async function openMore() {
	(target.querySelector('[data-testid="context-tools-more"]') as HTMLElement).click();
	flushSync();
	await tick();
	return Array.from(document.body.querySelectorAll<HTMLElement>('[role="menu"][aria-label="More tools"] button'));
}

function nameOf(el: Element): string {
	const ids = el.getAttribute('aria-labelledby');
	if (!ids) return el.getAttribute('aria-label') ?? '';
	return ids
		.split(' ')
		.map((id) => document.getElementById(id)?.textContent?.trim() ?? '')
		.join(' ');
}

function saveButton() {
	return target.querySelector('[data-testid="session-save"]') as HTMLButtonElement;
}

describe('ContextToolbar background', () => {
	it('paints one surface on the bar root and leaves every zone transparent', async () => {
		await render();
		const root = bar();
		expect(root.classList.contains('bg-surface-1')).toBe(true);
		const zones = Array.from(root.children) as HTMLElement[];
		expect(zones.map((zone) => zone.dataset.testid)).toEqual(['context-zone', 'context-gap', 'context-tools']);
		for (const zone of zones) {
			expect(Array.from(zone.classList).filter((token) => token.startsWith('bg-')), zone.dataset.testid).toEqual([]);
		}
		expect(root.querySelector('[data-testid="context-zone"]')!.classList.contains('border-r')).toBe(true);
	});
});

describe('ContextToolbar control styles', () => {
	it('draws context controls as ghost segments, Save as a filled pill and tools as bordered buttons', async () => {
		await render({ session: sessionState({ hasUnsavedChanges: true }) });
		const preset = target.querySelector('[data-testid="preset-picker-trigger"]') as HTMLElement;
		const mode = target.querySelector('[data-testid="preset-header-mode"] button') as HTMLElement;
		const session = target.querySelector('button[aria-label="Session"]') as HTMLElement;
		for (const control of [preset, mode, session]) {
			expect(control.classList.contains('bg-surface-3')).toBe(true);
			expect(control.classList.contains('hover:bg-line-hover')).toBe(true);
			expect(control.classList.contains('border-line-strong')).toBe(false);
		}
		expect(preset.classList.contains('text-md')).toBe(true);
		expect(preset.classList.contains('font-semibold')).toBe(true);
		expect(mode.classList.contains('font-medium')).toBe(true);
		mode.click();
		await tick();
		flushSync();
		expect(mode.getAttribute('aria-expanded')).toBe('true');
		expect(mode.classList.contains('bg-line-hover')).toBe(true);
		expect(mode.classList.contains('bg-surface-3')).toBe(false);
		expect(mode.className).not.toMatch(/segment-open|shadow-|border-b|border-signal/);
		expect(mode.querySelector('svg:last-child')!.getAttribute('class')).toContain('rotate-180');
		session.click();
		flushSync();
		expect(session.getAttribute('aria-expanded')).toBe('true');
		expect(session.classList.contains('bg-line-hover')).toBe(true);
		expect(session.className).not.toMatch(/segment-open|shadow-|border-signal/);
		const save = saveButton();
		expect(['h-7', 'rounded', 'bg-surface-3', 'text-accent', 'font-semibold'].every((token) => save.classList.contains(token))).toBe(true);
		for (const tool of Array.from(target.querySelectorAll<HTMLElement>('[data-context-tool], [data-testid="formulas-button"], [data-testid="context-tools-more"]'))) {
			expect(['h-8', 'border', 'border-line', 'bg-surface-2'].every((token) => tool.classList.contains(token)), tool.textContent ?? '').toBe(true);
		}
	});

	it('greys a clean Save on a half-filled pill', async () => {
		await render();
		const save = saveButton();
		expect(['bg-surface-3/50', 'text-fg-disabled', 'cursor-not-allowed'].every((token) => save.classList.contains(token))).toBe(true);
	});
});

describe('ContextToolbar context group', () => {
	it('puts each label left of its control and names each control by its label and value', async () => {
		await render();
		expect(labels()).toEqual(['Preset', 'Mode', 'Session']);
		for (const label of Array.from(bar().querySelectorAll<HTMLElement>('[data-context-label]'))) {
			const control = label.nextElementSibling as HTMLElement;
			expect(control.contains(label)).toBe(false);
			expect(label.classList.contains('text-sm')).toBe(true);
			expect(label.parentElement!.classList.contains('items-center')).toBe(true);
		}
		const preset = target.querySelector('[data-testid="preset-picker-trigger"]') as HTMLElement;
		expect(nameOf(preset)).toBe('Preset Krea-2');
		expect(preset.classList.contains('h-8')).toBe(true);
		const mode = target.querySelector('[data-testid="preset-header-mode"] button') as HTMLElement;
		expect(nameOf(mode)).toBe('Mode Text to Image');
		expect(mode.classList.contains('h-8')).toBe(true);
		const session = target.querySelector('button[aria-label="Session"]') as HTMLElement;
		expect(session.textContent).toContain('Neon alley');
		expect(target.querySelector('[data-testid="context-session"]')!.classList.contains('h-8')).toBe(true);
		expect(target.querySelector('[data-testid="preset-variant-switch"]')).toBeNull();
	});

	it('changes mode from the Mode control list', async () => {
		const { onModeChange } = await render();
		(target.querySelector('[data-testid="preset-header-mode"] button') as HTMLElement).click();
		await tick();
		flushSync();
		const options = Array.from(document.body.querySelectorAll<HTMLElement>('[role="option"]'));
		options.find((o) => o.textContent?.includes('Edit'))!.click();
		expect(onModeChange).toHaveBeenCalledWith('edit');
	});

	it('shows Variant only when the mode has variants and picks one', async () => {
		const { onVariantChange } = await render({ availableModes: VARIANT_MODES });
		expect(labels()).toEqual(['Preset', 'Mode', 'Variant', 'Session']);
		const variant = target.querySelector('[data-testid="preset-variant-switch"] button') as HTMLElement;
		expect(nameOf(variant)).toBe('Variant Image');
		variant.click();
		await tick();
		flushSync();
		Array.from(document.body.querySelectorAll<HTMLElement>('[role="option"]'))
			.find((o) => o.textContent?.includes('Video'))!
			.click();
		expect(onVariantChange).toHaveBeenCalledWith('video');
	});

	it('shows a preset call to action and an unavailable session before a preset is chosen', async () => {
		tabsStore.updateTab(tabId, { selectedPreset: null, selectedMode: null });
		await render({
			availableModes: [],
			session: sessionState({ sessionControlsEnabled: false, currentSession: null, selectedSessionId: '' })
		});
		const preset = target.querySelector('[data-testid="preset-picker-trigger"]') as HTMLElement;
		expect(nameOf(preset)).toBe('Preset Choose a preset');
		expect(preset.getAttribute('aria-haspopup')).toBe('dialog');
		expect(preset.classList.contains('preset-picker-cta')).toBe(true);
		expect(saveButton().getAttribute('aria-disabled')).toBe('true');
		expect(saveButton().getAttribute('aria-label')).toBe('Session save unavailable');
	});
});

describe('ContextToolbar session Save', () => {
	it('is always present at the same width and only enabled when there is something to save', async () => {
		const widths: string[] = [];
		const states = [
			{ overrides: {}, disabled: 'true', label: 'Session saved' },
			{ overrides: { hasUnsavedChanges: true }, disabled: null, label: 'Save session' },
			{ overrides: { currentSession: null, selectedSessionId: '' }, disabled: null, label: 'Save as a new session' },
			{ overrides: { isQuickSaving: true, hasUnsavedChanges: true }, disabled: null, label: 'Saving session' },
			{ overrides: { sessionControlsEnabled: false, currentSession: null }, disabled: 'true', label: 'Session save unavailable' }
		];
		for (const state of states) {
			const { actions } = await render({ session: sessionState(state.overrides) });
			const save = saveButton();
			expect(save, state.label).not.toBeNull();
			expect(save.getAttribute('aria-label')).toBe(state.label);
			expect(save.getAttribute('aria-disabled')).toBe(state.disabled);
			expect(target.querySelector('[data-testid="context-session"]')!.contains(save)).toBe(true);
			widths.push(save.className.split(/\s+/).filter((token) => token.startsWith('w-')).join(' '));
			save.click();
			const called = actions.save.mock.calls.length + actions.saveAs.mock.calls.length;
			expect(called, state.label).toBe(state.disabled ? 0 : 1);
			stopTabSync?.();
			stopTabSync = null;
			unmount(component!);
			component = null;
			target.remove();
		}
		expect(new Set(widths)).toEqual(new Set(['w-[78px]']));
	});
});

describe('ContextToolbar tools', () => {
	it('carries the six real tools: three on the bar, the window toggles in More', async () => {
		await render();
		expect(toolOrder()).toEqual(['formulas', 'fields', 'layout', 'more']);
		expect(target.querySelector('[data-testid="formulas-count"]')?.textContent).toBe('3');
		const items = await openMore();
		expect(items.map((item) => item.getAttribute('aria-label'))).toEqual(['Float form', 'Float workbench', 'Fold form']);
		expect(items.map((item) => item.getAttribute('role'))).toEqual(['menuitemcheckbox', 'menuitemcheckbox', 'menuitemcheckbox']);
		expect(items.map((item) => item.getAttribute('aria-checked'))).toEqual(['false', 'false', 'false']);
	});

	it('toggles field visibility and the pane layout', async () => {
		await render();
		const fields = target.querySelector('[data-context-tool="fields"]') as HTMLElement;
		expect(fields.getAttribute('aria-label')).toBe('Field visibility: Simple');
		expect(fields.getAttribute('aria-pressed')).toBe('false');
		fields.click();
		flushSync();
		expect(get(formAudienceStore)).toBe('advanced');
		expect(fields.getAttribute('aria-label')).toBe('Field visibility: Advanced');
		expect(fields.getAttribute('aria-pressed')).toBe('true');
		(target.querySelector('[data-context-tool="layout"]') as HTMLElement).click();
		expect(tab().layoutMode).toBe('three');
	});

	it('floats the form, floats the workbench and folds the form from More', async () => {
		await render();
		(await openMore()).find((item) => item.getAttribute('aria-label') === 'Float form')!.click();
		expect(tab().formFloating).toBe(true);
		(await openMore()).find((item) => item.getAttribute('aria-label') === 'Float workbench')!.click();
		expect(tab().workbenchFloating).toBe(true);
		expect(tab().formFloating).toBe(false);
		(await openMore()).find((item) => item.getAttribute('aria-label') === 'Fold form')!.click();
		expect(tab().leftPanelCollapsed).toBe(true);
	});

	it('folds the lowest priority tools into More tools and keeps Formulas', async () => {
		measure({ 'context-bar': { client: 1000, scroll: (el) => barContent(el) + 300 } });
		await render();
		expect(toolOrder()).toEqual(['formulas', 'fields', 'more']);
		const items = await openMore();
		expect(items.map((item) => item.getAttribute('aria-label'))).toEqual(['Layout: Two panes', 'Float form', 'Float workbench', 'Fold form']);
		items[0].click();
		expect(tab().layoutMode).toBe('three');
	});

	it('folds tools when the context zone spills over even though the bar itself fits', async () => {
		measure({
			'context-bar': { client: 1200, scroll: () => 1200 },
			'context-zone': { client: 500, scroll: (zone) => 440 + 40 * count(zone.parentElement!, '[data-context-tool]') }
		});
		await render();
		expect(toolOrder()).toEqual(['formulas', 'fields', 'more']);
	});

	it('folds every tool but Formulas when there is no room and marks More while a folded tool is active', async () => {
		measure({ 'context-bar': { client: 1000, scroll: (el) => barContent(el) + 460 } });
		await render();
		expect(toolOrder()).toEqual(['formulas', 'more']);
		expect(target.querySelector('[data-testid="context-tools-more-active"]')).toBeNull();
		formAudienceStore.set('advanced');
		flushSync();
		await settle(8);
		flushSync();
		expect(toolOrder()).toEqual(['formulas', 'more']);
		expect(target.querySelector('[data-testid="context-tools-more-active"]')).not.toBeNull();
	});

	it('gives the tool slots to the other tools when the preset declares no formulas', async () => {
		declaration.current = null;
		await render();
		expect(toolOrder()).toEqual(['fields', 'layout', 'float-form', 'more']);
		expect(target.querySelector('[data-testid="formulas-button"]')).toBeNull();
	});
});

describe('ContextToolbar below 1000px', () => {
	it('folds every tool but Formulas when the bar is narrower than 944px, even with room to spare', async () => {
		measure({ 'context-bar': { client: 900, scroll: () => 600 } });
		await render();
		expect(toolOrder()).toEqual(['formulas', 'more']);
		expect(bar().querySelectorAll('[data-context-label].sr-only').length).toBe(0);
	});

	it('then caps preset and mode at 160px first, giving up only what the bar needs', async () => {
		measure({ 'context-bar': { client: 650, scroll: stagedContent } });
		await render();
		expect(toolOrder()).toEqual(['formulas', 'more']);
		expect(target.querySelector('[data-testid="preset-picker-trigger"]')!.classList.contains('max-w-[160px]')).toBe(true);
		expect(target.querySelector('[data-testid="preset-header-mode"] button')!.classList.contains('max-w-[160px]')).toBe(true);
		expect(bar().querySelectorAll('[data-context-label].sr-only').length).toBe(0);
	});

	it('then truncates preset and mode to their minimum before hiding the labels', async () => {
		measure({ 'context-bar': { client: 600, scroll: stagedContent } });
		await render();
		expect(toolOrder()).toEqual(['formulas', 'more']);
		expect(target.querySelector('[data-testid="preset-picker-trigger"]')!.classList.contains('max-w-[96px]')).toBe(true);
		expect(target.querySelector('[data-testid="preset-header-mode"] button')!.classList.contains('max-w-[96px]')).toBe(true);
		expect(target.querySelector('[data-testid="preset-header-mode"]')!.classList.contains('flex-shrink-0')).toBe(true);
		expect(bar().querySelectorAll('[data-context-label].sr-only').length).toBe(0);
	});

	it('hides the inline labels last and keeps every control named', async () => {
		measure({ 'context-bar': { client: 500, scroll: stagedContent } });
		await render();
		expect(Array.from(bar().querySelectorAll('[data-context-label]')).every((label) => label.classList.contains('sr-only'))).toBe(true);
		expect(nameOf(target.querySelector('[data-testid="preset-picker-trigger"]')!)).toBe('Preset Krea-2');
		expect(nameOf(target.querySelector('[data-testid="preset-header-mode"] button')!)).toBe('Mode Text to Image');
		expect(target.querySelector('button[aria-label="Session"]')).not.toBeNull();
	});
});
