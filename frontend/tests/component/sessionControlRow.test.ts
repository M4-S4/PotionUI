import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { get } from 'svelte/store';
import type { Session } from '$lib/types/api';

const { default: SessionControl } = await import('$lib/components/session/SessionControl.svelte');
const { sideDrawer } = await import('$lib/stores/sideDrawer');

const LONG_NAME = 'Portrait study, rim light, 85mm, overcast rooftop, second pass';

function makeSession(id: string, name: string): Session {
	const stamp = new Date(Date.now() - 120_000).toISOString();
	return { id, preset_id: 'p', name, data: {}, created_at: stamp, updated_at: stamp };
}

function state(overrides: Record<string, unknown> = {}) {
	const current = makeSession('a', 'Neon alley');
	return {
		sessions: [current, makeSession('b', 'Forest fog')],
		currentSession: current as Session | null,
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

function actions() {
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

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement;

function render(sessionOverrides: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const spies = actions();
	instance = mount(SessionControl as never, { target, props: { session: state(sessionOverrides), actions: spies } });
	flushSync();
	return spies;
}

function trigger(): HTMLButtonElement {
	return target.querySelector<HTMLButtonElement>('button[aria-label="Session"]')!;
}

function saveSegment(): HTMLButtonElement {
	return Array.from(target.querySelectorAll<HTMLButtonElement>('button')).find((el) => el !== trigger())!;
}

function dialog(): HTMLElement | null {
	return document.body.querySelector('[role="dialog"][aria-label="Sessions"]');
}

function pointerDown(el: Element) {
	el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));
	flushSync();
}

async function settle() {
	for (let i = 0; i < 6; i++) await new Promise((resolve) => setTimeout(resolve, 0));
	flushSync();
}

afterEach(async () => {
	if (instance) unmount(instance);
	instance = undefined;
	sideDrawer.close('sessions');
	sideDrawer.close('formulas');
	await settle();
	document.body.innerHTML = '';
});

describe('SessionControl save segment', () => {
	it('saves a draft that has no session yet as a new session', () => {
		const spies = render({ currentSession: null, selectedSessionId: '' });
		expect(saveSegment().getAttribute('aria-label')).toBe('Save as a new session');
		expect(saveSegment().textContent?.trim()).toBe('Save');
		expect(trigger().textContent).toContain('No saved session');
		saveSegment().click();
		expect(spies.saveAs).toHaveBeenCalledTimes(1);
		expect(spies.save).not.toHaveBeenCalled();
	});

	it('saves unsaved changes into the current session', () => {
		const spies = render({ hasUnsavedChanges: true });
		expect(saveSegment().getAttribute('aria-label')).toBe('Save session');
		saveSegment().click();
		expect(spies.save).toHaveBeenCalledTimes(1);
		expect(spies.saveAs).not.toHaveBeenCalled();
	});

	it('stays in place as a disabled Save while clean and reads Saving while a save runs', () => {
		const spies = render();
		expect(saveSegment().getAttribute('aria-label')).toBe('Session saved');
		expect(saveSegment().textContent?.trim()).toBe('Save');
		expect(saveSegment().getAttribute('aria-disabled')).toBe('true');
		saveSegment().click();
		expect(spies.save).not.toHaveBeenCalled();
		expect(spies.saveAs).not.toHaveBeenCalled();
		unmount(instance!);
		instance = undefined;
		target.remove();
		render({ isQuickSaving: true, hasUnsavedChanges: true });
		expect(saveSegment().getAttribute('aria-label')).toBe('Saving session');
		expect(saveSegment().textContent?.trim()).toBe('Saving');
	});

	it('disables the row until a preset and mode are chosen', () => {
		const spies = render({ sessionControlsEnabled: false, currentSession: null, selectedSessionId: '' });
		expect(saveSegment().getAttribute('aria-label')).toBe('Session save unavailable');
		expect(saveSegment().getAttribute('aria-disabled')).toBe('true');
		expect(trigger().disabled).toBe(true);
		expect(trigger().textContent).toContain('Session unavailable');
		trigger().click();
		flushSync();
		expect(dialog()).toBeNull();
		expect(spies.saveAs).not.toHaveBeenCalled();
	});
});

describe('SessionControl trigger', () => {
	it('keeps a long name on one truncated line and uses no native title', () => {
		render({ currentSession: makeSession('a', LONG_NAME) });
		const name = trigger().querySelector('span.truncate');
		expect(name?.textContent).toBe(LONG_NAME);
		expect(target.querySelectorAll('[title]').length).toBe(0);
	});

	it('opens the sessions dropdown, claims the side-drawer slot and closes on a second click', async () => {
		render();
		trigger().click();
		flushSync();
		expect(dialog()).not.toBeNull();
		expect(trigger().getAttribute('aria-expanded')).toBe('true');
		expect(get(sideDrawer)).toBe('sessions');
		pointerDown(trigger());
		trigger().click();
		flushSync();
		await settle();
		expect(dialog()).toBeNull();
		expect(get(sideDrawer)).toBeNull();
	});

	it('closes when Formulas takes the side-drawer slot', async () => {
		render();
		trigger().click();
		flushSync();
		sideDrawer.open('formulas');
		flushSync();
		await settle();
		expect(dialog()).toBeNull();
	});

	it('closes on a click outside but not on a click on its own Save segment', async () => {
		const spies = render({ hasUnsavedChanges: true });
		trigger().click();
		flushSync();
		pointerDown(saveSegment());
		saveSegment().click();
		flushSync();
		expect(spies.save).toHaveBeenCalledTimes(1);
		expect(dialog()).not.toBeNull();
		const outside = document.createElement('div');
		document.body.appendChild(outside);
		pointerDown(outside);
		await settle();
		expect(dialog()).toBeNull();
	});
});
