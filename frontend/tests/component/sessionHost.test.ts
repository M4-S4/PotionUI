// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { writable } from 'svelte/store';
import { keybindingsStore } from '$lib/stores/keybindings';
import SessionHost from '$lib/components/session/SessionHost.svelte';
import { createSessionDialogs } from '$lib/session/sessionDialogs';
import type { SessionController } from '$lib/session/sessionController';

let instance: ReturnType<typeof mount> | null = null;

function fakeController() {
	return {
		state: writable({
			currentSession: { id: 's1', name: 'Alpha' },
			nameError: '',
			error: null,
			isSaving: false,
			isSessionLoading: false
		}),
		openDialog: vi.fn(),
		closeDialog: vi.fn(),
		clearNameError: vi.fn(),
		clearFeedback: vi.fn(),
		quickSave: vi.fn().mockResolvedValue(true),
		saveAs: vi.fn().mockResolvedValue(true),
		deleteSession: vi.fn().mockResolvedValue(true)
	} as unknown as SessionController;
}

function mountHost(controller = fakeController()) {
	const dialogs = createSessionDialogs(controller);
	const target = document.createElement('div');
	document.body.appendChild(target);
	instance = mount(SessionHost, { target, props: { controller, dialogs } });
	flushSync();
	return { controller, dialogs };
}

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
	vi.restoreAllMocks();
	vi.useRealTimers();
});

describe('SessionHost', () => {
	it('registers the save_session handler on mount and removes it on destroy', () => {
		const register = vi.spyOn(keybindingsStore, 'registerHandler');
		const unregister = vi.spyOn(keybindingsStore, 'unregisterHandler');
		const { dialogs } = mountHost();
		expect(register).toHaveBeenCalledWith('save_session', dialogs.quickSave);
		unmount(instance!);
		instance = null;
		expect(unregister).toHaveBeenCalledWith('save_session');
	});

	it('shows the rename dialog prefilled when opened', () => {
		const { dialogs } = mountHost();
		dialogs.openSave();
		flushSync();
		const input = document.querySelector<HTMLInputElement>('#session-save-name');
		expect(input?.value).toBe('Alpha');
	});

	it('shows the delete confirm naming the session', () => {
		const { dialogs } = mountHost();
		dialogs.openDelete();
		flushSync();
		expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Alpha');
	});

	it('cancelling the delete confirm closes it and declares the cancel', async () => {
		const { controller, dialogs } = mountHost();
		dialogs.openDelete();
		flushSync();
		document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		flushSync();
		await vi.advanceTimersByTimeAsync(300);
		expect(document.querySelector('[role="alertdialog"]')).toBeNull();
		expect(controller.closeDialog).toHaveBeenCalledWith('delete');
	});
});
