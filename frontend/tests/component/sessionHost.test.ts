// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { writable, type Writable } from 'svelte/store';
import { keybindingsStore } from '$lib/stores/keybindings';
import SessionHost from '$lib/components/session/SessionHost.svelte';
import { createSessionDialogs } from '$lib/session/sessionDialogs';
import type { SessionController } from '$lib/session/sessionController';

let instance: ReturnType<typeof mount> | null = null;

function fakeController() {
	return {
		state: writable({
			currentSession: { id: 's1', name: 'Alpha' },
			hasUnsavedChanges: true,
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
		deleteSession: vi.fn().mockResolvedValue(true),
		reload: vi.fn().mockResolvedValue(true)
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

	describe('reload confirm', () => {
		function confirmButton(): HTMLButtonElement {
			return Array.from(document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')).find((el) => /^Reload/.test(el.textContent?.trim() ?? ''))!;
		}

		it('asks to discard the changes and reload the last snapshot of the session by name', () => {
			const { dialogs } = mountHost();
			dialogs.openReload();
			flushSync();
			const dialog = document.querySelector('[role="alertdialog"]')!;
			expect(dialog.textContent).toContain('Discard your changes and reload the last snapshot of "Alpha"?');
			expect(confirmButton()).toBeTruthy();
		});

		it('reloads the snapshot once confirmed', async () => {
			const { controller, dialogs } = mountHost();
			dialogs.openReload();
			flushSync();
			confirmButton().click();
			flushSync();
			await vi.advanceTimersByTimeAsync(300);
			expect(controller.reload).toHaveBeenCalledTimes(1);
			expect(document.querySelector('[role="alertdialog"]')).toBeNull();
		});

		for (const flag of ['isQuickSaving', 'isSaving']) {
			it(`keeps the confirm busy while ${flag} is set, so the reload cannot outrun a save`, async () => {
				const { controller, dialogs } = mountHost();
				dialogs.openReload();
				flushSync();
				const dialog = document.querySelector('[role="alertdialog"]')!;
				expect(Array.from(dialog.querySelectorAll('button')).some((button) => button.disabled)).toBe(false);

				(controller.state as unknown as Writable<Record<string, unknown>>).update((state) => ({ ...state, [flag]: true }));
				flushSync();
				expect(Array.from(dialog.querySelectorAll('button')).every((button) => button.disabled)).toBe(true);
			});
		}

		it('keeps the edits when cancelled', async () => {
			const { controller, dialogs } = mountHost();
			dialogs.openReload();
			flushSync();
			document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
			flushSync();
			await vi.advanceTimersByTimeAsync(300);
			expect(document.querySelector('[role="alertdialog"]')).toBeNull();
			expect(controller.reload).not.toHaveBeenCalled();
		});
	});
});
