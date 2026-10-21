import { describe, it, expect, vi } from 'vitest';
import { get, writable } from 'svelte/store';
import { createSessionDialogs } from './sessionDialogs';
import type { SessionController } from './sessionController';

function fakeController(currentName: string | null, overrides: Partial<Record<string, unknown>> = {}) {
	const state = writable({ currentSession: currentName ? { id: 's1', name: currentName } : null });
	const controller = {
		state,
		openDialog: vi.fn(),
		closeDialog: vi.fn(),
		clearNameError: vi.fn(),
		clearFeedback: vi.fn(),
		quickSave: vi.fn().mockResolvedValue(true),
		saveAs: vi.fn().mockResolvedValue(true),
		deleteSession: vi.fn().mockResolvedValue(true),
		select: vi.fn(),
		startNew: vi.fn(),
		togglePin: vi.fn(),
		toggleAutosave: vi.fn(),
		setAutosaveInterval: vi.fn(),
		openHistory: vi.fn(),
		closeHistory: vi.fn(),
		restoreVersion: vi.fn(),
		...overrides
	};
	return controller as unknown as SessionController & typeof controller;
}

describe('createSessionDialogs', () => {
	it('opens save-as with an empty name and declares the dialog', () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openSaveAs();
		expect(get(dialogs.state)).toMatchObject({ saveAsOpen: true, saveOpen: false, name: '' });
		expect(controller.openDialog).toHaveBeenCalledWith('save');
		expect(controller.clearNameError).toHaveBeenCalled();
	});

	it('opens rename prefilled with the current session name', () => {
		const dialogs = createSessionDialogs(fakeController('Alpha'));
		dialogs.openSave();
		expect(get(dialogs.state)).toMatchObject({ saveOpen: true, saveAsOpen: false, name: 'Alpha' });
	});

	it('falls back to save-as when there is no current session', () => {
		const dialogs = createSessionDialogs(fakeController(null));
		dialogs.openSave();
		expect(get(dialogs.state)).toMatchObject({ saveOpen: false, saveAsOpen: true });
	});

	it('opens the delete confirm and declares it', () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openDelete();
		expect(get(dialogs.state).deleteOpen).toBe(true);
		expect(controller.openDialog).toHaveBeenCalledWith('delete');
	});

	it('tells the controller about a cancel so a pending save cannot answer later', () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openSave();
		dialogs.close();
		expect(get(dialogs.state)).toEqual({ saveOpen: false, saveAsOpen: false, deleteOpen: false, name: '' });
		expect(controller.closeDialog).toHaveBeenCalledWith('save');
		expect(controller.closeDialog).toHaveBeenCalledWith('delete');
		expect(controller.clearFeedback).toHaveBeenCalled();
	});

	it('saves under the typed name and closes only on success', async () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openSaveAs();
		dialogs.state.update((s) => ({ ...s, name: 'Beta' }));
		controller.saveAs.mockResolvedValueOnce(false);
		await dialogs.confirmSave(true);
		expect(controller.saveAs).toHaveBeenCalledWith('Beta', 'save-as');
		expect(get(dialogs.state).saveAsOpen).toBe(true);
		await dialogs.confirmSave(true);
		expect(get(dialogs.state).saveAsOpen).toBe(false);
	});

	it('uses rename mode for the rename dialog', async () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openSave();
		await dialogs.confirmSave(false);
		expect(controller.saveAs).toHaveBeenCalledWith('Alpha', 'rename');
	});

	it('keeps the delete confirm open when the delete fails', async () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		dialogs.openDelete();
		controller.deleteSession.mockResolvedValueOnce(false);
		await dialogs.confirmDelete();
		expect(get(dialogs.state).deleteOpen).toBe(true);
		await dialogs.confirmDelete();
		expect(get(dialogs.state).deleteOpen).toBe(false);
		expect(controller.closeDialog).toHaveBeenCalledWith('delete');
	});

	it('quick save opens save-as when there is nothing to quick-save', async () => {
		const controller = fakeController(null, { quickSave: vi.fn().mockResolvedValue(false) });
		const dialogs = createSessionDialogs(controller);
		await dialogs.quickSave();
		expect(get(dialogs.state).saveAsOpen).toBe(true);
	});

	it('exposes controller actions plus the dialog-backed ones', () => {
		const controller = fakeController('Alpha');
		const dialogs = createSessionDialogs(controller);
		expect(dialogs.actions.select).toBe(controller.select);
		expect(dialogs.actions.saveAs).toBe(dialogs.openSaveAs);
		expect(dialogs.actions.rename).toBe(dialogs.openSave);
		expect(dialogs.actions.remove).toBe(dialogs.openDelete);
		expect(dialogs.actions.save).toBe(dialogs.quickSave);
	});
});
