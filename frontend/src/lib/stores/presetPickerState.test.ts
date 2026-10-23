import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const calls = vi.hoisted(() => ({
	getPresetPickerState: vi.fn(),
	pinPreset: vi.fn(),
	unpinPreset: vi.fn(),
	pushRecentPreset: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({ api: calls }));

async function freshStore() {
	vi.resetModules();
	return import('./presetPickerState');
}

const ok = (pinned: string[], recent: string[]) => ({ success: true, data: { pinned, recent } });

describe('stores/presetPickerState', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		calls.getPresetPickerState.mockResolvedValue(ok(['seedvr2'], ['krea', 'wan']));
	});

	it('loads the pinned and recent presets from the server', async () => {
		const { presetPickerState } = await freshStore();
		expect(get(presetPickerState)).toEqual({ pinned: [], recent: [], loaded: false });
		await presetPickerState.load();
		expect(get(presetPickerState)).toEqual({ pinned: ['seedvr2'], recent: ['krea', 'wan'], loaded: true });
	});

	it('pins optimistically and keeps what the server returns', async () => {
		const { presetPickerState } = await freshStore();
		await presetPickerState.load();
		let resolve: (value: unknown) => void = () => {};
		calls.pinPreset.mockReturnValue(new Promise((r) => (resolve = r)));
		const pending = presetPickerState.togglePin('krea');
		expect(get(presetPickerState).pinned).toEqual(['seedvr2', 'krea']);
		resolve(ok(['seedvr2', 'krea'], ['krea', 'wan']));
		await pending;
		expect(calls.pinPreset).toHaveBeenCalledWith('krea');
		expect(get(presetPickerState).pinned).toEqual(['seedvr2', 'krea']);
	});

	it('unpins a pinned preset through the unpin route', async () => {
		const { presetPickerState } = await freshStore();
		await presetPickerState.load();
		calls.unpinPreset.mockResolvedValue(ok([], ['krea', 'wan']));
		await presetPickerState.togglePin('seedvr2');
		expect(calls.unpinPreset).toHaveBeenCalledWith('seedvr2');
		expect(calls.pinPreset).not.toHaveBeenCalled();
		expect(get(presetPickerState).pinned).toEqual([]);
	});

	it('rolls a pin back when the server refuses it', async () => {
		const { presetPickerState } = await freshStore();
		await presetPickerState.load();
		calls.pinPreset.mockResolvedValue({ success: false, error: 'preset_not_found' });
		await presetPickerState.togglePin('gone');
		expect(get(presetPickerState).pinned).toEqual(['seedvr2']);
		calls.pinPreset.mockRejectedValue(new Error('offline'));
		await presetPickerState.togglePin('krea');
		expect(get(presetPickerState).pinned).toEqual(['seedvr2']);
	});

	it('moves a chosen preset to the front of recent and records it on the server', async () => {
		const { presetPickerState } = await freshStore();
		await presetPickerState.load();
		calls.pushRecentPreset.mockResolvedValue(ok(['seedvr2'], ['wan', 'krea']));
		const pending = presetPickerState.pushRecent('wan');
		expect(get(presetPickerState).recent).toEqual(['wan', 'krea']);
		await pending;
		expect(calls.pushRecentPreset).toHaveBeenCalledWith('wan');
		expect(get(presetPickerState).recent).toEqual(['wan', 'krea']);
	});
});
