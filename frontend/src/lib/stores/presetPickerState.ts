import { writable } from 'svelte/store';
import { api } from '$lib/services/api/index';
import type { APIResponse, PresetPickerState } from '$lib/types/api';
import { logger } from '$lib/utils/logger';

interface PresetPickerStoreState extends PresetPickerState {
	loaded: boolean;
}

const initialState: PresetPickerStoreState = { pinned: [], recent: [], loaded: false };

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((entry) => typeof entry === 'string');
}

function createPresetPickerStateStore() {
	const { subscribe, set, update } = writable<PresetPickerStoreState>(initialState);

	function apply(response: APIResponse<PresetPickerState>): boolean {
		const data = response?.data;
		if (!response?.success || !data || !isStringArray(data.pinned) || !isStringArray(data.recent)) return false;
		update((state) => ({ ...state, pinned: data.pinned, recent: data.recent, loaded: true }));
		return true;
	}

	return {
		subscribe,

		async load() {
			try {
				apply(await api.getPresetPickerState());
			} catch (error) {
				logger.error('Failed to load pinned and recent presets:', error);
			} finally {
				update((state) => ({ ...state, loaded: true }));
			}
		},

		async togglePin(presetId: string) {
			let previous: string[] = [];
			let pinning = true;
			update((state) => {
				previous = state.pinned;
				pinning = !previous.includes(presetId);
				return { ...state, pinned: pinning ? [...previous, presetId] : previous.filter((id) => id !== presetId) };
			});
			try {
				const response = pinning ? await api.pinPreset(presetId) : await api.unpinPreset(presetId);
				if (!apply(response)) update((state) => ({ ...state, pinned: previous }));
			} catch (error) {
				logger.error('Failed to save pinned presets:', error);
				update((state) => ({ ...state, pinned: previous }));
			}
		},

		async pushRecent(presetId: string) {
			update((state) => ({ ...state, recent: [presetId, ...state.recent.filter((id) => id !== presetId)] }));
			try {
				apply(await api.pushRecentPreset(presetId));
			} catch (error) {
				logger.error('Failed to record the recent preset:', error);
			}
		},

		reset() {
			set(initialState);
		}
	};
}

export const presetPickerState = createPresetPickerStateStore();
