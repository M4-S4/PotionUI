import type { PresetInfo } from '$lib/types/api';

export function presetInfo(overrides: Partial<PresetInfo> = {}): PresetInfo {
	return {
		id: 'krea2',
		name: 'Krea-2',
		version: '1.0.0',
		description: '',
		tags: [],
		category: 'image',
		engine: 'native',
		installed: false,
		...overrides
	};
}
