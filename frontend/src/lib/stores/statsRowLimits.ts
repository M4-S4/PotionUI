import { persistedStore } from './persistedStore';

export type StatsSection =
	| 'presets'
	| 'models'
	| 'samplers'
	| 'schedulers'
	| 'resolutions'
	| 'steps'
	| 'cfgs'
	| 'denoises'
	| 'storage'
	| 'presetTiming'
	| 'presetResources';

export const STATS_ROW_LIMIT_OPTIONS = [5, 8, 10, 15, 20, 30, 50, 100] as const;

const DEFAULTS: Record<StatsSection, number> = {
	presets: 8,
	models: 8,
	samplers: 8,
	schedulers: 8,
	resolutions: 8,
	steps: 8,
	cfgs: 8,
	denoises: 8,
	storage: 30,
	presetTiming: 10,
	presetResources: 10
};

const store = persistedStore<Record<StatsSection, number>>(
	'admin-stats-row-limits',
	(raw) => ({ ...DEFAULTS, ...(JSON.parse(raw) as Partial<Record<StatsSection, number>>) }),
	{ ...DEFAULTS },
	JSON.stringify
);

export const statsRowLimits = {
	subscribe: store.subscribe,
	setLimit(section: StatsSection, limit: number) {
		store.update((state) => ({ ...state, [section]: limit }));
	}
};
