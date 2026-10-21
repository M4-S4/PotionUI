import type { StatsDimension } from '$lib/services/api/stats';
import type { StatsSection } from '$lib/stores/statsRowLimits';
import { formatCount } from '$lib/utils/format';

export type BreakdownSection = Exclude<StatsSection, 'storage' | 'presetTiming' | 'presetResources'>;

export interface BreakdownConfig {
	section: BreakdownSection;
	dimension: StatsDimension;
	title: string;
	subtitle?: string;
	label: string;
	group: 'activity' | 'parameters';
	colorBy: 'single' | 'group';
	groupLabel?: string;
	valueFormat?: (n: number) => string;
}

const roundedCount = (n: number) => formatCount(Math.round(n));

export const BREAKDOWNS: readonly BreakdownConfig[] = [
	{
		section: 'presets',
		dimension: 'preset',
		title: 'Top presets',
		subtitle: 'By generation count',
		label: 'Preset',
		group: 'activity',
		colorBy: 'single',
		valueFormat: roundedCount
	},
	{
		section: 'models',
		dimension: 'model',
		title: 'Top models',
		subtitle: 'Colored by model type',
		label: 'Model',
		group: 'activity',
		colorBy: 'group',
		groupLabel: 'Type',
		valueFormat: roundedCount
	},
	{
		section: 'resolutions',
		dimension: 'resolution',
		title: 'Top resolutions',
		subtitle: 'By generation count',
		label: 'Resolution',
		group: 'activity',
		colorBy: 'single',
		valueFormat: roundedCount
	},
	{ section: 'samplers', dimension: 'sampler', title: 'Samplers', label: 'Sampler', group: 'parameters', colorBy: 'single' },
	{ section: 'schedulers', dimension: 'scheduler', title: 'Schedulers', label: 'Scheduler', group: 'parameters', colorBy: 'single' },
	{ section: 'steps', dimension: 'steps', title: 'Steps', label: 'Steps', group: 'parameters', colorBy: 'single' },
	{ section: 'cfgs', dimension: 'cfg', title: 'CFG scale', label: 'CFG', group: 'parameters', colorBy: 'single' },
	{ section: 'denoises', dimension: 'denoise', title: 'Denoise', label: 'Denoise', group: 'parameters', colorBy: 'single' }
];

export const BREAKDOWN_BY_SECTION = Object.fromEntries(BREAKDOWNS.map((b) => [b.section, b])) as Record<
	BreakdownSection,
	BreakdownConfig
>;
