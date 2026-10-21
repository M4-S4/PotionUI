import type { PresetInfo } from '$lib/types/api';
import { presetCategoryIcon, presetCategoryLabel, presetCategoryOrder } from '$lib/utils/presetCategories';
import { formatGb, vramShortfall } from '$lib/utils/presetHardware';

export const RECENT_SHOWN = 3;

export interface PresetSection {
	key: string;
	label: string;
	icon: string;
	family: boolean;
	presets: PresetInfo[];
}

export interface PresetIssue {
	kind: 'backend' | 'missing' | 'vram';
	label: string;
	detail: string;
	blocking: boolean;
}

export interface PickerLists {
	pinned: string[];
	recent: string[];
}

export function engineLabel(engine: string | null | undefined): string {
	if (!engine) return 'unknown';
	return engine.toLowerCase() === 'comfyui' ? 'ComfyUI' : engine;
}

export function presetIssue(preset: PresetInfo, detectedVramGb: number | null = null): PresetIssue | null {
	if (preset.engine_available === false) {
		const engine = engineLabel(preset.engine);
		return {
			kind: 'backend',
			label: `No ${engine} backend`,
			detail: `This preset runs on ${engine}, and no ${engine} backend is set up.`,
			blocking: true
		};
	}
	const missing = preset.requirements_summary?.missing ?? 0;
	if (missing > 0) {
		return {
			kind: 'missing',
			label: `${missing} missing`,
			detail: missing === 1 ? 'A required file is missing.' : `${missing} required files are missing.`,
			blocking: true
		};
	}
	const shortfall = vramShortfall(preset.requires, detectedVramGb);
	if (shortfall && preset.requires?.min_vram_gb != null && detectedVramGb != null) {
		return {
			kind: 'vram',
			label: shortfall,
			detail: `Needs about ${formatGb(preset.requires.min_vram_gb)} GB of VRAM; this GPU has ${formatGb(detectedVramGb)} GB.`,
			blocking: false
		};
	}
	return null;
}

export function presetUnavailableReason(preset: PresetInfo | null | undefined): string | undefined {
	if (!preset) return undefined;
	if (preset.engine_available === false) return `No ${engineLabel(preset.engine)} backend is set up`;
	const missing = preset.requirements_summary?.missing ?? 0;
	if (missing === 1) return '1 required file is missing';
	if (missing > 1) return `${missing} required files are missing`;
	return undefined;
}

export function matchesPreset(preset: PresetInfo, query: string): boolean {
	const needle = query.trim().toLowerCase();
	if (!needle) return true;
	return (
		preset.name.toLowerCase().includes(needle) ||
		!!preset.category?.toLowerCase().includes(needle) ||
		presetCategoryLabel(preset.category).toLowerCase().includes(needle) ||
		!!preset.engine?.toLowerCase().includes(needle) ||
		!!preset.tags?.some((tag) => tag.toLowerCase().includes(needle))
	);
}

function byFamily(presets: PresetInfo[]): PresetSection[] {
	const families = new Map<string, PresetInfo[]>();
	for (const preset of presets) {
		const key = preset.category || 'other';
		families.set(key, [...(families.get(key) ?? []), preset]);
	}
	return [...families.entries()]
		.sort(([a], [b]) => presetCategoryOrder(a) - presetCategoryOrder(b) || a.localeCompare(b))
		.map(([key, members]) => ({
			key,
			label: presetCategoryLabel(key === 'other' ? null : key),
			icon: presetCategoryIcon(key === 'other' ? null : key),
			family: true,
			presets: [...members].sort(
				(a, b) => Number(!!presetIssue(a)?.blocking) - Number(!!presetIssue(b)?.blocking)
			)
		}));
}

export function presetSections(
	presets: PresetInfo[],
	query: string,
	lists: PickerLists,
	currentId: string | null | undefined
): PresetSection[] {
	if (query.trim()) return byFamily(presets.filter((preset) => matchesPreset(preset, query)));
	const byId = new Map(presets.map((preset) => [preset.id, preset]));
	const pick = (ids: string[]) => ids.map((id) => byId.get(id)).filter((preset): preset is PresetInfo => !!preset);
	const recent = pick(lists.recent.filter((id) => id !== currentId)).slice(0, RECENT_SHOWN);
	const pinned = pick(lists.pinned);
	return [
		...(recent.length ? [{ key: 'recent', label: 'Recent', icon: 'clock', family: false, presets: recent }] : []),
		...(pinned.length ? [{ key: 'pinned', label: 'Pinned', icon: 'pin', family: false, presets: pinned }] : []),
		...byFamily(presets)
	];
}
