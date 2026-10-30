import type { PresetModeVariant } from '$lib/types/api';
import { modeLabel } from '$lib/utils/modeLabel.svelte';
import { hasIcon } from '$lib/utils/IconLibrary';
import { modeRequirementIssue } from '$lib/utils/modeRequirements';

export interface PresetModeChoice {
	id: string;
	label: string;
	variants?: PresetModeVariant[];
	sourcePlugin?: string | null;
	description?: string | null;
	icon?: string | null;
	requirementsMissing?: string[] | null;
}

export interface PresetModeOption {
	value: string;
	label: string;
	icon?: string;
	description?: string;
	marker?: string;
	markerLabel?: string;
	issue?: string;
}

export function presetModeOptions(modes: PresetModeChoice[]): PresetModeOption[] {
	return modes.map((mode) => ({
		value: mode.id,
		label: modeLabel(mode.id, mode.label),
		icon: mode.icon && hasIcon(mode.icon) ? mode.icon : undefined,
		description: mode.description || undefined,
		marker: mode.sourcePlugin ? '•' : undefined,
		markerLabel: mode.sourcePlugin ? `contributed by ${mode.sourcePlugin}` : undefined,
		issue: modeRequirementIssue(mode.requirementsMissing)
	}));
}
