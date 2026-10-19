import type { VariableRoll } from '$lib/utils/variableDefs';

export interface ChatFormStateTab {
	id: string;
	selectedPreset?: string | null;
	selectedMode?: string | null;
	selectedVariant?: string | null;
	formData?: Record<string, unknown>;
	variableRolls?: Record<string, VariableRoll>;
}

export interface ChatFormStateInput {
	tab: ChatFormStateTab | null | undefined;
	generatePageActive: boolean;
	generationPresetActive: boolean;
	sessionId: string | null;
	videoDirector: unknown;
	musicDirector: unknown;
}

function composeLastRoll(roll: VariableRoll): string {
	const value = roll.value.trim();
	const eligibility = roll.eligible !== undefined && roll.total !== undefined ? `${roll.eligible} of ${roll.total} eligible` : undefined;

	if (roll.because) {
		const because = `because $${roll.because.var} = ${roll.because.value}`;
		return `${value} (${eligibility ? `${because}, ${eligibility}` : because})`;
	}
	if (eligibility) return `${value} (${eligibility})`;
	return value;
}

function lastRolls(rolls: Record<string, VariableRoll> | undefined): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [name, roll] of Object.entries(rolls ?? {})) {
		if (roll?.value?.trim()) out[name] = composeLastRoll(roll);
	}
	return out;
}

export function buildChatFormState({
	tab,
	generatePageActive,
	generationPresetActive,
	sessionId,
	videoDirector,
	musicDirector
}: ChatFormStateInput) {
	return {
		preset: generationPresetActive ? tab?.selectedPreset || null : null,
		tab_id: generatePageActive ? (tab?.id ?? null) : null,
		session_id: sessionId,
		mode: tab?.selectedMode || null,
		variant: tab?.selectedVariant || null,
		form_data: generatePageActive ? tab?.formData || {} : {},
		variable_rolls: lastRolls(tab?.variableRolls),
		video_director: videoDirector,
		music_director: musicDirector
	};
}
