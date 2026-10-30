export interface PromptCopy {
	label: string | null;
	placeholder: string | null;
	optional: boolean;
}

export const NO_PROMPT_COPY: PromptCopy = { label: null, placeholder: null, optional: false };

export const DEFAULT_PROMPT_LABEL = 'Prompt';
export const DEFAULT_PROMPT_PLACEHOLDER = 'Enter prompt content... (# for phrasebook)';

function text(value: unknown): string | null {
	return typeof value === 'string' && value.trim() ? value.trim() : null;
}

export function promptModeCopy(
	presetVars: Record<string, unknown> | null | undefined,
	mode: string | null | undefined,
	optional: boolean
): PromptCopy {
	const prompt = presetVars?.prompt;
	const modes =
		prompt && typeof prompt === 'object' ? (prompt as { modes?: unknown }).modes : null;
	const entry = mode && modes && typeof modes === 'object' ? (modes as Record<string, unknown>)[mode] : null;
	const record = entry && typeof entry === 'object' ? (entry as Record<string, unknown>) : {};
	return { label: text(record.label), placeholder: text(record.placeholder), optional };
}

export function promptTitle(copy: PromptCopy): string {
	const base = copy.label ?? DEFAULT_PROMPT_LABEL;
	return copy.optional ? `${base} (optional)` : base;
}
