import {
	clampNumberValue,
	normalizeVariableDef,
	optionText,
	type VariableDef,
	type VariablesMap,
	type VariableRoll
} from './variableDefs';

const MAX_OPTIONS = 12;
const MAX_VALUE_CHARS = 80;

function clip(text: string, limit = MAX_VALUE_CHARS): string {
	return text.length > limit ? text.slice(0, limit) + '…' : text;
}

function validOptions(def: Extract<VariableDef, { type: 'choice' }>): string[] {
	return def.options.map((o) => optionText(o).trim()).filter((o) => o.length > 0);
}

function describeVariable(def: VariableDef, roll: VariableRoll | undefined): string {
	if (def.type === 'text') {
		const value = def.value.trim();
		return value ? clip(value) : 'Text variable';
	}
	if (def.type === 'number') {
		const range = `a number from ${def.min} to ${def.max} in steps of ${def.step}`;
		if (def.mode === 'pin') return `${range} — pinned to ${clampNumberValue(def)}`;
		if (def.mode === 'per-image') return `${range} — re-rolls per image`;
		const last = roll?.value ? `; last roll: ${clip(roll.value.trim())}` : '';
		return `${range} — shuffles each generation${last}`;
	}
	const opts = validOptions(def);
	const shown = opts.slice(0, MAX_OPTIONS).map((o) => clip(o));
	let listing = shown.join(', ');
	if (opts.length > MAX_OPTIONS) listing += ', …';
	let phrase: string;
	if (def.mode === 'pin') {
		const pinnedOption = def.pinnedIndex !== null ? def.options[def.pinnedIndex] : undefined;
		const pinned = pinnedOption !== undefined ? optionText(pinnedOption).trim() : '';
		phrase = pinned ? `pinned to ${clip(pinned)}` : 'pinned';
	} else if (def.mode === 'per-image') {
		phrase = 're-rolls per image';
	} else {
		phrase = 'shuffles each generation';
	}
	let text = `one of ${listing} — ${phrase}`;
	if (def.mode === 'shuffle' && roll?.value) text += `; last roll: ${clip(roll.value.trim())}`;
	return text;
}

/**
 * Map of variable name → tooltip text for the chat message `${name}` chips.
 * A name present here is "known" (renders as a chip); absent names stay
 * literal text. Choice variables with no valid options are still known (the
 * user is mid-authoring), matching the editor's VariableUsageChip.
 */
export function buildVariableChipTooltips(
	variables: VariablesMap | undefined,
	rolls?: Record<string, VariableRoll> | undefined
): Record<string, string> {
	const out: Record<string, string> = {};
	if (!variables) return out;
	for (const [name, stored] of Object.entries(variables)) {
		if (!name.trim()) continue;
		out[name] = describeVariable(normalizeVariableDef(stored), rolls?.[name]);
	}
	return out;
}
