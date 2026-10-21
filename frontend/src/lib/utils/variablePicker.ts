import { compileTextMatcher, type TextMatcher } from './textMatch';
import { isValidVariableName } from './promptVariables';
import {
	normalizeVariableDef,
	numberRangeLabel,
	optionText,
	type VariableDef,
	type VariablesMap
} from './variableDefs';

export type VariablePickerType = VariableDef['type'];

export interface VariablePickerRow {
	name: string;
	type: VariablePickerType;
	typeLabel: string;
	mode: string;
	preview: string;
	disabledReason: string | null;
}

export interface VariablePickerOptions {
	allowedTypes?: readonly VariablePickerType[];
	disabledReasons?: Record<string, string>;
}

const TYPE_LABELS: Record<VariablePickerType, string> = {
	text: 'Text',
	choice: 'Choice',
	number: 'Number'
};

const PREVIEW_OPTION_COUNT = 3;

export function variableModeLabel(def: VariableDef): string {
	if (def.type === 'text') return '';
	if (def.mode === 'pin') return 'pinned';
	if (def.mode === 'per-image') return 'per image';
	return 'shuffle';
}

export function variableValuePreview(def: VariableDef): string {
	if (def.type === 'text') return def.value.trim();
	if (def.type === 'number') return numberRangeLabel(def);
	const options = def.options.map((o) => optionText(o).trim()).filter(Boolean);
	if (options.length === 0) return '';
	const shown = options.slice(0, PREVIEW_OPTION_COUNT).join(' · ');
	const rest = options.length - PREVIEW_OPTION_COUNT;
	return rest > 0 ? `${shown} +${rest}` : shown;
}

export function normalizePickerQuery(query: string): string {
	return query.trim().replace(/^\$\{?/, '').replace(/\}$/, '');
}

export function variablePickerMatcher(query: string): TextMatcher | null {
	return compileTextMatcher(normalizePickerQuery(query)).matcher;
}

function searchableValue(def: VariableDef): string {
	if (def.type === 'text') return def.value;
	if (def.type === 'number') return numberRangeLabel(def);
	return def.options.map((o) => optionText(o)).join('\n');
}

export function variablePickerRows(
	variables: VariablesMap,
	query: string,
	options: VariablePickerOptions = {}
): VariablePickerRow[] {
	const needle = normalizePickerQuery(query).toLowerCase();
	const ranked: { row: VariablePickerRow; tier: number }[] = [];
	for (const name of Object.keys(variables)) {
		const def = normalizeVariableDef(variables[name]);
		if (options.allowedTypes && !options.allowedTypes.includes(def.type)) continue;
		const lower = name.toLowerCase();
		let tier = 0;
		if (needle) {
			if (lower.startsWith(needle)) tier = 0;
			else if (lower.includes(needle)) tier = 1;
			else if (searchableValue(def).toLowerCase().includes(needle)) tier = 2;
			else continue;
		}
		ranked.push({
			tier,
			row: {
				name,
				type: def.type,
				typeLabel: TYPE_LABELS[def.type],
				mode: variableModeLabel(def),
				preview: variableValuePreview(def),
				disabledReason: options.disabledReasons?.[name] ?? null
			}
		});
	}
	if (needle) ranked.sort((a, b) => a.tier - b.tier || a.row.name.localeCompare(b.row.name));
	return ranked.map((entry) => entry.row);
}

export function creatableVariableName(query: string, variables: VariablesMap): string | null {
	const name = normalizePickerQuery(query);
	if (!isValidVariableName(name)) return null;
	return name in variables ? null : name;
}
