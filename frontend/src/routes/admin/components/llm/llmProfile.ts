import type {
	LLMProfile,
	LLMProfileField,
	LLMProfileOverrides,
	LLMProfileSchema,
	LLMProfileSchemaField,
	LLMProfileSource
} from '$lib/types/llm';

export type ProfileDraft = Record<string, unknown>;

export const EDITABLE_KINDS = ['enum', 'set', 'int', 'float', 'bool', 'string'] as const;

export const SOURCE_LABEL: Record<LLMProfileSource, string> = {
	seed: 'Seed',
	discovered: 'Discovered',
	override: 'Override'
};

export const SOURCE_VARIANT: Record<LLMProfileSource, 'neutral' | 'info' | 'signal'> = {
	seed: 'neutral',
	discovered: 'info',
	override: 'signal'
};

export interface ProfileRow {
	name: string;
	label: string;
	kind: string;
	options: string[];
	min?: number;
	max?: number;
	value: unknown;
	source: LLMProfileSource;
	seed: unknown;
	discovered: unknown;
	savedValue: unknown;
	savedOverride: unknown;
	edited: boolean;
	editable: boolean;
	problem: string | null;
}

export interface ProfileGroup {
	key: string;
	label: string;
	rows: ProfileRow[];
}

const OTHER_GROUP = 'other';

export function humanizeProfileName(name: string): string {
	const spaced = name.replace(/[_-]+/g, ' ').trim();
	return spaced ? spaced.charAt(0).toUpperCase() + spaced.slice(1) : name;
}

function isSet(value: unknown): boolean {
	return value !== null && value !== undefined;
}

function comparable(value: unknown): string {
	return JSON.stringify(Array.isArray(value) ? [...value].map(String).sort() : (value ?? null));
}

export function sameProfileValue(a: unknown, b: unknown): boolean {
	return comparable(a) === comparable(b);
}

function rowProblem(kind: string, value: unknown, min?: number, max?: number): string | null {
	if (kind !== 'int' && kind !== 'float') return null;
	if (typeof value !== 'number') return null;
	if (kind === 'int' && !Number.isInteger(value)) return 'Enter a whole number';
	if (min !== undefined && value < min) return `Minimum is ${min.toLocaleString('en-US')}`;
	if (max !== undefined && value > max) return `Maximum is ${max.toLocaleString('en-US')}`;
	return null;
}

function buildRow(
	schema: LLMProfileSchemaField | null,
	name: string,
	field: LLMProfileField,
	draft: ProfileDraft
): ProfileRow {
	const kind = schema?.kind ?? 'unknown';
	const edited = Object.prototype.hasOwnProperty.call(draft, name);
	const draftValue = edited ? draft[name] : undefined;
	const cleared = edited && draftValue === null;
	const pinned = edited && !cleared;
	const fallback = isSet(field.discovered) ? field.discovered : field.seed;
	const value = pinned ? draftValue : cleared ? fallback : field.value;
	const source: LLMProfileSource = pinned
		? 'override'
		: cleared
			? isSet(field.discovered)
				? 'discovered'
				: 'seed'
			: field.source;
	return {
		name,
		label: schema?.label ?? humanizeProfileName(name),
		kind,
		options: schema?.options ?? [],
		min: schema?.min,
		max: schema?.max,
		value,
		source,
		seed: field.seed,
		discovered: field.discovered,
		savedValue: field.value,
		savedOverride: field.override,
		edited,
		editable: (EDITABLE_KINDS as readonly string[]).includes(kind),
		problem: pinned ? rowProblem(kind, value, schema?.min, schema?.max) : null
	};
}

export function buildProfileGroups(
	profile: LLMProfile,
	schema: LLMProfileSchema,
	draft: ProfileDraft
): ProfileGroup[] {
	const groups = new Map<string, ProfileGroup>();
	const place = (groupKey: string, row: ProfileRow) => {
		const existing = groups.get(groupKey);
		if (existing) existing.rows.push(row);
		else groups.set(groupKey, { key: groupKey, label: humanizeProfileName(groupKey), rows: [row] });
	};
	const known = new Set<string>();
	for (const entry of schema.fields) {
		const field = profile.fields[entry.name];
		if (!field) continue;
		known.add(entry.name);
		place(entry.group || 'general', buildRow(entry, entry.name, field, draft));
	}
	for (const [name, field] of Object.entries(profile.fields)) {
		if (!known.has(name)) place(OTHER_GROUP, buildRow(null, name, field, draft));
	}
	const ordered = [...groups.values()];
	const other = ordered.findIndex((group) => group.key === OTHER_GROUP);
	if (other > -1 && other !== ordered.length - 1) ordered.push(...ordered.splice(other, 1));
	return ordered;
}

export function setProfileDraft(
	draft: ProfileDraft,
	row: Pick<ProfileRow, 'name' | 'savedValue' | 'savedOverride'>,
	next: unknown
): ProfileDraft {
	const rest = { ...draft };
	delete rest[row.name];
	if (next === null || next === undefined) {
		return isSet(row.savedOverride) ? { ...rest, [row.name]: null } : rest;
	}
	return sameProfileValue(next, row.savedValue) ? rest : { ...rest, [row.name]: next };
}

export function clearProfileDraft(
	draft: ProfileDraft,
	row: Pick<ProfileRow, 'name' | 'savedValue' | 'savedOverride'>
): ProfileDraft {
	return setProfileDraft(draft, row, null);
}

export function profileDraftCount(draft: ProfileDraft): number {
	return Object.keys(draft).length;
}

export function profileDraftPayload(draft: ProfileDraft): LLMProfileOverrides {
	return { ...draft };
}

export function profileHasProblems(groups: ProfileGroup[]): boolean {
	return groups.some((group) => group.rows.some((row) => row.problem !== null));
}

export function isOverridden(row: ProfileRow): boolean {
	return row.source === 'override';
}

export function formatProfileValue(value: unknown): string {
	if (value === null || value === undefined) return '—';
	if (typeof value === 'boolean') return value ? 'Yes' : 'No';
	if (typeof value === 'number') return value.toLocaleString('en-US');
	if (Array.isArray(value)) return value.length ? value.map(String).join(', ') : 'None';
	if (typeof value === 'object') return JSON.stringify(value);
	return String(value);
}

export function profileLayers(row: ProfileRow): string[] {
	const layers: string[] = [];
	if (isSet(row.seed) && row.source !== 'seed') layers.push(`Seed ${formatProfileValue(row.seed)}`);
	if (isSet(row.discovered) && row.source !== 'discovered') {
		layers.push(`Discovered ${formatProfileValue(row.discovered)}`);
	}
	return layers;
}

export function recommendedSampling(profile: LLMProfile | null | undefined): Record<string, number> {
	const value = profile?.fields.recommended_sampling?.value;
	if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
	return Object.fromEntries(
		Object.entries(value as Record<string, unknown>).filter(
			(entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1])
		)
	);
}

export function modelDefaultHint(recommended: number | undefined): string {
	return recommended === undefined ? 'Model default' : `Model default · ${recommended}`;
}
