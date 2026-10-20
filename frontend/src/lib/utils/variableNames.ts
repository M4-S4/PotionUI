import { isValidVariableName } from './promptVariables';

export interface NamedRow {
	key: string;
	name: string;
}

export const INVALID_NAME_ERROR = 'Letters, numbers, underscore; must not start with a number.';
export const DUPLICATE_NAME_ERROR = 'Another variable already uses this name. Names are not case-sensitive.';

export function variableNameKey(name: string): string {
	return name.trim().toLowerCase();
}

export function variableNameErrors(rows: readonly NamedRow[]): Map<string, string | null> {
	const counts = new Map<string, number>();
	for (const row of rows) {
		const key = variableNameKey(row.name);
		if (key) counts.set(key, (counts.get(key) ?? 0) + 1);
	}
	return new Map(
		rows.map((row) => {
			const trimmed = row.name.trim();
			if (!trimmed) return [row.key, null];
			if (!isValidVariableName(trimmed)) return [row.key, INVALID_NAME_ERROR];
			if ((counts.get(variableNameKey(trimmed)) ?? 0) > 1) return [row.key, DUPLICATE_NAME_ERROR];
			return [row.key, null];
		})
	);
}

export function usableNamedRows<T extends NamedRow>(rows: readonly T[]): T[] {
	const seen = new Set<string>();
	const usable: T[] = [];
	for (const row of rows) {
		const trimmed = row.name.trim();
		if (!trimmed || !isValidVariableName(trimmed)) continue;
		const key = variableNameKey(trimmed);
		if (seen.has(key)) continue;
		seen.add(key);
		usable.push(row);
	}
	return usable;
}

export function findNameCollision(variables: Record<string, unknown>, name: string): string | null {
	const key = variableNameKey(name);
	if (!key) return null;
	return Object.keys(variables).find((existing) => variableNameKey(existing) === key) ?? null;
}

export function dedupeVariableNames<T>(variables: Record<string, T>): { variables: Record<string, T>; dropped: string[] } {
	const kept: Record<string, T> = {};
	const dropped: string[] = [];
	for (const name of Object.keys(variables)) {
		if (findNameCollision(kept, name) !== null) dropped.push(name);
		else kept[name] = variables[name];
	}
	return { variables: kept, dropped };
}
