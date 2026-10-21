import type { PresetConfigurationEntry } from '$lib/types/api';
import { jsonEqual } from '$lib/utils/equal';

export function valuesFrom(entries: PresetConfigurationEntry[]): Record<string, unknown> {
	return Object.fromEntries(entries.map((entry) => [entry.key, entry.value]));
}

export function dirtyKeysOf(
	entries: PresetConfigurationEntry[],
	pending: Record<string, unknown>,
	original: Record<string, unknown>
): string[] {
	return entries
		.filter((entry) => !jsonEqual(pending[entry.key], original[entry.key]))
		.map((entry) => entry.key);
}

export function buildSavePayload(dirtyKeys: string[], pending: Record<string, unknown>): Record<string, unknown> {
	return Object.fromEntries(dirtyKeys.map((key) => [key, pending[key]]));
}
