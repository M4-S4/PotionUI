import type { PluginSettingSchema } from '$lib/stores/plugins';

const TRUE_STRINGS = new Set(['true', '1', 'yes', 'on']);

function toBoolean(value: unknown): boolean {
	return value === true || TRUE_STRINGS.has(String(value).trim().toLowerCase());
}

function toNumber(value: unknown): unknown {
	if (typeof value !== 'string' || value.trim() === '') return value;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : value;
}

/**
 * The Settings form's initial values for a plugin. `settings_values` holds
 * strings: a `boolean` entry becomes a boolean and a numeric `number` entry a
 * number; a missing entry gets its coerced `default`, an `info` entry never
 * gets a value, a stored key the schema does not declare is kept as is.
 */
export function pluginSettingFormValues(
	schema: PluginSettingSchema[] | undefined,
	stored: Record<string, unknown> | undefined
): Record<string, unknown> {
	const values: Record<string, unknown> = { ...(stored ?? {}) };
	for (const entry of schema ?? []) {
		if (entry.type === 'info') {
			delete values[entry.name];
			continue;
		}
		if (!(entry.name in values)) {
			if (entry.default === undefined) continue;
			values[entry.name] = entry.default;
		}
		if (entry.type === 'boolean') values[entry.name] = toBoolean(values[entry.name]);
		else if (entry.type === 'number') values[entry.name] = toNumber(values[entry.name]);
	}
	return values;
}
