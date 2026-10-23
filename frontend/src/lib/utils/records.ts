export function isRecord(v: unknown): v is Record<string, unknown> {
	return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function mergeRawCapabilities(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
	const merged: Record<string, unknown> = { ...base };
	for (const [key, value] of Object.entries(override)) {
		if (key !== 'modes') merged[key] = value;
	}
	const baseModes = isRecord(base.modes) ? base.modes : {};
	const overrideModes = isRecord(override.modes) ? override.modes : {};
	const mergedModes: Record<string, unknown> = { ...baseModes };
	for (const [compMode, compOverride] of Object.entries(overrideModes)) {
		const baseEntry = baseModes[compMode];
		mergedModes[compMode] = isRecord(compOverride) && isRecord(baseEntry) ? { ...baseEntry, ...compOverride } : compOverride;
	}
	merged.modes = mergedModes;
	return merged;
}
