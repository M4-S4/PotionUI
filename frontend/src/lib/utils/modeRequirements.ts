export function modeRequirementIssue(missing: readonly string[] | null | undefined): string | undefined {
	const count = missing?.length ?? 0;
	if (count === 0) return undefined;
	return count === 1 ? 'A required file is missing.' : `${count} required files are missing.`;
}

export function modeUnavailableReason(
	modes: ReadonlyArray<{ name: string; requirements_missing?: readonly string[] | null }> | null | undefined,
	selectedMode: string | null | undefined
): string | undefined {
	if (!selectedMode) return undefined;
	const mode = modes?.find((entry) => entry.name === selectedMode);
	return modeRequirementIssue(mode?.requirements_missing);
}
