export function gridShortcutTarget(selected: readonly string[], focused: string | null): string | null {
	if (selected.length === 1) return selected[0];
	if (selected.length === 0) return focused;
	return null;
}
