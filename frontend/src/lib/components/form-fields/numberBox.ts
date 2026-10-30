export function parseNumberBox(text: string, integer: boolean): number | null | undefined {
	const trimmed = text.trim();
	if (trimmed === '') return null;
	if (!/^-?\d*\.?\d*$/.test(trimmed) || trimmed === '-' || trimmed === '.' || trimmed === '-.') {
		return undefined;
	}
	const parsed = Number(trimmed);
	if (!Number.isFinite(parsed)) return undefined;
	return integer ? Math.round(parsed) : parsed;
}

export function numberBoxIssue(
	value: number | null,
	text: string,
	min: unknown,
	max: unknown
): string | null {
	if (text.trim() !== '' && parseNumberBox(text, false) === undefined) return 'Enter a number';
	if (value === null) return null;
	if (typeof min === 'number' && value < min) return `At least ${min}`;
	if (typeof max === 'number' && value > max) return `At most ${max}`;
	return null;
}
