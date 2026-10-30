export const TRANSPARENT = 'transparent';

export function normalizeHex(value: unknown): string | null {
	if (typeof value !== 'string') return null;
	const match = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value.trim());
	if (!match) return null;
	const digits = match[1].toUpperCase();
	const full = digits.length === 3 ? [...digits].map((digit) => digit + digit).join('') : digits;
	return `#${full}`;
}
