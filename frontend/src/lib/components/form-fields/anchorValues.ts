export const ANCHOR_VALUES = [
	'top_left',
	'top',
	'top_right',
	'left',
	'center',
	'right',
	'bottom_left',
	'bottom',
	'bottom_right'
] as const;

export type AnchorValue = (typeof ANCHOR_VALUES)[number];

export function anchorLabel(anchor: string): string {
	const words = anchor.replace(/[_-]+/g, ' ').trim();
	return words ? words.charAt(0).toUpperCase() + words.slice(1) : '';
}

export function isAnchorValue(value: unknown): value is AnchorValue {
	return (ANCHOR_VALUES as readonly unknown[]).includes(value);
}
