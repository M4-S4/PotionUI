import { getContext } from 'svelte';
import { readable, type Readable } from 'svelte/store';

/**
 * Set by Tooltip.svelte so descendants can tell they already have a tooltip.
 *
 * Controls that carry their own native `title` as a fallback hint (IconButton)
 * would otherwise render twice inside a Tooltip: the styled one and the
 * browser's. Reading this lets them drop the attribute instead of every call
 * site having to remember to.
 */
export const INSIDE_TOOLTIP_CONTEXT_KEY = 'insideTooltip';

export const TOOLTIP_DESCRIPTION_CONTEXT_KEY = 'tooltipDescription';

const noDescription = readable<string | undefined>(undefined);

let tooltipCount = 0;

export function nextTooltipId(): string {
	tooltipCount += 1;
	return `tooltip-${tooltipCount}`;
}

// :focus-visible heuristic (WICG polyfill): focus counts as keyboard unless the last input was a pointer.
let lastInput: 'keyboard' | 'pointer' = 'keyboard';
let trackingInput = false;

export function trackInputModality(): void {
	if (trackingInput || typeof document === 'undefined') return;
	trackingInput = true;
	const pointer = () => {
		lastInput = 'pointer';
	};
	document.addEventListener(
		'keydown',
		(event) => {
			if (!event.metaKey && !event.altKey && !event.ctrlKey) lastInput = 'keyboard';
		},
		true
	);
	document.addEventListener('mousedown', pointer, true);
	document.addEventListener('pointerdown', pointer, true);
	document.addEventListener('touchstart', pointer, { capture: true, passive: true });
}

export function lastInputWasKeyboard(): boolean {
	return lastInput === 'keyboard';
}

export function tooltipDescription(): Readable<string | undefined> {
	return getContext<Readable<string | undefined> | undefined>(TOOLTIP_DESCRIPTION_CONTEXT_KEY) ?? noDescription;
}
