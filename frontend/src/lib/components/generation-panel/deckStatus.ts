export type DeckState = 'running' | 'loading' | 'other' | 'blocked' | 'failed' | 'done' | 'ready';

export interface DeckStateInput {
	isGenerating: boolean;
	hasProgressFraction: boolean;
	otherTabName?: string | null;
	disabled: boolean;
	resultStatus?: string | null;
}

export function deriveDeckState(input: DeckStateInput): DeckState {
	if (input.isGenerating) return input.hasProgressFraction ? 'running' : 'loading';
	if (input.otherTabName) return 'other';
	if (input.disabled) return 'blocked';
	if (input.resultStatus === 'failed') return 'failed';
	if (input.resultStatus === 'completed') return 'done';
	return 'ready';
}

export interface DeckTitleInput {
	state: DeckState;
	otherTabName?: string | null;
	compareArmed: boolean;
	stopping: boolean;
	hasResult: boolean;
	playful: boolean;
}

export interface DeckTitle {
	literal: string;
	visible: string;
}

export function deckTitle(input: DeckTitleInput): DeckTitle {
	const literal = literalTitle(input);
	const brew = input.state === 'ready' && !input.compareArmed && !input.hasResult && input.playful;
	return { literal, visible: brew ? 'Ready to brew' : literal };
}

function literalTitle(input: DeckTitleInput): string {
	switch (input.state) {
		case 'running':
			return input.stopping ? 'Finishing this image' : 'Generating';
		case 'loading':
			return 'Loading model';
		case 'other':
			return `Generating in ${input.otherTabName}`;
		case 'blocked':
			return input.compareArmed ? "Can't compare yet" : "Can't generate yet";
		case 'failed':
			return 'Failed';
		case 'done':
			return 'Done';
		default:
			return input.compareArmed ? 'Ready to compare' : 'Ready to generate';
	}
}

export function deckEyebrow(state: DeckState, hasResult: boolean, lastText: string): string {
	const known = lastText !== 'none';
	if (state === 'running' || state === 'loading') return 'Now generating';
	if (state === 'done') return known ? `New result · ${lastText}` : 'New result';
	if (!hasResult) return 'No results yet';
	return known ? `Last result · ${lastText}` : 'Last result';
}

export function shouldAnnounceQuarter(percent: number, announced: number): number | null {
	const reached = Math.floor(percent / 25) * 25;
	return reached > announced && reached < 100 ? reached : null;
}
