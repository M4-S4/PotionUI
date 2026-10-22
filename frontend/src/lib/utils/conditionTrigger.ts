export const CONDITION_TRIGGER = '<<';

export interface ConditionTriggerMatch {
	start: number;
	end: number;
}

export function detectConditionTrigger(text: string, cursorOffset: number): ConditionTriggerMatch | null {
	const start = cursorOffset - CONDITION_TRIGGER.length;
	if (start < 0 || text.slice(start, cursorOffset) !== CONDITION_TRIGGER) return null;
	if (text[start - 1] === '<' || text[cursorOffset] === '<') return null;
	return { start, end: cursorOffset };
}

export function isTypedCharacter(event: Event, character: string): boolean {
	const input = event as InputEvent;
	return input.inputType === 'insertText' && input.data === character && !input.isComposing;
}
