import { writable } from 'svelte/store';
import type { PromptUndo } from '$lib/chat/promptActions';
import { removedActionKey } from '$lib/stores/removedSegmentActions';

function createPromptActionUndosStore() {
	const { subscribe, update } = writable<Record<string, PromptUndo>>({});

	return {
		subscribe,

		set(messageId: string, actionIndex: number, undo: PromptUndo) {
			update((current) => ({ ...current, [removedActionKey(messageId, actionIndex)]: undo }));
		},

		clear(messageId: string, actionIndex: number) {
			update((current) => {
				const key = removedActionKey(messageId, actionIndex);
				if (!(key in current)) return current;
				const next = { ...current };
				delete next[key];
				return next;
			});
		}
	};
}

export const promptActionUndos = createPromptActionUndosStore();

export function promptActionMarkerKey(messageId: string, actionIndex: number): string {
	return `prompt-action:${removedActionKey(messageId, actionIndex)}`;
}
