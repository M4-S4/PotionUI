import { writable } from 'svelte/store';
import type { SegmentRemoval } from '$lib/utils/promptSegments';

function createRemovedSegmentActionsStore() {
	const { subscribe, update } = writable<Record<string, SegmentRemoval>>({});

	return {
		subscribe,

		set(messageId: string, actionIndex: number, removal: SegmentRemoval) {
			update((current) => ({ ...current, [removedActionKey(messageId, actionIndex)]: removal }));
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

export function removedActionKey(messageId: string, actionIndex: number): string {
	return `${messageId}:${actionIndex}`;
}

export const removedSegmentActions = createRemovedSegmentActionsStore();
