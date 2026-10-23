import { writable } from 'svelte/store';

let live: Record<string, boolean> = {};
const store = writable(live);

export const tabSessionDirty = { subscribe: store.subscribe };

export function setTabSessionDirty(tabId: string, dirty: boolean): void {
	if (live[tabId] === dirty) return;
	live = { ...live, [tabId]: dirty };
	store.set(live);
}

export function clearTabSessionDirty(tabId: string): void {
	if (!(tabId in live)) return;
	const next = { ...live };
	delete next[tabId];
	live = next;
	store.set(live);
}
