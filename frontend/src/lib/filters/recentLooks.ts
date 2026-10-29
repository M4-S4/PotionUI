import { get, writable } from 'svelte/store';
import { browser } from '$app/environment';

export interface RecentLook {
	id: string;
	name: string;
	group: string;
}

export const RECENT_LOOKS_LIMIT = 3;
export const RECENT_LOOKS_STORAGE_KEY = 'potionui-recent-looks';

function sanitize(value: unknown): RecentLook[] {
	if (!Array.isArray(value)) return [];
	const seen = new Set<string>();
	const looks: RecentLook[] = [];
	for (const entry of value) {
		if (!entry || typeof entry !== 'object') continue;
		const { id, name, group } = entry as Record<string, unknown>;
		if (typeof id !== 'string' || !id || typeof name !== 'string' || !name || seen.has(id)) continue;
		seen.add(id);
		looks.push({ id, name, group: typeof group === 'string' ? group : '' });
		if (looks.length === RECENT_LOOKS_LIMIT) break;
	}
	return looks;
}

function readStorage(): RecentLook[] {
	if (!browser) return [];
	try {
		const raw = localStorage.getItem(RECENT_LOOKS_STORAGE_KEY);
		return raw ? sanitize(JSON.parse(raw)) : [];
	} catch {
		return [];
	}
}

function writeStorage(looks: RecentLook[]): void {
	try {
		localStorage.setItem(RECENT_LOOKS_STORAGE_KEY, JSON.stringify(looks));
	} catch {
		return;
	}
}

export function nextRecentLooks(current: readonly RecentLook[], look: RecentLook): RecentLook[] {
	const entry: RecentLook = { id: look.id, name: look.name, group: look.group };
	return [entry, ...current.filter((existing) => existing.id !== look.id)].slice(0, RECENT_LOOKS_LIMIT);
}

export const recentLooks = writable<RecentLook[]>(readStorage());

export function reloadRecentLooks(): void {
	recentLooks.set(readStorage());
}

export function recordRecentLook(look: RecentLook): RecentLook[] {
	if (!look.id || !look.name) return get(recentLooks);
	const next = nextRecentLooks(get(recentLooks), look);
	recentLooks.set(next);
	writeStorage(next);
	return next;
}
