import { browser } from '$app/environment';

export const storage = {
	get(key: string, fallback?: string): string | null {
		if (!browser) return fallback ?? null;
		try {
			return localStorage.getItem(key) ?? fallback ?? null;
		} catch {
			return fallback ?? null;
		}
	},
	set(key: string, value: string): void {
		if (!browser) return;
		try {
			localStorage.setItem(key, value);
		} catch {}
	},
	remove(key: string): void {
		if (!browser) return;
		try {
			localStorage.removeItem(key);
		} catch {}
	},
	getJSON<T>(key: string, fallback?: T): T | null {
		if (!browser) return fallback ?? null;
		try {
			const raw = localStorage.getItem(key);
			return raw ? (JSON.parse(raw) as T) : (fallback ?? null);
		} catch {
			return fallback ?? null;
		}
	},
	setJSON(key: string, value: unknown): void {
		if (!browser) return;
		try {
			localStorage.setItem(key, JSON.stringify(value));
		} catch {}
	},
};
