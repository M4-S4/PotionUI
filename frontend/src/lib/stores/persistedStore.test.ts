import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('$app/environment', () => ({ browser: true }));

function installStorage(overrides: Partial<Storage> = {}) {
	const data = new Map<string, string>();
	const storage = {
		getItem: (key: string) => data.get(key) ?? null,
		setItem: (key: string, value: string) => void data.set(key, value),
		removeItem: (key: string) => void data.delete(key),
		...overrides
	};
	vi.stubGlobal('localStorage', storage);
	return data;
}

function blockedStorage() {
	const blocked = () => {
		throw new Error('SecurityError');
	};
	installStorage({ getItem: blocked, setItem: blocked });
}

describe('persistedStore', () => {
	beforeEach(() => {
		vi.resetModules();
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('restores a stored value through parse and writes sets back', async () => {
		const data = installStorage();
		data.set('k', '7');
		const { persistedStore } = await import('./persistedStore');
		const store = persistedStore('k', Number, 1);
		expect(get(store)).toBe(7);
		store.set(9);
		expect(data.get('k')).toBe('9');
		expect(get(store)).toBe(9);
	});

	it('falls back when nothing is stored or parse throws', async () => {
		const data = installStorage();
		const { persistedStore } = await import('./persistedStore');
		expect(get(persistedStore('missing', Number, 3))).toBe(3);
		data.set('bad', '{');
		expect(get(persistedStore('bad', (raw) => JSON.parse(raw) as number, 4))).toBe(4);
	});

	it('keeps working in memory when storage throws on read and write', async () => {
		blockedStorage();
		const { persistedStore } = await import('./persistedStore');
		const store = persistedStore('k', Number, 5);
		expect(get(store)).toBe(5);
		store.update((value) => value + 1);
		expect(get(store)).toBe(6);
	});

	it.each(['./formAudience', './historyTileSize', './theme', './statsRowLimits'])(
		'importing %s does not throw when storage is blocked',
		async (path) => {
			blockedStorage();
			vi.stubGlobal('window', { matchMedia: () => ({ matches: false, addEventListener() {} }) });
			vi.stubGlobal('document', { documentElement: { dataset: {} }, querySelector: () => null });
			await expect(import(/* @vite-ignore */ path)).resolves.toBeTruthy();
		}
	);
});
