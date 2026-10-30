import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

const storage = new Map<string, string>();
const localStorageStub = {
	getItem: vi.fn((key: string) => (storage.has(key) ? storage.get(key)! : null)),
	setItem: vi.fn((key: string, value: string) => void storage.set(key, value)),
	removeItem: vi.fn((key: string) => void storage.delete(key))
};

vi.stubGlobal('localStorage', localStorageStub);
vi.mock('$app/environment', () => ({ browser: true }));

import {
	nextRecentLooks,
	RECENT_LOOKS_LIMIT,
	RECENT_LOOKS_STORAGE_KEY,
	recentLooks,
	recordRecentLook,
	reloadRecentLooks
} from './recentLooks';

const look = (id: string) => ({ id, name: id.toUpperCase(), group: 'Colour' });

beforeEach(() => {
	storage.clear();
	localStorageStub.getItem.mockClear();
	localStorageStub.setItem.mockClear();
	recentLooks.set([]);
});

afterEach(() => {
	vi.restoreAllMocks();
});

describe('nextRecentLooks', () => {
	it('puts the newest look first', () => {
		expect(nextRecentLooks([look('a')], look('b')).map((entry) => entry.id)).toEqual(['b', 'a']);
	});

	it('keeps at most three', () => {
		const list = [look('a'), look('b'), look('c')];
		expect(RECENT_LOOKS_LIMIT).toBe(3);
		expect(nextRecentLooks(list, look('d')).map((entry) => entry.id)).toEqual(['d', 'a', 'b']);
	});

	it('moves a look applied again to the front instead of listing it twice', () => {
		const list = [look('a'), look('b'), look('c')];
		expect(nextRecentLooks(list, look('c')).map((entry) => entry.id)).toEqual(['c', 'a', 'b']);
	});

	it('refreshes the stored name of a look applied again', () => {
		const renamed = { id: 'a', name: 'Renamed', group: 'Mine' };
		expect(nextRecentLooks([look('a')], renamed)).toEqual([renamed]);
	});
});

describe('recordRecentLook', () => {
	it('records an applied look in the store and in storage', () => {
		recordRecentLook(look('noir'));

		expect(get(recentLooks)).toEqual([look('noir')]);
		expect(JSON.parse(storage.get(RECENT_LOOKS_STORAGE_KEY)!)).toEqual([look('noir')]);
	});

	it('keeps only the last three looks applied', () => {
		for (const id of ['a', 'b', 'c', 'd']) recordRecentLook(look(id));

		expect(get(recentLooks).map((entry) => entry.id)).toEqual(['d', 'c', 'b']);
		expect(JSON.parse(storage.get(RECENT_LOOKS_STORAGE_KEY)!)).toHaveLength(3);
	});

	it('does not list a look twice when it is applied twice', () => {
		recordRecentLook(look('a'));
		recordRecentLook(look('b'));
		recordRecentLook(look('a'));

		expect(get(recentLooks).map((entry) => entry.id)).toEqual(['a', 'b']);
	});

	it('ignores a look with no id or no name', () => {
		recordRecentLook({ id: '', name: 'x', group: '' });
		recordRecentLook({ id: 'x', name: '', group: '' });

		expect(get(recentLooks)).toEqual([]);
		expect(localStorageStub.setItem).not.toHaveBeenCalled();
	});

	it('still remembers the look for this session when storage refuses the write', () => {
		localStorageStub.setItem.mockImplementation(() => {
			throw new DOMException('full', 'QuotaExceededError');
		});

		expect(() => recordRecentLook(look('a'))).not.toThrow();
		expect(get(recentLooks)).toEqual([look('a')]);
	});

	it('survives storage that throws on access, as in a blocked-cookies window', () => {
		vi.stubGlobal('localStorage', {
			get getItem(): never {
				throw new DOMException('blocked', 'SecurityError');
			},
			get setItem(): never {
				throw new DOMException('blocked', 'SecurityError');
			}
		});

		expect(() => recordRecentLook(look('a'))).not.toThrow();
		expect(() => reloadRecentLooks()).not.toThrow();
		expect(get(recentLooks)).toEqual([]);
		vi.stubGlobal('localStorage', localStorageStub);
	});
});

describe('reloadRecentLooks', () => {
	it('reads what an earlier visit recorded', () => {
		storage.set(RECENT_LOOKS_STORAGE_KEY, JSON.stringify([look('a'), look('b')]));

		reloadRecentLooks();

		expect(get(recentLooks)).toEqual([look('a'), look('b')]);
	});

	it('renders nothing for storage holding something else', () => {
		for (const raw of ['not json', '{"a":1}', 'null', '[1,2,"x"]', '[{"id":3}]']) {
			storage.set(RECENT_LOOKS_STORAGE_KEY, raw);
			reloadRecentLooks();
			expect(get(recentLooks)).toEqual([]);
		}
	});

	it('drops duplicates, nameless entries and anything past three', () => {
		storage.set(
			RECENT_LOOKS_STORAGE_KEY,
			JSON.stringify([look('a'), look('a'), { id: 'x', name: '' }, look('b'), look('c'), look('d')])
		);

		reloadRecentLooks();

		expect(get(recentLooks).map((entry) => entry.id)).toEqual(['a', 'b', 'c']);
	});

	it('reads a stored look that has no group as one with an empty group', () => {
		storage.set(RECENT_LOOKS_STORAGE_KEY, JSON.stringify([{ id: 'a', name: 'A' }]));

		reloadRecentLooks();

		expect(get(recentLooks)).toEqual([{ id: 'a', name: 'A', group: '' }]);
	});

	it('reads empty storage as no looks', () => {
		reloadRecentLooks();

		expect(get(recentLooks)).toEqual([]);
	});
});
