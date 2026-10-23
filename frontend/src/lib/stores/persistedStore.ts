import { writable, get } from 'svelte/store';
import { storage } from '$lib/utils/storage';

export function persistedStore<T>(
	key: string,
	parse: (raw: string) => T,
	fallback: T,
	serialize: (value: T) => string = String
) {
	let initial = fallback;
	const raw = storage.get(key);
	if (raw !== null) {
		try {
			initial = parse(raw);
		} catch {}
	}
	const store = writable<T>(initial);

	function set(value: T) {
		storage.set(key, serialize(value));
		store.set(value);
	}

	return {
		subscribe: store.subscribe,
		set,
		update(fn: (value: T) => T) {
			set(fn(get(store)));
		}
	};
}
