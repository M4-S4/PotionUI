import { writable } from 'svelte/store';

export const page = writable({
	url: new URL('http://localhost/generate'),
	params: {},
	route: { id: null },
	status: 200,
	error: null,
	data: {},
	form: undefined,
	state: {} as Record<string, unknown>
});
export const navigating = writable(null);
export const updated = writable(false);
