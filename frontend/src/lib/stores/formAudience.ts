import { get } from 'svelte/store';
import type { FormAudience } from '$lib/utils/audienceFilter';
import { persistedStore } from './persistedStore';

const store = persistedStore<FormAudience>(
	'potionui-form-audience',
	(raw) => (raw === 'advanced' ? 'advanced' : 'simple'),
	'simple'
);

export const formAudienceStore = {
	subscribe: store.subscribe,
	set: store.set,
	toggle() {
		store.set(get(store) === 'simple' ? 'advanced' : 'simple');
	}
};
