import { persistedStore } from '$lib/stores/persistedStore';

export type LibraryCardDensity = 'comfortable' | 'compact';

export const libraryCardDensity = persistedStore<LibraryCardDensity>(
	'admin-library-card-density-v2',
	(raw) => (raw === 'compact' ? 'compact' : 'comfortable'),
	'comfortable'
);
