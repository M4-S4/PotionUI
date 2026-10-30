import { writable } from 'svelte/store';
import type { LibraryItem } from '$lib/services/api/library';

export interface LandingSource {
	generationId: string;
	fileIndex: number;
}

export interface LibraryModalRequest {
	item: LibraryItem;
	source: LandingSource | null;
}

export const libraryModalRequest = writable<LibraryModalRequest | null>(null);

export function openLibraryResultModal(item: LibraryItem, source: LandingSource | null): void {
	libraryModalRequest.set({ item, source });
}

export function closeLibraryResultModal(): void {
	libraryModalRequest.set(null);
}
