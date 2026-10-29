import { goto } from '$app/navigation';
import { toasts } from '$lib/stores/toast';
import type { EditedMediaItem } from '$lib/services/api/media';

export function savedToLibraryMessage(items: readonly EditedMediaItem[]): string {
	if (items.length === 1) return `${items[0].original_filename ?? items[0].filename} is in your Library.`;
	return `${items.length} items are in your Library.`;
}

export function notifySavedToLibrary(items: readonly EditedMediaItem[]): void {
	if (items.length === 0) return;
	toasts.show('success', savedToLibraryMessage(items), {
		title: 'Saved to Library',
		action: { label: 'Open Library', onClick: () => void goto('/library') }
	});
}
