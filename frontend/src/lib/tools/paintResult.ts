import { libraryStore } from '$lib/stores/library';
import { toasts } from '$lib/stores/toast';
import type { MediaEditorResult } from '$lib/media/editors/types';

export const PAINT_SAVED_AS_NEW = 'Saved to your library as a new image';
export const PAINT_REPLACED = 'Replaced — tags and collections kept';

export async function applyLibraryPaintResult(
	result: MediaEditorResult,
	sourceIds: string[]
): Promise<void> {
	if (result.type !== 'item' && result.type !== 'items') return;
	const items = result.type === 'item' ? [result.item] : result.items;
	const replaced = result.type === 'item' && result.replaced;

	await libraryStore.applyToolResult(items, sourceIds, { replaced });
	toasts.success(replaced ? PAINT_REPLACED : PAINT_SAVED_AS_NEW);
}
