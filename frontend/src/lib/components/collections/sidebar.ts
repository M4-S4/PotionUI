export const COLLECTION_SHEET = Symbol('collection-sheet');

export interface CollectionSheet {
	close(): void;
}

export const MENU_ITEM_CLASS =
	'flex min-h-[38px] w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm hover:bg-surface-3 focus-visible:bg-surface-3 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50';
