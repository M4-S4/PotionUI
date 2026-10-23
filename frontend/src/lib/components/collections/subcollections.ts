import type { CollectionLike } from './types';

export function hasSubcollections<T extends CollectionLike>(
	collections: T[],
	activeId: string | null | undefined
): boolean {
	return !!activeId && collections.some((c) => (c.parent_id ?? null) === activeId);
}

export function excludesSubcollections<T extends CollectionLike>(
	collections: T[],
	activeId: string | null | undefined,
	directOnly: boolean | undefined
): boolean {
	return !!directOnly && hasSubcollections(collections, activeId);
}
