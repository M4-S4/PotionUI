export const LIBRARY_ITEM_PARAM = 'item';

const UNSAFE_ID = /[/\\?#]|\.\./;

export function libraryItemIdFromUrl(url: URL): string | null {
	const id = url.searchParams.get(LIBRARY_ITEM_PARAM)?.trim();
	if (!id || UNSAFE_ID.test(id)) return null;
	return id;
}

export function hrefWithLibraryItem(url: URL, itemId: string | null): string {
	const params = new URLSearchParams(url.searchParams);
	if (itemId) params.set(LIBRARY_ITEM_PARAM, itemId);
	else params.delete(LIBRARY_ITEM_PARAM);
	const query = params.toString();
	return `${url.pathname}${query ? `?${query}` : ''}${url.hash}`;
}
