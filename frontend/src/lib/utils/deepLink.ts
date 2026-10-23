export const GENERATION_PARAM = 'generation';

const UNSAFE_ID = /[/\\?#]|\.\./;

export function generationIdFromUrl(url: URL): string | null {
	const id = url.searchParams.get(GENERATION_PARAM)?.trim();
	if (!id || UNSAFE_ID.test(id)) return null;
	return id;
}

export function hrefWithGeneration(url: URL, generationId: string | null): string {
	const params = new URLSearchParams(url.searchParams);
	if (generationId) params.set(GENERATION_PARAM, generationId);
	else params.delete(GENERATION_PARAM);
	const query = params.toString();
	return `${url.pathname}${query ? `?${query}` : ''}${url.hash}`;
}
