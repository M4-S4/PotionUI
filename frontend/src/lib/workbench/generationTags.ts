import { api } from '$lib/services/api/index';

export interface TagChip {
	id: string;
	name: string;
}

export function tagChips(tags: ReadonlyArray<TagChip> | null | undefined): TagChip[] {
	return (tags ?? []).map(({ id, name }) => ({ id, name }));
}

export async function fetchGenerationTags(generationId: string): Promise<TagChip[] | null> {
	const response = await api.getGenerationById(generationId, true, false);
	if (!response.success || !response.data?.tags) return null;
	return tagChips(response.data.tags);
}

export async function saveGenerationTags(generationId: string, tagIds: string[]) {
	const response = await api.updateGenerationTags(generationId, tagIds);
	if (!response.success || !response.data) return null;
	return response.data.tags;
}
