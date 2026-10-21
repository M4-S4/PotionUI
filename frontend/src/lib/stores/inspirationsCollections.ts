import { api } from '$lib/services/api/index';
import { createCollectionTreeStore } from './collectionTree';

async function forEachInspiration(
	collectionId: string,
	inspirationIds: string[],
	run: (collectionId: string, inspirationId: string) => ReturnType<typeof api.addInspirationToCollection>
) {
	let last: Awaited<ReturnType<typeof run>> = { success: true };
	for (const inspirationId of inspirationIds) {
		last = await run(collectionId, inspirationId);
		if (!last.success) return last;
	}
	return last;
}

export const inspirationsCollectionsStore = createCollectionTreeStore(
	{
		async list(directOnly: boolean) {
			const response = await api.listInspirationCollections(directOnly);
			if (!response.success || !response.data) return null;
			return { collections: response.data.items, smartCounts: response.data.smart_counts ?? null };
		},
		create: (name: string, parentId?: string | null) => api.createInspirationCollection(name, parentId),
		rename: (id: string, name: string) => api.updateInspirationCollection(id, { name }),
		move: (id: string, parentId: string | null) => api.updateInspirationCollection(id, { parent_id: parentId }),
		remove: (id: string) => api.deleteInspirationCollection(id),
		addMembers: (id: string, inspirationIds: string[]) =>
			forEachInspiration(id, inspirationIds, api.addInspirationToCollection),
		removeMembers: (id: string, inspirationIds: string[]) =>
			forEachInspiration(id, inspirationIds, api.removeInspirationFromCollection)
	},
	{ label: 'inspiration collections' }
);
