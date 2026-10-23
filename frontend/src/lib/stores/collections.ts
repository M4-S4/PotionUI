import { api } from '$lib/services/api/index';
import type { CollectionScope } from '$lib/types/history';
import { createCollectionTreeStore } from './collectionTree';

function createCollectionsStore(scope: CollectionScope) {
	return createCollectionTreeStore(
		{
			async list(directOnly: boolean) {
				const response = await api.listCollections(scope, directOnly);
				if (!response.success || !response.data) return null;
				return { collections: response.data.collections, smartCounts: response.data.smart_counts ?? null };
			},
			create: (name: string, parentId?: string | null) => api.createCollection(name, scope, parentId),
			rename: (id: string, name: string) => api.renameCollection(id, name, scope),
			move: (id: string, parentId: string | null) => api.moveCollection(id, parentId, scope),
			bulkMove: (ids: string[], parentId: string | null) => api.bulkMoveCollections(ids, parentId, scope),
			remove: (id: string) => api.deleteCollection(id, scope),
			addMembers: (id: string, generationIds: string[]) => api.addToCollection(id, generationIds, scope)
		},
		{ label: 'collections' }
	);
}

export const historyCollectionsStore = createCollectionsStore('history');
export const libraryCollectionsStore = createCollectionsStore('library');
export const promptsCollectionsStore = createCollectionsStore('prompts');
