import { api } from '$lib/services/api/index';
import { createCollectionTreeStore } from './collectionTree';

export const modelLibraryStore = createCollectionTreeStore(
	{
		async list(directOnly: boolean) {
			const response = await api.listModelCollections(directOnly);
			if (!response.success || !response.data) return null;
			return { collections: response.data.collections, smartCounts: response.data.smart_counts ?? null };
		},
		create: (name: string, parentId?: string | null) => api.createModelCollection(name, parentId),
		rename: (id: string, name: string) => api.renameModelCollection(id, name),
		move: (id: string, parentId: string | null) => api.moveModelCollection(id, parentId),
		bulkMove: (ids: string[], parentId: string | null) => api.bulkMoveModelCollections(ids, parentId),
		remove: (id: string) => api.deleteModelCollection(id),
		addMembers: (id: string, modelIds: string[]) => api.addToModelCollection(id, modelIds),
		removeMembers: (id: string, modelIds: string[]) => api.removeFromModelCollection(id, modelIds)
	},
	{ label: 'model collections', cached: true }
);
