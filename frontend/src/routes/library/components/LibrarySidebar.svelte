<script lang="ts">
	import { libraryCollectionsStore as collectionsStore } from '$lib/stores/collections';
	import { libraryStore } from '$lib/stores/library';
	import CollectionLibrarySidebar from '$lib/components/collections/CollectionLibrarySidebar.svelte';
	import type { SmartView, TreeActions } from '$lib/components/collections/types';
	import { buildSmartViews, createFilterSelection } from '$lib/components/collections/scopeActions';

	let allView: SmartView;
	let favoritesView: SmartView;
	let unsortedView: SmartView;

	export let onCollapse: () => void;

	$: collections = $collectionsStore.collections;
	$: smartCounts = $collectionsStore.smartCounts;
	$: activeId = $libraryStore.filters.collectionId;
	$: favoritesOnly = !!$libraryStore.filters.favoritesOnly;
	$: unsorted = !!$libraryStore.filters.unsorted;
	$: directOnly = !!$libraryStore.filters.directOnly;
	$: isAll = !activeId && !favoritesOnly && !unsorted;
	$: void collectionsStore.setDirectOnly(directOnly);

	const { applySelection, setDirectOnly } = createFilterSelection(
		(key, value) => libraryStore.setFilter(key, value),
		() => libraryStore.load()
	);

	async function handleDelete(id: string, blockedIds: Set<string>) {
		const response = await collectionsStore.remove(id);
		if (response.success) {
			const active = $libraryStore.filters.collectionId;
			if (active && blockedIds.has(active)) await applySelection({});
		}
		return response;
	}

	$: ({ allView, favoritesView, unsortedView } = buildSmartViews({
		icon: 'photo',
		label: 'All uploads',
		isAll,
		favoritesOnly,
		unsorted,
		counts: smartCounts,
		onAll: () => applySelection({}),
		onFavorites: () => applySelection({ favoritesOnly: true }),
		onUnsorted: () => applySelection({ unsorted: true })
	}));

	const treeActions: TreeActions = {
		onSelect: (id) => applySelection({ collectionId: id }),
		onRename: (id, name) => collectionsStore.rename(id, name),
		onCreate: (name, parentId) => collectionsStore.create(name, parentId),
		onDelete: handleDelete,
		onMove: (id, parentId) => collectionsStore.move(id, parentId),
		onBulkMove: (ids, parentId) => collectionsStore.bulkMove(ids, parentId)
	};
</script>

<CollectionLibrarySidebar
	storageKey="library-expanded-collections"
	rulesScope="library"
	{collections}
	{activeId}
	{allView}
	{favoritesView}
	{unsortedView}
	{directOnly}
	onDirectOnlyChange={setDirectOnly}
	autoOrganizeSubject="uploads"
	{treeActions}
	onCreateRoot={(name) => collectionsStore.create(name, null)}
	{onCollapse}
/>
