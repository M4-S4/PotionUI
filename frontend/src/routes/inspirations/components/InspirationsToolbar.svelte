<script lang="ts">
	import { onDestroy } from 'svelte';
	import { debounce } from '$lib/utils/time';
	import { inspirationsStore } from '$lib/stores/inspirations';
	import { PageHeader, PageTitle, IconButton } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import CollectionSidebarSheet from '$lib/components/collections/CollectionSidebarSheet.svelte';
	import InspirationsSidebar from './InspirationsSidebar.svelte';
	import SubcollectionFilters from '$lib/components/collections/SubcollectionFilters.svelte';
	import { inspirationsCollectionsStore } from '$lib/stores/inspirationsCollections';

	export let sidebarOpen = true;
	export let onOpenSidebar: () => void = () => {};

	$: state = $inspirationsStore;

	function setDirectOnly(value: boolean) {
		inspirationsStore.setFilter('directOnly', value);
		inspirationsStore.load();
	}

	// Search matches title/description server-side, debounced - same pattern
	// as the Library toolbar.
	const scheduleSearch = debounce(() => inspirationsStore.load(), 300);
	function handleSearchInput(event: Event) {
		const value = (event.currentTarget as HTMLInputElement).value;
		inspirationsStore.setFilter('search', value);
		scheduleSearch();
	}

	onDestroy(scheduleSearch.cancel);

	async function toggleSaved() {
		inspirationsStore.setFilter('saved', !state.filters.saved);
		await inspirationsStore.load();
	}
</script>

<PageHeader wrap sticky={false}>
	<div class="flex flex-wrap items-center gap-2 md:gap-4 w-full">
		<CollectionSidebarSheet label="Inspirations" collapsed={!sidebarOpen} onExpand={onOpenSidebar}>
			<InspirationsSidebar />
		</CollectionSidebarSheet>
		<PageTitle title="Inspirations" count={state.totalCount} countLabel="items" />

		<div class="hidden md:block h-6 w-px bg-line-strong flex-shrink-0"></div>

		<!-- Search -->
		<div class="relative flex-1 min-w-[8rem] max-w-md">
			<Icon
				name="search"
				className="w-3.5 h-3.5 text-fg-subtle absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
			/>
			<input
				type="text"
				class="input text-xs py-1.5 pl-8 pr-3 bg-surface-2/50 w-full"
				placeholder="Search title or description..."
				value={state.filters.search}
				on:input={handleSearchInput}
			/>
		</div>

		<!-- Saved filter -->
		<button
			class="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded transition-colors duration-100 flex-shrink-0 {state
				.filters.saved
				? 'bg-signal/10 text-signal'
				: 'bg-surface-2/50 text-fg-muted hover:bg-surface-3/50 hover:text-fg'}"
			aria-pressed={state.filters.saved}
			on:click={toggleSaved}
		>
			<Icon name="save" className="w-3.5 h-3.5" />
			Saved
		</button>

		<SubcollectionFilters
			class="hidden md:block flex-shrink-0"
			collections={$inspirationsCollectionsStore.collections}
			activeId={state.filters.collectionId}
			directOnly={!!state.filters.directOnly}
			onDirectOnlyChange={setDirectOnly}
		/>

		<div class="flex-1 hidden md:block"></div>

		<!-- Right: actions -->
		<div class="flex items-center gap-2 ml-auto md:ml-0 flex-shrink-0">
			<InspirationsSidebar part="actions" />
			<IconButton icon="refresh" label="Refresh" onclick={() => inspirationsStore.load()} />
		</div>
	</div>
</PageHeader>
