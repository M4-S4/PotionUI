<script lang="ts" generics="T extends CollectionLike">
	import FiltersPopover from './FiltersPopover.svelte';
	import IncludeSubcollections from './IncludeSubcollections.svelte';
	import { excludesSubcollections, hasSubcollections } from './subcollections';
	import type { CollectionLike } from './types';

	let {
		collections,
		activeId,
		directOnly,
		onDirectOnlyChange,
		class: className = ''
	}: {
		collections: T[];
		activeId: string | undefined;
		directOnly: boolean;
		onDirectOnlyChange: (directOnly: boolean) => void;
		class?: string;
	} = $props();

	let shown = $derived(hasSubcollections(collections, activeId));
	let excluding = $derived(excludesSubcollections(collections, activeId, directOnly));
</script>

{#if shown}
	<FiltersPopover
		class={className}
		activeCount={excluding ? 1 : 0}
		canClear={excluding}
		onClear={() => onDirectOnlyChange(false)}
	>
		{#snippet children()}
			<IncludeSubcollections {collections} {activeId} {directOnly} {onDirectOnlyChange} />
		{/snippet}
	</FiltersPopover>
{/if}
