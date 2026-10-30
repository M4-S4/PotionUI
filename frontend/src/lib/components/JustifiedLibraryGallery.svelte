<script lang="ts">
	import {
		layoutJustifiedRows,
		flattenJustifiedRows,
		type FlatJustifiedBox
	} from '$lib/utils/justifiedLayout';
	import { libraryItemAspect } from '$lib/library/libraryItemMeta';
	import type { LibraryItem } from '$lib/services/api/library';

	export let items: LibraryItem[] = [];

	const GAP = 12;
	let width = 0;

	$: targetRowHeight = Math.round(width > 0 ? Math.max(180, Math.min(320, width / 4.6)) : 260);

	$: boxes = flattenJustifiedRows(
		layoutJustifiedRows(
			items.map((item) => ({ item, aspect: libraryItemAspect(item) })),
			width,
			targetRowHeight,
			GAP
		)
	) as FlatJustifiedBox<LibraryItem>[];
</script>

<div bind:clientWidth={width}>
	{#if width > 0}
		<div class="flex flex-wrap" style="gap: {GAP}px" data-testid="library-mosaic">
			{#each boxes as box (box.item.id)}
				<div
					data-row-end={box.rowEnd}
					style="flex: {box.rowEnd ? '1 1' : '0 0'} {box.width}px"
				>
					<slot name="card" item={box.item} tile={{ width: box.width, height: box.height }} />
				</div>
			{/each}
		</div>
	{/if}
</div>
