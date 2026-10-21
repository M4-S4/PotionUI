<script lang="ts">
	import LibraryCard, { stop } from '../components/LibraryCard.svelte';
	import LibraryCardMenu from '../components/LibraryCardMenu.svelte';
	import type { SegmentCategory } from '$lib/types/segments';
	import { Badge, Button } from '$lib/components/ui';
	import { timeAgo } from '$lib/utils/relativeTime';

	let {
		category,
		count,
		selected = false,
		onToggleSelect,
		onOpen,
		onNewSegment,
		onDelete
	}: {
		category: SegmentCategory;
		count: number;
		selected?: boolean;
		onToggleSelect: (category: SegmentCategory) => void;
		onOpen: (category: SegmentCategory) => void;
		onNewSegment: (category: SegmentCategory) => void;
		onDelete: (category: SegmentCategory) => void;
	} = $props();

	let menuOpen = $state(false);

	const menuItems = [
		{ id: 'new-segment', label: 'New segment here', icon: 'plus' },
		{ id: 'sep', label: '', separator: true },
		{ id: 'delete', label: 'Delete', icon: 'trash', danger: true }
	];

	function handleMenuSelect(id: string) {
		if (id === 'new-segment') onNewSegment(category);
		else if (id === 'delete') onDelete(category);
	}

	const footerLabel = $derived(count === 0 ? 'No segments' : count === 1 ? '1 segment' : `${count} segments`);
	const relativeLabel = $derived.by(() => {
		const when = category.updated_at ?? category.created_at;
		return when ? timeAgo(when) : null;
	});
</script>

<LibraryCard
	kind="category"
	label={category.name}
	{selected}
	onOpen={() => onOpen(category)}
	onToggleSelect={() => onToggleSelect(category)}
>
	<div class="flex min-w-0 flex-1 flex-col gap-1.5">
		<div class="flex min-w-0 items-center gap-2">
			<span class="h-2.5 w-2.5 flex-shrink-0 rounded-full" style="background: {category.color}"></span>
			<h3 class="min-w-0 flex-1 truncate text-sm font-semibold text-fg">{category.name}</h3>
			<Badge size="sm"><span class="font-mono tabular-nums">{count}</span></Badge>
		</div>

		{#if category.description}
			<p class="line-clamp-2 text-sm leading-relaxed text-fg-muted">{category.description}</p>
		{:else}
			<p class="text-sm italic text-fg-subtle">No description</p>
		{/if}

		<div class="mt-auto flex h-6 min-w-0 items-center gap-2.5" data-card-footer>
			<span class="whitespace-nowrap font-mono text-xs tabular-nums text-fg-subtle">{footerLabel}</span>
			<div class="ml-auto flex h-6 flex-shrink-0 items-center">
				{#if relativeLabel}
					<span
						class="whitespace-nowrap font-mono text-xs tabular-nums text-fg-subtle {menuOpen
							? 'hidden'
							: 'group-hover:hidden group-focus-within:hidden'}">{relativeLabel}</span
					>
				{/if}
				<div
					class="items-center gap-1 {menuOpen ? 'flex' : 'hidden group-hover:flex group-focus-within:flex'}"
					data-card-actions
				>
					<Button size="xs" variant="secondary" onclick={stop(() => onNewSegment(category))}>New segment</Button>
					<Button size="xs" variant="secondary" onclick={stop(() => onOpen(category))}>Edit</Button>
					<LibraryCardMenu items={menuItems} onselect={handleMenuSelect} bind:open={menuOpen} />
				</div>
			</div>
		</div>
	</div>
</LibraryCard>
