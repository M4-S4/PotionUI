<script lang="ts">
	import LibraryCard, { stop } from '../components/LibraryCard.svelte';
	import LibraryCardMenu from '../components/LibraryCardMenu.svelte';
	import type { SavedSegment } from '$lib/types/segments';
	import { Badge, Button } from '$lib/components/ui';
	import { timeAgo } from '$lib/utils/relativeTime';

	let {
		segment,
		categoryName = null,
		selected = false,
		onToggleSelect,
		onOpen,
		onInsert,
		onDuplicate,
		onDelete
	}: {
		segment: SavedSegment;
		categoryName?: string | null;
		selected?: boolean;
		onToggleSelect: (segment: SavedSegment) => void;
		onOpen: (segment: SavedSegment) => void;
		onInsert: (segment: SavedSegment) => void;
		onDuplicate: (segment: SavedSegment) => void;
		onDelete: (segment: SavedSegment) => void;
	} = $props();

	let menuOpen = $state(false);

	const menuItems = [
		{ id: 'duplicate', label: 'Duplicate', icon: 'copy' },
		{ id: 'sep', label: '', separator: true },
		{ id: 'delete', label: 'Delete', icon: 'trash', danger: true }
	];

	function handleMenuSelect(id: string) {
		if (id === 'duplicate') onDuplicate(segment);
		else if (id === 'delete') onDelete(segment);
	}

	const color = $derived(segment.effective_color || segment.color || '#3B82F6');
	const tags = $derived((segment.tags ?? []).slice(0, 2));
	const relativeLabel = $derived.by(() => {
		const when = segment.updated_at ?? segment.created_at;
		return when ? timeAgo(when) : null;
	});
</script>

<LibraryCard
	kind="segment"
	label={segment.name}
	{selected}
	onOpen={() => onOpen(segment)}
	onToggleSelect={() => onToggleSelect(segment)}
>
	<div class="flex min-w-0 flex-1 flex-col gap-1.5">
		<div class="flex min-w-0 items-center gap-2">
			<span class="h-2.5 w-2.5 flex-shrink-0 rounded-full" style="background: {color}"></span>
			<h3 class="min-w-0 flex-1 truncate text-sm font-semibold text-fg">{segment.name}</h3>
			{#if !segment.enabled}
				<Badge size="sm">Disabled</Badge>
			{/if}
		</div>

		<p class="line-clamp-3 text-sm leading-relaxed {segment.content ? 'text-fg-muted' : 'italic text-fg-subtle'}">
			{segment.content || 'Empty starter content'}
		</p>

		<div class="mt-auto flex h-6 min-w-0 items-center gap-1.5" data-card-footer>
			{#if categoryName}
				<Badge size="sm">{categoryName}</Badge>
			{/if}
			{#each tags as tag (tag)}
				<Badge size="sm" class="font-mono">#{tag}</Badge>
			{/each}
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
					<Button size="xs" variant="primary" onclick={stop(() => onInsert(segment))}>Insert</Button>
					<Button size="xs" variant="secondary" onclick={stop(() => onOpen(segment))}>Edit</Button>
					<LibraryCardMenu items={menuItems} onselect={handleMenuSelect} bind:open={menuOpen} />
				</div>
			</div>
		</div>
	</div>
</LibraryCard>
