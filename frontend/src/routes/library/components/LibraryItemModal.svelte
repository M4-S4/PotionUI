<script lang="ts">
	import tooltip from '$lib/actions/tooltip';
	import BaseModal from '$lib/components/modals/BaseModal.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { parseServerDate } from '$lib/utils/relativeTime';
	import AudioPlayer from '$lib/components/AudioPlayer.svelte';
	import TagSelector from '$lib/components/TagSelector.svelte';
	import FavoriteButton from '$lib/components/FavoriteButton.svelte';
	import { Badge } from '$lib/components/ui';
	import { libraryStore, LIBRARY_TAG_TYPE } from '$lib/stores/library';
	import { formatBytes, formatSeconds } from '$lib/utils/format';
	import { libraryItemDisplayName, libraryItemIcon } from '$lib/library/libraryItemMeta';
	import MediaToolRow, { type RowPageAction } from '$lib/media-tools/MediaToolRow.svelte';
	import { buildLibraryToolContext } from '$lib/tools/tools';
	import { toasts } from '$lib/stores/toast';
	import type { EditedMediaItem } from '$lib/services/api/media';
	import type { LibraryItem } from '$lib/services/api/library';
	import ProvenanceLine from '$lib/components/organize/ProvenanceLine.svelte';
	import { startRuleFrom } from '$lib/organize/handoff';
	import { uploadItemRule } from '$lib/organize/itemPrefill';

	export let item: LibraryItem;
	export let onClose: () => void;
	export let onDeleteRequest: (item: LibraryItem) => void;

	$: displayName = libraryItemDisplayName(item);
	$: kind = (item.media_type ?? '').toLowerCase();
	$: selectedTagIds = item.tags.map((tag) => tag.id);
	$: audioTracks = [{ type: 'mixed' as const, url: item.url, duration: item.duration_seconds }];

	$: details = [
		item.width && item.height ? { label: 'Dimensions', value: `${item.width}×${item.height}` } : null,
		typeof item.duration_seconds === 'number'
			? { label: 'Duration', value: formatSeconds(item.duration_seconds) }
			: null,
		typeof item.fps === 'number' && item.fps > 0
			? { label: 'FPS', value: String(Math.round(item.fps)) }
			: null,
		typeof item.size === 'number' && item.size > 0
			? { label: 'Size', value: formatBytes(item.size) }
			: null,
		item.created_at
			? { label: 'Added', value: parseServerDate(item.created_at)?.toLocaleString() ?? 'Unknown' }
			: null,
		item.mime_type ? { label: 'Type', value: item.mime_type } : null
	].filter((entry): entry is { label: string; value: string } => entry !== null);

	function handleDownload() {
		const link = document.createElement('a');
		link.href = item.url;
		link.download = displayName;
		link.click();
	}

	async function handleTagsChange(event: CustomEvent<string[]>) {
		await libraryStore.setItemTags(item.id, event.detail);
	}

	async function handleFavoriteToggle() {
		const ok = await libraryStore.setItemFavorite(item.id, !item.is_favorite);
		if (!ok) toasts.error('Could not update the favorite');
	}

	let toolEditing = false;

	$: rowItems = buildLibraryToolContext([item], null).items;

	let libraryActions: RowPageAction[] = [];
	$: libraryActions = [
		{
			id: 'auto-file',
			label: 'Auto-file items like this…',
			icon: 'sparkles',
			onclick: () => startRuleFrom(uploadItemRule(item))
		},
		{
			id: 'delete',
			label: 'Delete',
			icon: 'trash',
			tone: 'danger',
			onclick: () => onDeleteRequest(item)
		},
		{ id: 'download', label: 'Download', icon: 'download', onclick: handleDownload }
	];

	async function handleToolResult(results: EditedMediaItem[]) {
		await libraryStore.applyToolResult(results, [item.id]);
	}
</script>

<BaseModal isOpen={true} size="xl" title={displayName} closeable={!toolEditing} on:close={onClose}>
	<div class="p-4 md:p-6 flex flex-col gap-4">
		<div class="relative rounded-lg border border-line-strong bg-black overflow-hidden flex items-center justify-center min-h-[16rem] max-h-[60vh]">
			<div
				data-library-favorite
				class="absolute top-3 right-3 z-20 flex items-center rounded-lg bg-black/50 p-3 shadow-lg backdrop-blur-sm transition-colors hover:bg-black/70"
			>
				<FavoriteButton
					active={!!item.is_favorite}
					tone="onMedia"
					size="lg"
					onToggle={handleFavoriteToggle}
				/>
			</div>
			{#if kind === 'video'}
				<video src={item.url} class="max-h-[60vh] max-w-full" controls playsinline>
					<track kind="captions" />
				</video>
			{:else if kind === 'audio'}
				<div class="w-full p-6 bg-surface-1">
					<AudioPlayer tracks={audioTracks} showWaveform={true} showDownload={false} />
				</div>
			{:else}
				<img src={item.url} alt={displayName} class="max-h-[60vh] max-w-full object-contain" />
			{/if}
		</div>

		<MediaToolRow
			items={rowItems}
			scope="library"
			pageLabel="Library"
			pageActions={libraryActions}
			onResult={handleToolResult}
			onEditingChange={(editing) => (toolEditing = editing)}
		/>

		<div class="flex flex-wrap items-center gap-2">
			<Badge variant="neutral" class="inline-flex items-center gap-1.5">
				<Icon name={libraryItemIcon(item.media_type)} className="w-3 h-3" />
				{kind || 'media'}
			</Badge>
			<div class="flex-1"></div>
			<TagSelector
				{selectedTagIds}
				tagType={LIBRARY_TAG_TYPE}
				allowCreate={true}
				triggerStyle="pills"
				placeholder="Add tags..."
				on:change={handleTagsChange}
			/>
		</div>

		{#if details.length > 0}
			<dl class="grid grid-cols-2 md:grid-cols-3 gap-3">
				{#each details as detail (detail.label)}
					<div class="rounded border border-line bg-surface-2/50 px-3 py-2 min-w-0">
						<dt class="font-mono text-2xs uppercase tracking-[0.07em] text-fg-subtle">
							{detail.label}
						</dt>
						<dd class="mt-0.5 font-mono tabular-nums text-xs text-fg truncate" use:tooltip={{ text: detail.value, truncateOnly: true }}>
							{detail.value}
						</dd>
					</div>
				{/each}
			</dl>
		{/if}

		<ProvenanceLine itemType="upload" itemId={item.id} />
	</div>
</BaseModal>
