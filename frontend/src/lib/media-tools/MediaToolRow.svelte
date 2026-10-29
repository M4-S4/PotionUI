<script lang="ts" module>
	export interface RowPageAction {
		id: string;
		label: string;
		icon: string;
		onclick: () => void;
		tone?: 'danger';
		disabled?: boolean;
		loading?: boolean;
		tooltip?: string;
	}
</script>

<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { Button, Menu } from '$lib/components/ui';
	import MediaEditors from '$lib/media/editors/MediaEditors.svelte';
	import type {
		MediaEditorRequest,
		MediaEditorResult
	} from '$lib/media/editors/types';
	import { editImageRequest } from '$lib/tools/editImage';
	import { loadPluginMediaTools, mediaToolRegistrations } from '$lib/tools/pluginTools';
	import { buildRowContext, buildRowModel, type RowTool } from '$lib/tools/rowModel';
	import type { MediaToolItem as RowItem } from '$lib/tools/tools';
	import type { MediaToolItem as ToolResultItem } from './types';
	import { openMediaTool } from './open';
	import { SAVED_TO_LIBRARY_ANNOUNCE } from './announce';
	import { notifySavedToLibrary } from '$lib/tools/savedToLibrary';

	let {
		items,
		scope,
		pageLabel,
		pageActions = [],
		onResult,
		onEditingChange
	}: {
		items: RowItem[];
		scope: 'history' | 'library';
		pageLabel: string;
		pageActions?: RowPageAction[];
		onResult?: (items: ToolResultItem[]) => void;
		onEditingChange?: (editing: boolean) => void;
	} = $props();

	let paintRequest = $state<MediaEditorRequest | null>(null);
	let moreOpen = $state(false);

	let model = $derived.by(() => {
		void $mediaToolRegistrations;
		return buildRowModel(buildRowContext(scope, items));
	});

	$effect(() => {
		const editing = paintRequest !== null;
		untrack(() => onEditingChange?.(editing));
	});

	onMount(() => {
		void loadPluginMediaTools();
	});

	function openPaint(initialTool: string | null) {
		const request = editImageRequest(buildRowContext(scope, items));
		if (!request) return;
		paintRequest = initialTool ? { ...request, initialTool } : request;
	}

	function run(entry: RowTool) {
		if (!entry.enabled) return;
		if (entry.action === 'edit-image') return openPaint(null);
		if (entry.action === 'apply-look') return openPaint('filters');
		const toolId = entry.tool?.catalogToolId;
		if (!toolId) return;
		void openMediaTool({
			toolId,
			items,
			scope,
			announce: scope === 'history' ? SAVED_TO_LIBRARY_ANNOUNCE : undefined,
			onResult: (results: ToolResultItem[]) => onResult?.(results)
		});
	}

	function handlePaintResult(result: MediaEditorResult) {
		const saved = result.type === 'item' ? [result.item] : result.type === 'items' ? result.items : [];
		if (scope === 'history') notifySavedToLibrary(saved);
		if (saved.length > 0) onResult?.(saved);
	}

	function subline(entry: RowTool): string {
		return entry.enabled ? entry.note : entry.reason;
	}

	const labelClass = 'font-mono text-[12px] leading-4 uppercase tracking-[0.07em] text-fg-subtle';
</script>

{#if model.editButtons.length > 0 || model.moreTools.length > 0 || model.effects.length > 0 || pageActions.length > 0}
	<div class="flex flex-col gap-2" role="toolbar" aria-label="Edit tools" data-media-tool-row>
		{#if model.editButtons.length > 0 || model.moreTools.length > 0}
			<div class="flex flex-wrap items-center gap-2" role="group" aria-label="Edit" data-row-group="edit">
				<span class={labelClass}>Edit</span>
				{#each model.editButtons as entry (entry.id)}
					<Tooltip text={entry.enabled ? entry.note : entry.reason}>
						<Button
							variant="secondary"
							size="sm"
							icon={entry.icon}
							disabled={!entry.enabled}
							onclick={() => run(entry)}
						>
							{entry.label}
						</Button>
					</Tooltip>
				{/each}
				{#if model.moreTools.length > 0}
					<Menu label="More tools" menuLabel="More tools" width={256} bind:open={moreOpen} align="left">
						{#snippet trigger({ open, toggle })}
							<Button
								variant="secondary"
								size="sm"
								ariaExpanded={open}
								ariaHaspopup="menu"
								onclick={toggle}
							>
								More tools
								<Icon name="chevron-down" className="w-3 h-3" />
							</Button>
						{/snippet}
						{#snippet children({ close })}
							{#each model.moreTools as entry (entry.id)}
								<button
									type="button"
									role="menuitem"
									class="flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-sm focus-visible:outline-none {entry.enabled
										? 'text-fg hover:bg-surface-3 focus-visible:bg-surface-3'
										: 'cursor-not-allowed text-fg-muted opacity-40'}"
									aria-disabled={!entry.enabled}
									data-row-tool={entry.id}
									onclick={() => {
										if (!entry.enabled) return;
										close(true);
										run(entry);
									}}
								>
									<Icon name={entry.icon} className="h-4 w-4 flex-shrink-0" />
									<span class="flex min-w-0 flex-1 flex-col">
										<span class="truncate">{entry.label}</span>
										{#if subline(entry)}
											<span class="truncate text-[12px] leading-4 text-fg-subtle">{subline(entry)}</span>
										{/if}
									</span>
								</button>
							{/each}
						{/snippet}
					</Menu>
				{/if}
			</div>
		{/if}

		{#if model.effects.length > 0}
			<div class="flex flex-wrap items-center gap-2" role="group" aria-label="Effects" data-row-group="effects">
				<span class={labelClass}>Effects</span>
				{#each model.effects as entry (entry.id)}
					<Tooltip text={entry.enabled ? '' : entry.reason}>
						<Button
							variant="secondary"
							size="sm"
							icon={entry.icon}
							disabled={!entry.enabled}
							onclick={() => run(entry)}
						>
							{entry.label}
						</Button>
					</Tooltip>
				{/each}
			</div>
		{/if}

		{#if pageActions.length > 0}
			<div
				class="flex flex-wrap items-center gap-2"
				role="group"
				aria-label="{pageLabel} actions"
				data-row-group="page"
			>
				<span class={labelClass}>{pageLabel}</span>
				{#each pageActions as action (action.id)}
					<Tooltip text={action.tooltip ?? ''}>
						<Button
							variant={action.tone === 'danger' ? 'danger' : 'secondary'}
							size="sm"
							icon={action.icon}
							disabled={action.disabled}
							loading={action.loading}
							onclick={action.onclick}
						>
							{action.label}
						</Button>
					</Tooltip>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<MediaEditors
	request={paintRequest}
	onClose={() => (paintRequest = null)}
	onResult={handlePaintResult}
/>
