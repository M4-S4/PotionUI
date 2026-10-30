<script lang="ts">
	import Tooltip from '$lib/components/Tooltip.svelte';
	import ToolsMenu from '$lib/components/tools/ToolsMenu.svelte';
	import type { EntryMenuModel } from '$lib/tools/entryMenu';
	import type { MediaTool, MediaToolContext, MediaToolScope, ToolRunExtra } from '$lib/tools/tools';
	import { getIconPath } from '$lib/utils/IconLibrary';

	const starPath = getIconPath('star') as string;

	let {
		scope,
		getModel,
		onPick,
		favorite = null,
		onFavorite,
		favoriteInRow = true,
		chipSize = 26
	}: {
		scope: MediaToolScope;
		getModel: () => EntryMenuModel;
		onPick: (tool: MediaTool, ctx: MediaToolContext, extra?: ToolRunExtra) => void;
		favorite?: boolean | null;
		onFavorite?: () => void;
		favoriteInRow?: boolean;
		chipSize?: number;
	} = $props();

	let open = $state(false);
	let sheet = $state(false);
	let anchorPoint = $state<{ x: number; y: number } | null>(null);
	let model = $state<EntryMenuModel | null>(null);

	function touchPrimary(): boolean {
		return typeof window !== 'undefined' && !!window.matchMedia?.('(hover: none)').matches;
	}

	function show(point: { x: number; y: number } | null, asSheet: boolean) {
		model = getModel();
		anchorPoint = point;
		sheet = asSheet;
		open = true;
	}

	export function openAt(x: number, y: number) {
		show({ x, y }, touchPrimary());
	}

	export function openSheet() {
		show(null, true);
	}

	export function close() {
		open = false;
	}

	function toggle() {
		if (open) {
			open = false;
			return;
		}
		show(null, touchPrimary());
	}

	function handlePick(tool: MediaTool, extra?: ToolRunExtra) {
		if (model) onPick(tool, model.ctx, extra);
	}

	let revealClass = $derived(
		open
			? 'opacity-100'
			: 'opacity-0 group-hover:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100'
	);
</script>

<div class="absolute top-2 right-2 z-30 flex items-center gap-1" data-entry-actions>
	{#if favorite !== null && favoriteInRow}
		<Tooltip text={favorite ? 'Remove favorite' : 'Favorite'} position="bottom" delay={150}>
			<button
				type="button"
				class="inline-flex items-center justify-center rounded border border-line-strong bg-surface-2 shadow-raised hover:border-line-hover hover:bg-surface-3 transition-[opacity,background-color,border-color,color] duration-100 {favorite
					? 'text-signal opacity-100'
					: `text-fg ${revealClass}`}"
				onclick={(event) => {
					event.stopPropagation();
					event.preventDefault();
					onFavorite?.();
				}}
				aria-label={favorite ? 'Remove favorite' : 'Favorite'}
				aria-pressed={favorite}
				data-entry-favorite
				style="width: {chipSize}px; height: {chipSize}px"
			>
				<svg
					class="h-3.5 w-3.5"
					viewBox="0 0 24 24"
					fill={favorite ? 'currentColor' : 'none'}
					stroke="currentColor"
					stroke-width="2"
					aria-hidden="true"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d={starPath} />
				</svg>
			</button>
		</Tooltip>
	{/if}
	<ToolsMenu
		{open}
		{scope}
		variant="entry"
		groups={model?.groups ?? []}
		ctx={model?.ctx ?? null}
		{anchorPoint}
		{sheet}
		header={model && model.selectionCount > 1 ? `${model.selectionCount} selected` : ''}
		title={model?.title ?? ''}
		subtitle={model?.subtitle ?? ''}
		rating={model?.rating ?? 0}
		collections={model?.collections ?? []}
		{chipSize}
		placement="down"
		align="right"
		triggerClass="{revealClass} [@media(hover:none)]:!w-8 [@media(hover:none)]:!h-8"
		onToggle={toggle}
		onClose={() => (open = false)}
		onPick={handlePick}
	/>
</div>
