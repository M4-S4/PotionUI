<script lang="ts">
	import LibraryCard, { stop } from './LibraryCard.svelte';
	import LibraryCardMenu from './LibraryCardMenu.svelte';
	import type { Prompt } from '$lib/types/segments';
	import { Button } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { parseVariableUsageTokens } from '$lib/utils/promptVariables';
	import { timeAgo } from '$lib/utils/relativeTime';
	import { promptCardTitle, summarizePromptVariables } from '$lib/prompts/promptCardDisplay';

	let {
		prompt,
		selected = false,
		onToggleSelect,
		onOpen,
		onUse,
		onCopy,
		onDuplicate,
		onAddToCollection,
		onExport,
		onDelete
	}: {
		prompt: Prompt;
		selected?: boolean;
		onToggleSelect: (prompt: Prompt) => void;
		onOpen: (prompt: Prompt) => void;
		onUse: (prompt: Prompt) => void;
		onCopy: (prompt: Prompt) => void;
		onDuplicate: (prompt: Prompt) => void;
		onAddToCollection: (prompt: Prompt) => void;
		onExport: (prompt: Prompt) => void;
		onDelete: (prompt: Prompt) => void;
	} = $props();

	let menuOpen = $state(false);

	const menuItems = [
		{ id: 'copy', label: 'Copy prompt', icon: 'copy' },
		{ id: 'duplicate', label: 'Duplicate', icon: 'copy' },
		{ id: 'collection', label: 'Add to collection', icon: 'folder' },
		{ id: 'export', label: 'Export', icon: 'download' },
		{ id: 'sep', label: '', separator: true },
		{ id: 'delete', label: 'Delete', icon: 'trash', danger: true }
	];

	function handleMenuSelect(id: string) {
		if (id === 'copy') onCopy(prompt);
		else if (id === 'duplicate') onDuplicate(prompt);
		else if (id === 'collection') onAddToCollection(prompt);
		else if (id === 'export') onExport(prompt);
		else if (id === 'delete') onDelete(prompt);
	}

	const title = $derived(promptCardTitle(prompt));
	const negative = $derived(prompt.usage_hint === 'negative');
	const thumbnailUrl = $derived(prompt.cover_thumbnail || null);
	const textTokens = $derived(parseVariableUsageTokens(prompt.flattened_text || ''));
	const variablesSummary = $derived(summarizePromptVariables(prompt.variables));
	const sourceLabel = $derived(
		prompt.source_provider && prompt.source_provider !== 'manual' ? prompt.source_provider : null
	);
	const usageLabel = $derived(
		prompt.usage_count ? `used ${prompt.usage_count}×` : 'never used'
	);
	const relativeLabel = $derived.by(() => {
		const when = prompt.last_used_at || prompt.created_at;
		return when ? timeAgo(when) : null;
	});
</script>

<LibraryCard
	kind="prompt"
	label={title.text}
	{selected}
	onOpen={() => onOpen(prompt)}
	onToggleSelect={() => onToggleSelect(prompt)}
>
	{#if thumbnailUrl}
		<div class="relative h-24 w-24 flex-shrink-0 overflow-hidden rounded bg-surface-3">
			<img src={thumbnailUrl} alt="" class="h-full w-full object-cover" loading="lazy" decoding="async" />
			{#if prompt.generation_count}
				<span
					class="absolute bottom-1 left-1 rounded bg-black/55 px-1.5 py-0 font-mono text-xs text-white backdrop-blur-sm"
				>
					{prompt.generation_count} gens
				</span>
			{/if}
		</div>
	{/if}

	<div class="flex min-w-0 flex-1 flex-col gap-1.5">
		<div class="flex min-w-0 items-center gap-2">
			<Tooltip text={negative ? 'Negative prompt' : 'Positive prompt'}>
				<span
					class="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded border font-mono text-xs {negative
						? 'border-danger/35 text-danger'
						: 'border-line-strong bg-surface-2 text-fg-muted'}"
				>
					{negative ? '−' : '+'}
				</span>
			</Tooltip>
			<h3
				class="min-w-0 flex-1 truncate text-sm font-semibold {title.untitled
					? 'font-medium italic text-fg-muted'
					: 'text-fg'}"
			>
				{title.text}
			</h3>
			<div class="flex flex-shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
				<Button size="xs" variant="primary" onclick={stop(() => onUse(prompt))}>Use</Button>
				<Button size="xs" variant="secondary" onclick={stop(() => onOpen(prompt))}>Edit</Button>
				<LibraryCardMenu items={menuItems} onselect={handleMenuSelect} bind:open={menuOpen} />
			</div>
		</div>

		<p class="line-clamp-4 text-sm leading-relaxed text-fg-muted">
			{#each textTokens as token, index (index)}
				{#if token.type === 'variable'}<span class="font-mono text-xs text-signal">{token.raw}</span
					>{:else}{token.raw}{/if}
			{/each}
		</p>

		<div class="mt-auto flex flex-wrap items-center gap-1.5">
			<span
				class="inline-flex h-5 max-w-full items-center gap-1 truncate rounded border border-line-strong bg-surface-2 px-1.5 font-mono text-xs text-fg-muted"
			>
				{prompt.model_name || 'any model'}
			</span>
			{#if sourceLabel}
				<span class="inline-flex h-5 items-center rounded border border-line-strong bg-surface-2 px-1.5 font-mono text-xs text-fg-muted">
					{sourceLabel}
				</span>
			{/if}
			{#if prompt.nsfw}
				<span class="inline-flex h-5 items-center rounded border border-line-strong bg-surface-2 px-1.5 font-mono text-xs text-fg-muted">
					nsfw
				</span>
			{/if}
			{#if variablesSummary.count > 0}
				<Tooltip text={variablesSummary.broken ? 'A condition references a missing variable' : 'Variables used in this prompt'}>
					<span
						class="inline-flex h-5 items-center gap-1 rounded border px-1.5 font-mono text-xs {variablesSummary.broken
							? 'border-warning/35 bg-warning/10 text-warning'
							: 'border-signal/28 bg-signal/10 text-signal'}"
					>
						<Icon name="braces" className="h-2.5 w-2.5" />
						{variablesSummary.count}{variablesSummary.linked > 0 ? ` · ${variablesSummary.linked} linked` : ''}
					</span>
				</Tooltip>
			{/if}
			<span class="ml-auto flex items-center gap-2.5 whitespace-nowrap font-mono text-xs tabular-nums text-fg-subtle">
				<span>{usageLabel}</span>
				{#if relativeLabel}<span>{relativeLabel}</span>{/if}
			</span>
		</div>
	</div>
</LibraryCard>
