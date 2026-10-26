<script lang="ts">
	import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';
	import { Badge, ChoiceCard } from '$lib/components/ui';
	import { formatBytes } from '$lib/utils/format';
	import { starterFit, starterNeedText, starterSourceText } from '$lib/utils/setupGuide';

	let {
		recipe,
		machine,
		checked,
		tabbable,
		recommended = false,
		onselect
	}: {
		recipe: SetupRecipe;
		machine: SetupMachine | null;
		checked: boolean;
		tabbable: boolean;
		recommended?: boolean;
		onselect: () => void;
	} = $props();

	const fit = $derived(starterFit(recipe, machine));
	const fitClass = { good: 'text-success', low: 'text-warning', dim: 'text-fg-muted' };
</script>

<ChoiceCard
	{checked}
	{tabbable}
	title={recipe.name}
	badge={recommended ? 'Recommended for your card' : undefined}
	description={recipe.blurb ?? recipe.summary}
	{onselect}
>
	<span
		class="mt-2 grid grid-cols-4 gap-3 border-t border-line pt-2.5 max-sm:grid-cols-2"
		data-spec
	>
		<span class="min-w-0">
			<span class="block font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">Download</span>
			<span class="mt-0.5 block font-mono tabular-nums text-fg" data-spec-download>
				{recipe.total_download_bytes != null ? formatBytes(recipe.total_download_bytes) : 'Not stated'}
			</span>
		</span>
		<span class="min-w-0">
			<span class="block font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">From</span>
			<span class="mt-0.5 block font-mono text-fg [overflow-wrap:anywhere]" data-spec-source>
				{starterSourceText(recipe)}
			</span>
		</span>
		<span class="min-w-0">
			<span class="block font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">Graphics memory</span>
			<span class="mt-0.5 block font-mono tabular-nums text-fg [overflow-wrap:anywhere]" data-spec-vram>
				{starterNeedText(recipe)}
			</span>
		</span>
		<span class="min-w-0">
			<span class="block font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">Your card</span>
			<span class="mt-0.5 block {fitClass[fit.tone]}" data-spec-fit>{fit.text}</span>
		</span>
	</span>
	{#if recipe.last_completed_at}
		<span class="mt-1"><Badge size="sm" variant="success">Installed</Badge></span>
	{/if}
</ChoiceCard>
