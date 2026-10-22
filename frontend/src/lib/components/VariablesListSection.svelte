<script lang="ts">
	import { DetailSection } from '$lib/components/detail';
	import { Button } from '$lib/components/ui';
	import { variableSummaryRows, type VariablesMap } from '$lib/utils/variableDefs';

	let {
		variables,
		emptyText,
		note,
		onManage
	}: {
		variables: VariablesMap;
		emptyText: string;
		note?: string;
		onManage: () => void;
	} = $props();

	const rows = $derived(variableSummaryRows(variables));
</script>

<DetailSection label="Variables">
	{#if rows.length === 0}
		<p class="text-xs text-fg-subtle">{emptyText}</p>
	{:else}
		<ul class="space-y-2" data-testid="variables-list">
			{#each rows as row (row.name)}
				<li class="flex items-center justify-between gap-2 text-xs">
					<span class="truncate font-mono text-signal">${row.name}</span>
					<span class="flex-shrink-0 text-fg-subtle">{row.kind}</span>
				</li>
			{/each}
		</ul>
	{/if}
	{#if note}
		<p class="mt-3 text-xs text-fg-subtle">{note}</p>
	{/if}
	<div class="mt-3">
		<Button size="xs" variant="secondary" onclick={onManage}>Manage variables</Button>
	</div>
</DetailSection>
