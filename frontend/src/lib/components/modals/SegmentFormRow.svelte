<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';

	let {
		label,
		forId,
		optional = false,
		problem = '',
		helper = '',
		class: className = '',
		children
	}: {
		label: string;
		forId?: string;
		optional?: boolean;
		problem?: string;
		helper?: string;
		class?: string;
		children: Snippet;
	} = $props();
</script>

<div class="min-w-0 {className}">
	{#if forId}
		<label class="mb-1.5 block text-sm font-medium text-fg-muted" for={forId}>
			{label}{#if optional}&nbsp;<span class="font-normal text-fg-subtle">(optional)</span>{/if}
		</label>
	{:else}
		<span class="mb-1.5 block text-sm font-medium text-fg-muted">
			{label}{#if optional}&nbsp;<span class="font-normal text-fg-subtle">(optional)</span>{/if}
		</span>
	{/if}
	{@render children()}
	{#if problem}
		<p class="mt-1.5 flex items-start gap-1.5 text-sm text-warning" data-testid="field-problem" role="status">
			<Icon name="warning" className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
			<span class="min-w-0">{problem}</span>
		</p>
	{:else if helper}
		<p class="mt-1.5 text-sm text-fg-subtle">{helper}</p>
	{/if}
</div>
