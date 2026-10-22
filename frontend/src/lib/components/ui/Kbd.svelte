<script lang="ts">
	const TEXT_CLASSES = { sm: 'text-2xs', md: 'text-xs', lg: 'text-sm' } as const;

	let { keys, size = 'sm' }: { keys: string | string[]; size?: keyof typeof TEXT_CLASSES } = $props();

	let keyList = $derived(Array.isArray(keys) ? keys : [keys]);
	let textClass = $derived(TEXT_CLASSES[size]);
</script>

<span class="inline-flex items-center gap-1 font-mono tabular-nums {textClass} text-fg-muted" aria-hidden="true">
	{#each keyList as key, i (key + i)}
		{#if i > 0}<span aria-hidden="true">+</span>{/if}
		<kbd class="px-1.5 py-0.5 {textClass} font-mono tabular-nums rounded-sm bg-surface-3 text-fg-muted border border-line-hover">
			{key}
		</kbd>
	{/each}
</span>
