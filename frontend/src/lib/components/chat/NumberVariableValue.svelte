<script lang="ts">
	import Badge from '$lib/components/ui/Badge.svelte';
	import { NUMBER_MODE_LABEL, type NumberVariableSummary } from '$lib/chat/conditionText';

	let { summary }: { summary: NumberVariableSummary } = $props();
</script>

<span class="inline-flex flex-wrap items-center gap-1.5" data-testid="number-variable-value">
	<span class="font-mono tabular-nums text-fg">{summary.min}&ndash;{summary.max}</span>
	{#if summary.step}
		<span class="font-mono text-sm tabular-nums text-fg-muted">step {summary.step}</span>
	{/if}
	<Badge variant={summary.mode === 'pin' ? 'signal' : 'neutral'}>
		{NUMBER_MODE_LABEL[summary.mode]}{#if summary.mode === 'pin' && summary.pinned !== null}
			<span class="ml-1 font-mono tabular-nums">{summary.pinned}</span>{/if}
	</Badge>
</span>
