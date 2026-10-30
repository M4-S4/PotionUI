<script lang="ts" module>
	export type FactPart = string | { mono: string };

	export interface StepFact {
		label: string;
		body: FactPart[];
	}
</script>

<script lang="ts">
	import Card from '$lib/components/ui/Card.svelte';

	let { facts, class: className = '' }: { facts: StepFact[]; class?: string } = $props();

	const pathSegments = (value: string) => value.split(/(?<=[/\\])/);
</script>

<Card padding="none" class="grid grid-cols-1 overflow-hidden sm:grid-cols-3 {className}">
	{#each facts as fact, i (fact.label)}
		<div
			class="flex min-w-0 flex-col gap-1 px-4 py-3.5 {i > 0
				? 'border-t border-line sm:border-l sm:border-t-0'
				: ''}"
		>
			<span class="font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">{fact.label}</span>
			<p class="text-base text-fg-muted">
				{#each fact.body as part, j (j)}
					{#if typeof part === 'string'}{part}{:else}<span class="break-words font-mono tabular-nums text-fg"
							>{#each pathSegments(part.mono) as segment, k (k)}{segment}{#if k < pathSegments(part.mono).length - 1}<wbr
								/>{/if}{/each}</span
						>{/if}
				{/each}
			</p>
		</div>
	{/each}
</Card>
