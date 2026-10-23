<script lang="ts">
	import Icon from './Icon.svelte';
	import { Button } from '$lib/components/ui';
	import tooltip from '$lib/actions/tooltip';
	import { PRESET_COLORS } from '$lib/types/segments';

	let { value, onchange }: { value: string; onchange: (color: string) => void } = $props();

	const current = $derived((value || '').toLowerCase());
</script>

<div class="flex flex-wrap items-center gap-1.5" role="group" aria-label="Colour">
	<label
		class="relative grid h-7 w-7 flex-shrink-0 cursor-pointer place-items-center rounded border border-line-strong bg-surface-2 text-fg-subtle"
		style={value ? `background-color: ${value};` : undefined}
		use:tooltip={'Pick a custom colour'}
	>
		<span class="sr-only">Pick a custom colour</span>
		<input
			type="color"
			class="sr-only"
			value={value || PRESET_COLORS[0].value}
			oninput={(event) => onchange(event.currentTarget.value)}
		/>
		{#if !value}<Icon name="plus" className="h-3.5 w-3.5" />{/if}
	</label>
	{#each PRESET_COLORS as option (option.value)}
		{@const active = current === option.value.toLowerCase()}
		<button
			type="button"
			class="h-7 w-7 flex-shrink-0 rounded border-2 transition-shadow {active ? 'border-fg shadow-raised' : 'border-transparent'}"
			style="background-color: {option.value};"
			aria-label={option.name}
			aria-pressed={active}
			use:tooltip={option.name}
			onclick={() => onchange(option.value)}
		></button>
	{/each}
	{#if value}
		<Button variant="ghost" size="sm" onclick={() => onchange('')}>Clear</Button>
	{/if}
</div>
