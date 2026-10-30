<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';

	let {
		icon,
		label,
		description,
		state,
		danger = false,
		disabled = false,
		onclick
	}: {
		icon: string;
		label: string;
		description?: string;
		state?: string;
		danger?: boolean;
		disabled?: boolean;
		onclick: () => void;
	} = $props();
</script>

<button
	type="button"
	role="menuitem"
	{disabled}
	class="flex w-full items-center gap-3 rounded px-3 py-2 text-left text-md hover:bg-surface-3 focus-visible:bg-surface-3 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent {danger
		? 'text-danger'
		: 'text-fg'}"
	{onclick}
>
	<Icon name={icon} className="h-4 w-4 flex-shrink-0 {danger ? 'text-danger' : 'text-fg-muted'}" />
	<span class="flex min-w-0 flex-1 flex-col">
		<span>{label}</span>
		{#if description}
			<span class="text-sm text-fg-subtle">{description}</span>
		{/if}
	</span>
	{#if state}
		<span class="flex-shrink-0 font-mono text-sm tabular-nums text-signal">{state}</span>
	{/if}
</button>
