<script module lang="ts">
	export function stop(fn: () => void) {
		return (event: Event) => {
			event.stopPropagation();
			fn();
		};
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';

	let {
		kind,
		label,
		selected = false,
		onOpen,
		onToggleSelect,
		children
	}: {
		kind: 'prompt' | 'category' | 'segment' | 'template';
		label: string;
		selected?: boolean;
		onOpen: () => void;
		onToggleSelect: () => void;
		children: Snippet;
	} = $props();

	const hook = $derived({ [`data-${kind}-card`]: '' });

	function handleKeydown(event: KeyboardEvent) {
		if (event.target !== event.currentTarget) return;
		if (event.key === 'Enter') {
			event.preventDefault();
			onOpen();
		} else if (event.key === ' ') {
			event.preventDefault();
			onToggleSelect();
		}
	}

	function handleToggle(event: Event) {
		event.stopPropagation();
		onToggleSelect();
	}
</script>

<div
	class="group relative flex min-w-0 gap-3 rounded-lg border p-3 shadow-raised transition-colors {selected
		? 'border-signal bg-signal/[0.06] shadow-[0_0_0_1px_rgb(var(--signal))]'
		: 'border-line-strong bg-surface-1 hover:border-line-hover'}"
	role="button"
	tabindex="0"
	{...hook}
	aria-label={label}
	onclick={onOpen}
	onkeydown={handleKeydown}
>
	<button
		type="button"
		class="mt-1 flex h-4 w-4 flex-shrink-0 items-center justify-center rounded border transition-opacity {selected
			? 'border-accent bg-accent text-accent-contrast opacity-100'
			: 'border-line-hover bg-surface-2 text-transparent opacity-0 group-hover:opacity-100 group-focus-within:opacity-100'}"
		onclick={handleToggle}
		aria-label={selected ? `Deselect ${kind}` : `Select ${kind}`}
		aria-pressed={selected}
	>
		{#if selected}<Icon name="check" className="h-3 w-3" strokeWidth={3} />{/if}
	</button>

	{@render children()}
</div>
