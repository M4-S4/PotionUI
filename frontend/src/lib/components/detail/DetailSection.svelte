<script lang="ts">
	import { IconButton } from '$lib/components/ui';
	import type { Snippet } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { toggleOpen, sectionBoxClass, sectionBodyClass } from './detailSection';

	let {
		label,
		icon,
		headerExtra,
		footer,
		collapsible = false,
		open = $bindable(true),
		padded = true,
		children
	}: {
		label: string;
		icon?: string;
		headerExtra?: Snippet;
		footer?: Snippet;
		collapsible?: boolean;
		open?: boolean;
		padded?: boolean;
		children?: Snippet;
	} = $props();

	const showBody = $derived(!collapsible || open);

	function handleToggle() {
		open = toggleOpen(open);
	}
</script>

<section class={sectionBoxClass(padded)}>
	<div
		class="px-4 sm:px-5 py-3 flex items-center justify-between gap-2 {showBody ? 'border-b border-line' : ''}"
	>
		<h3 class="flex min-w-0 items-center gap-2 font-mono text-xs uppercase tracking-[0.07em] text-fg-muted">
			{#if icon}<Icon name={icon} className="w-3.5 h-3.5 flex-shrink-0" />{/if}
			<span class="truncate">{label}</span>
		</h3>
		<div class="flex items-center gap-2 flex-shrink-0">
			{@render headerExtra?.()}
			{#if collapsible}
				<IconButton icon="chevron-down" label={open ? 'Collapse' : 'Expand'} size="xs" onclick={handleToggle} ariaExpanded={open} />
			{/if}
		</div>
	</div>
	{#if showBody}
		<div class={sectionBodyClass(padded)}>
			{@render children?.()}
		</div>
	{/if}
	{#if footer}
		<div class="px-4 sm:px-5 py-3 border-t border-line flex items-center justify-end gap-2 flex-wrap">
			{@render footer()}
		</div>
	{/if}
</section>
