<script lang="ts">
	import type { Snippet } from 'svelte';
	import dismiss from '$lib/actions/dismiss';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import Icon from '$lib/components/Icon.svelte';
	import { Badge } from '$lib/components/ui';
	import { computeAnchoredMenuPosition } from '$lib/utils/menuPosition';

	let {
		activeCount,
		canClear,
		onClear,
		label = 'Advanced filters',
		class: className = '',
		children,
		footerAction
	}: {
		activeCount: number;
		canClear: boolean;
		onClear: () => void;
		label?: string;
		class?: string;
		children: Snippet<[() => void]>;
		footerAction?: Snippet<[() => void]>;
	} = $props();

	let open = $state(false);
	let rootEl: HTMLDivElement | undefined = $state();
	let buttonEl: HTMLButtonElement | undefined = $state();
	let panelEl: HTMLDivElement | undefined = $state();
	let position = $state({ top: 0, left: 0 });

	function updatePosition() {
		if (!buttonEl) return;
		const width = panelEl?.getBoundingClientRect().width ?? 352;
		position = computeAnchoredMenuPosition(buttonEl, { width });
	}

	function toggle() {
		open = !open;
		if (open) {
			updatePosition();
			requestAnimationFrame(updatePosition);
		}
	}

	function close() {
		open = false;
	}
</script>

<div class="relative {className}" bind:this={rootEl}>
	<button
		bind:this={buttonEl}
		type="button"
		class="input text-xs py-1.5 px-2.5 bg-surface-2/50 w-auto flex items-center gap-1.5 {activeCount > 0
			? 'text-signal'
			: 'text-fg-muted'}"
		aria-haspopup="dialog"
		aria-expanded={open}
		onclick={toggle}
	>
		<Icon name="filter" className="w-3.5 h-3.5" />
		Filters
		{#if activeCount > 0}
			<Badge variant="signal" size="sm">{activeCount}</Badge>
		{/if}
	</button>
	{#if open}
		<div
			use:portal
			use:overlayLayer
			bind:this={panelEl}
			use:dismiss={{ onDismiss: close, ignore: [rootEl] }}
			class="fixed z-overlay w-[22rem] max-w-[90vw] bg-surface-2 rounded-xl border border-line-strong shadow-floating p-3"
			style="top: {position.top}px; left: {position.left}px;"
			role="dialog"
			aria-label={label}
		>
			<div class="grid grid-cols-2 gap-2.5">
				{@render children(close)}
			</div>

			<div class="mt-3 pt-2.5 border-t border-line-strong/70 flex items-center justify-between">
				<span class="text-2xs text-fg-subtle">{activeCount} active</span>
				{@render footerAction?.(close)}
				<button
					type="button"
					class="text-xs text-fg-muted hover:text-fg transition-colors flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed"
					disabled={!canClear}
					onclick={onClear}
				>
					<Icon name="close" className="w-3 h-3" />
					Clear all
				</button>
			</div>
		</div>
	{/if}
</div>
