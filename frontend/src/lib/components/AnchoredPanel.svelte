<script lang="ts">
	import type { Snippet } from 'svelte';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import dismiss, { type DismissReason } from '$lib/actions/dismiss';
	import { computeFlippedMenuPosition, type FlippedMenuPosition } from '$lib/utils/menuPosition';
	import { CHIP_POPOVER_VIEWPORT_GUTTER, CHIP_POPOVER_WIDTH, isChipPopoverSheet } from '$lib/utils/chipPopover';

	let {
		anchor,
		label,
		width = CHIP_POPOVER_WIDTH,
		heightEstimate = 360,
		escape = false,
		testId,
		class: className = '',
		onDismiss,
		children
	}: {
		anchor: HTMLElement | undefined;
		label: string;
		width?: number;
		heightEstimate?: number;
		escape?: boolean;
		testId?: string;
		class?: string;
		onDismiss: (reason: DismissReason, event: Event) => void;
		children: Snippet;
	} = $props();

	const GUTTER = CHIP_POPOVER_VIEWPORT_GUTTER;

	let panel = $state<HTMLDivElement>();
	let sheet = $state(false);
	let panelWidth = $state(CHIP_POPOVER_WIDTH);
	let pos = $state<FlippedMenuPosition>({ left: 0, top: 0, maxHeight: 360 });

	const style = $derived(
		sheet
			? `position: fixed; left: ${GUTTER}px; right: ${GUTTER}px; bottom: ${GUTTER}px; top: auto; max-height: min(80dvh, 32rem);`
			: `position: fixed; left: ${pos.left}px; width: ${panelWidth}px; max-width: calc(100vw - ${GUTTER * 2}px); max-height: ${pos.maxHeight}px; ${
					pos.top !== undefined ? `top: ${pos.top}px; bottom: auto;` : `bottom: ${pos.bottom}px; top: auto;`
				}`
	);

	function update() {
		sheet = isChipPopoverSheet(window.innerWidth);
		if (!anchor || sheet) return;
		const next = Math.min(Math.max(anchor.offsetWidth, width), window.innerWidth - GUTTER * 2);
		panelWidth = next;
		pos = computeFlippedMenuPosition(anchor, {
			width: next,
			heightEstimate: panel?.scrollHeight || heightEstimate,
			gap: 4
		});
	}

	$effect(() => {
		update();
		const raf = requestAnimationFrame(update);
		window.addEventListener('resize', update);
		window.addEventListener('scroll', update, true);
		return () => {
			cancelAnimationFrame(raf);
			window.removeEventListener('resize', update);
			window.removeEventListener('scroll', update, true);
		};
	});
</script>

<div
	bind:this={panel}
	use:dismiss={{ onDismiss, ignore: [anchor], escape }}
	use:portal
	use:overlayLayer
	data-chip-layer
	data-testid={testId}
	class="z-overlay flex flex-col overflow-hidden rounded-xl border border-line-strong bg-surface-1 shadow-floating {className}"
	style={style}
	role="dialog"
	aria-label={label}
>
	{@render children()}
</div>
