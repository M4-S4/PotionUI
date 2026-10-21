<script lang="ts">
	import type { Snippet } from 'svelte';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import focusTrap from '$lib/actions/focusTrap';
	import dismiss, { type DismissReason } from '$lib/actions/dismiss';
	import { handleDrawerKeydown } from './drawerBehavior';
	import { MENU_EDGE_GUTTER, trackMenuPosition, type FlippedMenuPosition } from '$lib/utils/menuPosition';

	const GAP = 6;

	let {
		label,
		anchor,
		width = 440,
		fill = false,
		fullscreen = false,
		onClose,
		onEscape,
		searchEl = $bindable(undefined),
		root = $bindable(undefined),
		children
	}: {
		label: string;
		anchor: HTMLElement | undefined;
		width?: number;
		fill?: boolean;
		fullscreen?: boolean;
		onClose: () => void;
		onEscape?: () => boolean;
		searchEl?: HTMLInputElement;
		root?: HTMLElement;
		children: Snippet;
	} = $props();

	let position = $state<(FlippedMenuPosition & { width: number }) | null>(null);

	function panelWidth(): number {
		return Math.min(width, window.innerWidth - 2 * MENU_EDGE_GUTTER);
	}

	$effect(() => {
		if (fullscreen || !anchor || !root) return;
		const panel = root;
		return trackMenuPosition(
			anchor,
			() => ({
				width: panelWidth(),
				heightEstimate: panel.offsetHeight,
				gap: GAP,
				align: 'left',
				preferred: 'down'
			}),
			(next) => {
				position = { ...next, width: panelWidth() };
			}
		);
	});

	let placement = $derived.by(() => {
		if (fullscreen) return '';
		if (!position) return 'visibility: hidden; top: 0; left: 0;';
		const vertical = position.top !== undefined ? `top: ${position.top}px;` : `bottom: ${position.bottom}px;`;
		const height = fill ? ` height: ${position.maxHeight}px;` : '';
		return `${vertical} left: ${position.left}px; width: ${position.width}px; max-height: ${position.maxHeight}px;${height}`;
	});

	function dismissOutside(reason: DismissReason) {
		if (reason === 'outside') onClose();
	}

	function handleKeydown(event: KeyboardEvent) {
		handleDrawerKeydown(event, { root, searchEl, keepOpen: false, onEscape, onClose });
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div
	class="session-dropdown flex flex-col overflow-hidden bg-surface-1 {fullscreen
		? 'is-fullscreen'
		: 'rounded-xl border border-line-strong shadow-overlay'}"
	role="dialog"
	tabindex="-1"
	aria-label={label}
	aria-modal="true"
	style={placement}
	bind:this={root}
	use:portal
	use:overlayLayer
	use:focusTrap
	use:dismiss={{ onDismiss: dismissOutside, ignore: [anchor], skipModals: true, escape: false }}
>
	{@render children()}
</div>

<style>
	.session-dropdown {
		position: fixed;
		animation: dropdown-in 140ms cubic-bezier(0.25, 1, 0.5, 1);
	}

	.session-dropdown.is-fullscreen {
		inset: 0;
		animation: none;
	}

	@keyframes dropdown-in {
		from {
			opacity: 0;
			transform: translateY(-4px);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.session-dropdown {
			animation: none;
		}
	}
</style>
