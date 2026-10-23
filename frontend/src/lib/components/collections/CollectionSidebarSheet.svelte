<script lang="ts">
	import { setContext, type Snippet } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import focusTrap from '$lib/actions/focusTrap';
	import { IconButton } from '$lib/components/ui';
	import { COLLECTION_SHEET, type CollectionSheet } from './sidebar';

	let {
		label,
		collapsed = false,
		onExpand,
		children
	}: {
		label: string;
		collapsed?: boolean;
		onExpand?: () => void;
		children: Snippet;
	} = $props();

	const DESKTOP_QUERY = '(min-width: 768px)';
	const reducedMotion =
		typeof window !== 'undefined' &&
		typeof window.matchMedia === 'function' &&
		window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	const duration = reducedMotion ? 0 : 180;

	let open = $state(false);

	function desktop(): boolean {
		return typeof window.matchMedia === 'function' && window.matchMedia(DESKTOP_QUERY).matches;
	}

	function handleOpen() {
		if (onExpand && desktop()) {
			onExpand();
			return;
		}
		open = true;
	}

	function close() {
		open = false;
	}

	setContext<CollectionSheet>(COLLECTION_SHEET, { close });

	$effect(() => {
		if (!open) return;
		const previousOverflow = document.body.style.overflow;
		document.body.style.overflow = 'hidden';
		const query = typeof window.matchMedia === 'function' ? window.matchMedia(DESKTOP_QUERY) : null;
		const onChange = (event: MediaQueryListEvent) => {
			if (event.matches) close();
		};
		query?.addEventListener?.('change', onChange);
		return () => {
			document.body.style.overflow = previousOverflow;
			query?.removeEventListener?.('change', onChange);
		};
	});

	function handleKeydown(event: KeyboardEvent) {
		if (event.key !== 'Escape' || event.defaultPrevented) return;
		event.preventDefault();
		event.stopPropagation();
		close();
	}

	function handleBackdropClick(event: MouseEvent) {
		if (event.target === event.currentTarget) close();
	}
</script>

<IconButton
	icon="sidebar-expand"
	label="Show collections"
	size="sm"
	ariaExpanded={open}
	class="flex-shrink-0 {collapsed ? '' : 'md:hidden'}"
	onclick={handleOpen}
/>

{#if open}
	<div
		use:portal
		use:overlayLayer
		class="fixed inset-0 z-overlay bg-black/50"
		role="presentation"
		onclick={handleBackdropClick}
		transition:fade={{ duration }}
	>
		<div
			class="flex h-[100dvh] w-[min(340px,calc(100vw-24px))] flex-col bg-surface-1 pb-[env(safe-area-inset-bottom)] pt-[env(safe-area-inset-top)] shadow-overlay focus:outline-none"
			role="dialog"
			aria-modal="true"
			aria-label="{label} collections"
			tabindex="-1"
			use:focusTrap
			onkeydown={handleKeydown}
			transition:fly={{ x: -48, duration }}
		>
			{@render children()}
		</div>
	</div>
{/if}
