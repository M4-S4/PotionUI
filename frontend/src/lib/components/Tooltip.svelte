<script lang="ts">
	import { onMount, setContext } from 'svelte';
	import { writable } from 'svelte/store';
	import Kbd from '$lib/components/ui/Kbd.svelte';
	import {
		INSIDE_TOOLTIP_CONTEXT_KEY,
		TOOLTIP_DESCRIPTION_CONTEXT_KEY,
		lastInputWasKeyboard,
		nextTooltipId,
		trackInputModality
	} from './tooltipContext';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';

	// Lets descendants drop their own native `title` fallback rather than
	// rendering a second, unstyled tooltip beside this one.
	setContext(INSIDE_TOOLTIP_CONTEXT_KEY, true);

	const tooltipId = nextTooltipId();
	const describedBy = writable<string | undefined>(undefined);
	setContext(TOOLTIP_DESCRIPTION_CONTEXT_KEY, describedBy);

	export let text: string;
	export let position: 'top' | 'bottom' | 'left' | 'right' = 'top';
	export let delay: number = 200; // ms delay before showing
	// Optional formatted keyboard shortcut (e.g. "Ctrl+K"), rendered as a small
	// mono chip next to the tooltip text. Omit when the action has no binding.
	export let kbd: string | string[] | undefined = undefined;
	export let kbdSize: 'sm' | 'md' = 'sm';
	// Classes for the trigger wrappers. Defaults to inline layout (forms); the
	// sidebar passes a flex/centering class so vertically-stacked icon buttons
	// keep their layout instead of collapsing to inline boxes.
	export let wrapperClass: string = 'inline-flex items-center';

	let hoverVisible = false;
	let focusVisible = false;
	let timeoutId: ReturnType<typeof setTimeout> | null = null;
	let tooltipElement: HTMLDivElement;
	let triggerElement: HTMLDivElement;

	// Position for fixed tooltip
	let tooltipStyle = '';

	$: showTooltip = hoverVisible || focusVisible;
	$: describedBy.set(showTooltip && text ? tooltipId : undefined);
	$: kbdLabel = Array.isArray(kbd) ? kbd.join('+') : kbd;

	function calculatePosition() {
		if (!triggerElement || !tooltipElement) return;

		const triggerRect = triggerElement.getBoundingClientRect();
		const tooltipRect = tooltipElement.getBoundingClientRect();

		let top = 0;
		let left = 0;

		switch (position) {
			case 'top':
				top = triggerRect.top - tooltipRect.height - 8;
				left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2);
				break;
			case 'bottom':
				top = triggerRect.bottom + 8;
				left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2);
				break;
			case 'left':
				top = triggerRect.top + (triggerRect.height / 2) - (tooltipRect.height / 2);
				left = triggerRect.left - tooltipRect.width - 8;
				break;
			case 'right':
				top = triggerRect.top + (triggerRect.height / 2) - (tooltipRect.height / 2);
				left = triggerRect.right + 8;
				break;
		}

		// Keep tooltip within viewport
		const padding = 8;
		if (left < padding) left = padding;
		if (left + tooltipRect.width > window.innerWidth - padding) {
			left = window.innerWidth - tooltipRect.width - padding;
		}
		if (top < padding) top = padding;
		if (top + tooltipRect.height > window.innerHeight - padding) {
			top = window.innerHeight - tooltipRect.height - padding;
		}

		tooltipStyle = `top: ${top}px; left: ${left}px;`;
	}

	function handleMouseEnter() {
		timeoutId = setTimeout(() => {
			hoverVisible = true;
			// Wait for next tick to calculate position after tooltip is rendered
			requestAnimationFrame(() => {
				calculatePosition();
			});
		}, delay);
	}

	function handleMouseLeave() {
		if (timeoutId) {
			clearTimeout(timeoutId);
			timeoutId = null;
		}
		hoverVisible = false;
	}

	function handleFocusIn() {
		if (!lastInputWasKeyboard()) return;
		focusVisible = true;
		requestAnimationFrame(() => {
			calculatePosition();
		});
	}

	function handleFocusOut(event: FocusEvent) {
		if (triggerElement?.contains(event.relatedTarget as Node | null)) return;
		focusVisible = false;
	}

	onMount(() => {
		trackInputModality();
		const trigger = triggerElement;
		trigger.addEventListener('focusin', handleFocusIn);
		trigger.addEventListener('focusout', handleFocusOut);
		return () => {
			if (timeoutId) {
				clearTimeout(timeoutId);
			}
			trigger.removeEventListener('focusin', handleFocusIn);
			trigger.removeEventListener('focusout', handleFocusOut);
		};
	});
</script>

<div class={wrapperClass}>
	<!-- Trigger element -->
	<div
		bind:this={triggerElement}
		on:mouseenter={handleMouseEnter}
		on:mouseleave={handleMouseLeave}
		class={wrapperClass}
	>
		<slot />
	</div>
</div>

<!-- Tooltip rendered fixed to escape overflow containers; portaled to <body> so a
     `backdrop-filter` ancestor (e.g. the docked generation panel) can't trap it
     in its own stacking context, below overlays like the floating workbench. -->
{#if showTooltip && text}
	<div
		bind:this={tooltipElement}
		use:portal
		use:overlayLayer={'tooltip'}
		id={tooltipId}
		role="tooltip"
		style="min-width: max-content; max-width: 300px; {tooltipStyle}"
		class="fixed z-tooltip px-2 py-1 text-xs font-medium text-fg bg-surface-3 rounded-md shadow-lg pointer-events-none animate-in fade-in duration-150 break-words flex items-center gap-1.5"
	>
		<span>{text}</span>
		{#if kbd}
			<Kbd keys={kbd} size={kbdSize} />
			<span class="sr-only">{kbdLabel}</span>
		{/if}
		<!-- Arrow -->
		<div
			class="absolute w-2 h-2 bg-surface-3 rotate-45
				{position === 'top'
				? 'bottom-[-4px] left-1/2 -translate-x-1/2'
				: position === 'bottom'
					? 'top-[-4px] left-1/2 -translate-x-1/2'
					: position === 'left'
						? 'right-[-4px] top-1/2 -translate-y-1/2'
						: 'left-[-4px] top-1/2 -translate-y-1/2'}"
		/>
	</div>
{/if}

<style>
	@keyframes fade-in {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	.animate-in {
		animation: fade-in 150ms ease-out;
	}
</style>