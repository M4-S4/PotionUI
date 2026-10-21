<script lang="ts">
	import { onMount, setContext } from 'svelte';
	import { INSIDE_TOOLTIP_CONTEXT_KEY } from './tooltipContext';
	import TooltipBubble from './TooltipBubble.svelte';
	import { isOverflowing } from '$lib/actions/tooltip';

	setContext(INSIDE_TOOLTIP_CONTEXT_KEY, true);

	export let text: string;
	export let position: 'top' | 'bottom' | 'left' | 'right' = 'top';
	export let delay: number = 200;
	export let kbd: string | undefined = undefined;
	export let kbdSize: 'sm' | 'md' = 'sm';
	export let wrapperClass: string = 'inline-flex items-center';
	export let truncateOnly: boolean = false;

	const id = `tooltip-${Math.random().toString(36).slice(2, 10)}`;
	let showTooltip = false;
	let timeoutId: ReturnType<typeof setTimeout> | null = null;
	let triggerElement: HTMLDivElement;
	let describedElement: Element | null = null;

	function reveal() {
		if (truncateOnly && !isOverflowing(triggerElement)) return;
		showTooltip = true;
		describedElement = triggerElement.firstElementChild;
		describedElement?.setAttribute('aria-describedby', id);
	}

	function clear() {
		if (timeoutId) {
			clearTimeout(timeoutId);
			timeoutId = null;
		}
	}

	function handleMouseEnter() {
		clear();
		timeoutId = setTimeout(reveal, delay);
	}

	function handleFocusIn() {
		clear();
		reveal();
	}

	function hide() {
		clear();
		showTooltip = false;
		describedElement?.removeAttribute('aria-describedby');
		describedElement = null;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape' && showTooltip) hide();
	}

	onMount(() => clear);
</script>

<svelte:window on:keydown={handleKeydown} />

<div class={wrapperClass}>
	<div
		bind:this={triggerElement}
		on:mouseenter={handleMouseEnter}
		on:mouseleave={hide}
		on:focusin={handleFocusIn}
		on:focusout={hide}
		class={wrapperClass}
	>
		<slot />
	</div>
</div>

{#if showTooltip && text}
	<TooltipBubble {text} {id} {position} {kbd} {kbdSize} anchor={triggerElement} />
{/if}
