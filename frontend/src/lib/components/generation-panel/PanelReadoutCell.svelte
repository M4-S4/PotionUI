<script lang="ts">
	export let label: string;
	export let clickable = false;
	export let disabled = false;
	export let onclick: (() => void) | undefined = undefined;
	// Overrides the button's computed accessible name — needed when the
	// visible value text is state-dependent (a relative timestamp, a queue
	// count) and the trigger still needs one stable name for assistive tech
	// and tests to grab onto.
	export let ariaLabel: string | undefined = undefined;
	// A callback ref (not `bind:this`, which doesn't cross a component
	// boundary) so a clickable cell's caller can measure its rect — the
	// queue popover positions itself against this element, and it must stay
	// the mock's literal `.readout` (no extra wrapper div for the ref to
	// live on).
	export let onElement: ((el: HTMLElement) => void) | undefined = undefined;
	function bindElement(node: HTMLElement) {
		onElement?.(node);
	}
</script>

{#if clickable}
	<button
		type="button"
		class="readout"
		use:bindElement
		{disabled}
		aria-label={ariaLabel}
		on:click={() => {
			if (!disabled) onclick?.();
		}}
	>
		<span class="cell-label">{label}</span>
		<span class="cell-value"><slot /></span>
	</button>
{:else}
	<div class="readout" use:bindElement>
		<span class="cell-label">{label}</span>
		<span class="cell-value"><slot /></span>
	</div>
{/if}
