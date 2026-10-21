<script lang="ts">
	import { tick } from 'svelte';
	import Icon from './Icon.svelte';
	import VariablePicker from './VariablePicker.svelte';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import dismiss from '$lib/actions/dismiss';
	import { computeFlippedMenuPosition, type FlippedMenuPosition } from '$lib/utils/menuPosition';
	import { CHIP_POPOVER_VIEWPORT_GUTTER, CHIP_POPOVER_WIDTH, isChipPopoverSheet } from '$lib/utils/chipPopover';
	import type { VariablesMap } from '$lib/utils/variableDefs';
	import type { VariablePickerType } from '$lib/utils/variablePicker';

	let {
		variables,
		value,
		missing = false,
		allowedTypes = undefined,
		disabledReasons = undefined,
		label = 'Variable',
		disabled = false,
		class: className = '',
		onselect,
		oncreate
	}: {
		variables: VariablesMap;
		value: string;
		missing?: boolean;
		allowedTypes?: readonly VariablePickerType[];
		disabledReasons?: Record<string, string>;
		label?: string;
		disabled?: boolean;
		class?: string;
		onselect?: (name: string) => void;
		oncreate?: (name: string) => void;
	} = $props();

	const HEIGHT_ESTIMATE = 360;

	let open = $state(false);
	let sheet = $state(false);
	let triggerRef = $state<HTMLButtonElement>();
	let panelRef = $state<HTMLDivElement>();
	let pos = $state<FlippedMenuPosition>({ left: 0, top: 0, maxHeight: HEIGHT_ESTIMATE });

	const panelWidth = CHIP_POPOVER_WIDTH;

	const panelStyle = $derived(
		sheet
			? `position: fixed; left: ${CHIP_POPOVER_VIEWPORT_GUTTER}px; right: ${CHIP_POPOVER_VIEWPORT_GUTTER}px; bottom: ${CHIP_POPOVER_VIEWPORT_GUTTER}px; top: auto; max-height: min(70dvh, 28rem);`
			: `position: fixed; left: ${pos.left}px; width: ${panelWidth}px; max-width: calc(100vw - ${CHIP_POPOVER_VIEWPORT_GUTTER * 2}px); max-height: ${pos.maxHeight}px; ${
					pos.top !== undefined ? `top: ${pos.top}px; bottom: auto;` : `bottom: ${pos.bottom}px; top: auto;`
				}`
	);

	function updatePosition() {
		sheet = isChipPopoverSheet(window.innerWidth);
		if (!triggerRef || sheet) return;
		pos = computeFlippedMenuPosition(triggerRef, {
			width: panelWidth,
			heightEstimate: panelRef?.scrollHeight || HEIGHT_ESTIMATE,
			gap: 4
		});
	}

	$effect(() => {
		if (!open) return;
		updatePosition();
		const raf = requestAnimationFrame(updatePosition);
		window.addEventListener('resize', updatePosition);
		window.addEventListener('scroll', updatePosition, true);
		return () => {
			cancelAnimationFrame(raf);
			window.removeEventListener('resize', updatePosition);
			window.removeEventListener('scroll', updatePosition, true);
		};
	});

	function close(refocus: boolean) {
		open = false;
		if (refocus) void tick().then(() => triggerRef?.focus());
	}

	function toggle() {
		if (disabled) return;
		open = !open;
	}

	function handleTriggerKeydown(e: KeyboardEvent) {
		if (disabled || open) return;
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			open = true;
		}
	}
</script>

<button
	bind:this={triggerRef}
	type="button"
	{disabled}
	class="input flex h-8 min-w-0 items-center gap-1.5 py-1 text-left text-sm max-md:h-11 {className}"
	aria-label={label}
	aria-haspopup="listbox"
	aria-expanded={open}
	data-testid="variable-select"
	onclick={toggle}
	onkeydown={handleTriggerKeydown}
>
	<span class="min-w-0 flex-1 truncate font-mono {missing ? 'text-warning' : 'text-fg'}">${value}</span>
	{#if missing}<span class="flex-shrink-0 text-xs text-warning">missing</span>{/if}
	<Icon name="chevron-down" className="h-3 w-3 flex-shrink-0 text-fg-subtle" />
</button>

{#if open}
	<div
		bind:this={panelRef}
		use:dismiss={{ onDismiss: () => (open = false), ignore: [triggerRef], escape: false }}
		use:portal
		use:overlayLayer
		data-chip-layer
		class="z-overlay flex flex-col overflow-hidden rounded-xl border border-line-strong bg-surface-1 shadow-floating"
		style={panelStyle}
		role="dialog"
		aria-label={`Choose ${label.toLowerCase()}`}
	>
		<VariablePicker
			{variables}
			selected={missing ? null : value}
			{allowedTypes}
			{disabledReasons}
			{label}
			onselect={(name) => {
				onselect?.(name);
				close(true);
			}}
			oncreate={oncreate
				? (name) => {
						oncreate?.(name);
						close(true);
					}
				: undefined}
			onclose={(reason) => close(reason === 'escape')}
		/>
	</div>
{/if}
