<script lang="ts">
	import { getContext } from 'svelte';
	import Icon from '../Icon.svelte';
	import Tooltip from '../Tooltip.svelte';
	import IconButton from './IconButton.svelte';
	import { INSIDE_TOOLTIP_CONTEXT_KEY, tooltipDescription } from '../tooltipContext';

	type Size = 'xs' | 'sm' | 'md';
	type Variant = 'ghost' | 'secondary';

	let {
		icon,
		label,
		size = 'md',
		variant = 'ghost',
		active = false,
		loading = false,
		disabled = false,
		onclick,
		ariaExpanded,
		ariaPressed,
		tooltip,
		kbd,
		class: className = ''
	}: {
		icon: string;
		label: string;
		size?: Size;
		variant?: Variant;
		active?: boolean;
		loading?: boolean;
		disabled?: boolean;
		onclick?: (e: MouseEvent) => void;
		ariaExpanded?: boolean;
		ariaPressed?: boolean;
		tooltip?: string;
		kbd?: string | string[];
		class?: string;
	} = $props();

	let isDisabled = $derived(disabled || loading);

	// A wrapping Tooltip already shows `label`; keeping the native title too
	// renders two tooltips side by side.
	const insideTooltip = getContext<boolean | undefined>(INSIDE_TOOLTIP_CONTEXT_KEY) === true;
	const describedBy = tooltipDescription();
	let ownTooltip = $derived(tooltip !== undefined || kbd !== undefined);

	const sizeClasses: Record<Size, string> = {
		xs: 'min-w-6 min-h-6 p-1',
		sm: 'min-w-8 min-h-8 p-1.5',
		md: 'min-w-10 min-h-10 p-2'
	};

	const iconSizeClasses: Record<Size, string> = {
		xs: 'w-3.5 h-3.5',
		sm: 'w-4 h-4',
		md: 'w-5 h-5'
	};

	const variantClasses: Record<Variant, string> = {
		ghost: 'text-fg-muted hover:text-fg hover:bg-surface-3/50',
		secondary: 'bg-surface-3 text-fg hover:bg-line-hover'
	};

	let classes = $derived(
		`inline-flex items-center justify-center rounded transition-colors duration-100 touch-manipulation disabled:opacity-50 disabled:cursor-not-allowed ${sizeClasses[size]} ${active ? 'bg-signal/10 text-signal' : variantClasses[variant]} ${className}`
	);
</script>

{#if ownTooltip}
	<Tooltip text={tooltip ?? label} {kbd}>
		<IconButton
			{icon}
			{label}
			{size}
			{variant}
			{active}
			{loading}
			{disabled}
			{onclick}
			{ariaExpanded}
			{ariaPressed}
			class={className}
		/>
	</Tooltip>
{:else}
	<button
		type="button"
		class={classes}
		aria-label={label}
		aria-expanded={ariaExpanded}
		aria-pressed={ariaPressed}
		aria-busy={loading}
		aria-describedby={$describedBy}
		title={insideTooltip ? undefined : label}
		disabled={isDisabled}
		{onclick}
	>
		{#if loading}
			<span
				class="{iconSizeClasses[
					size
				]} rounded-full border-2 border-line-strong border-t-current animate-spin"
			></span>
		{:else}
			<Icon name={icon} className={iconSizeClasses[size]} />
		{/if}
	</button>
{/if}
