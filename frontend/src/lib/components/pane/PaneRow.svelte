<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { Badge } from '$lib/components/ui';
	import { placeholderTint } from '$lib/utils/placeholderTint';
	import tooltip from '$lib/actions/tooltip';

	type Size = 'md' | 'sm' | 'nav';
	type Role = 'option' | 'treeitem';
	type ThumbSize = 'sm' | 'md' | 'lg';

	let {
		onclick,
		selected = false,
		disabled = false,
		inactive = false,
		inactiveBadge,
		loading = false,
		size = 'md',
		role = 'option',
		depth = 0,
		expandable = false,
		expanded = false,
		onToggle,
		checkable = false,
		checked = false,
		onCheck,
		checkboxSpacer = false,
		icon,
		dot,
		thumbnail,
		thumbSize,
		thumbFallback,
		title,
		subtitle,
		subtitleMono = false,
		count,
		revealActions = false,
		class: className = '',
		leading,
		children,
		badges,
		meta,
		trailing,
		actions,
		preview
	}: {
		onclick?: () => void;
		selected?: boolean;
		disabled?: boolean;
		inactive?: boolean;
		inactiveBadge?: string;
		loading?: boolean;
		size?: Size;
		role?: Role;
		depth?: number;
		expandable?: boolean;
		expanded?: boolean;
		onToggle?: () => void;
		checkable?: boolean;
		checked?: boolean;
		onCheck?: (checked: boolean) => void;
		checkboxSpacer?: boolean;
		icon?: string;
		dot?: string;
		thumbnail?: string;
		/** No default: it distinguishes "unset" (no placeholder chip) from
		 *  an explicit size. Sizing falls back to 'md' at render time. */
		thumbSize?: ThumbSize;
		/** Text to derive an initial-letter fallback tile from when there's no
		 *  thumbnail. Unset keeps the generic icon fallback. */
		thumbFallback?: string;
		title?: string;
		subtitle?: string;
		subtitleMono?: boolean;
		count?: number | string;
		revealActions?: boolean;
		class?: string;
		leading?: Snippet;
		children?: Snippet;
		badges?: Snippet;
		meta?: Snippet;
		trailing?: Snippet;
		actions?: Snippet;
		preview?: Snippet;
	} = $props();

	const thumbSizeClasses: Record<ThumbSize, string> = {
		sm: 'w-6 h-6',
		md: 'w-10 h-10',
		lg: 'w-16 h-16'
	};

	const thumbIconSizeClasses: Record<ThumbSize, string> = {
		sm: 'w-3 h-3',
		md: 'w-4 h-4',
		lg: 'w-5 h-5'
	};

	// icon / dot / thumbnail: first one the consumer actually set wins; the
	// `leading` snippet always overrides all of them.
	let leadingKind = $derived(
		icon !== undefined
			? 'icon'
			: dot !== undefined
				? 'dot'
				: thumbnail !== undefined || thumbSize !== undefined
					? 'thumbnail'
					: 'none'
	);

	const NAV_INDENT = 21;
	const canHover =
		typeof window !== 'undefined' &&
		typeof window.matchMedia === 'function' &&
		window.matchMedia('(hover: hover)').matches;

	let nav = $derived(size === 'nav');
	let navTree = $derived(nav && role === 'treeitem');
	let showChevronSlot = $derived(expandable || loading || (!nav && depth > 0) || navTree);
	let indentPx = $derived(size === 'sm' ? depth * 12 + 2 : depth * 12);
	let titleTooltip = $derived(
		nav && title && canHover
			? { text: title, truncateOnly: true, position: 'right' as const, wrap: true, focusSelf: true, delay: 300 }
			: null
	);

	function sizeClasses(): string {
		if (size === 'md') return 'px-4 py-2.5 text-sm border-b border-line/50 last:border-b-0';
		if (size === 'sm') return 'h-7 px-2 rounded text-xs';
		return `max-md:min-h-11 [@media(hover:none)]:min-h-11 rounded text-sm ${navTree ? 'min-h-9 gap-px pl-px' : 'min-h-[38px] px-2.5'}`;
	}

	function stateClasses(): string {
		if (size === 'md') {
			return selected ? 'border-l-2 border-l-signal' : 'border-l-2 border-l-transparent hover:bg-surface-2/50';
		}
		if (size === 'sm') return selected ? 'bg-signal/10 text-signal' : 'hover:bg-surface-3/40';
		if (selected) {
			return navTree
				? 'bg-signal/10 text-signal before:absolute before:left-0 before:top-[9px] before:bottom-[9px] before:w-0.5 before:rounded-full before:bg-signal'
				: 'bg-signal/10 text-signal';
		}
		return navTree
			? 'text-fg hover:bg-surface-2 focus-within:bg-surface-2'
			: 'text-fg-muted hover:bg-surface-2 hover:text-fg';
	}

	let countClass = $derived(selected && !navTree ? 'text-signal' : 'text-fg-muted');

	function rowStyle(): string | undefined {
		if (size === 'md' && selected) {
			return 'background: linear-gradient(180deg, rgb(var(--signal) / 0.12), rgb(var(--signal) / 0.04));';
		}
		if (navTree && depth > 0) return `margin-left: ${depth * NAV_INDENT}px;`;
		return undefined;
	}

	function handleToggleClick(e: MouseEvent) {
		e.stopPropagation();
		onToggle?.();
	}

	function handleCheckClick(e: MouseEvent) {
		e.stopPropagation();
	}

	function handleCheckChange(e: Event) {
		onCheck?.((e.target as HTMLInputElement).checked);
	}

	function handleClick() {
		if (disabled) return;
		onclick?.();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (disabled) return;
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			onclick?.();
		} else if (e.key === 'ArrowRight' && expandable && !expanded) {
			e.preventDefault();
			onToggle?.();
		} else if (e.key === 'ArrowLeft' && expandable && expanded) {
			e.preventDefault();
			onToggle?.();
		}
	}
</script>

<!-- role is always 'option' or 'treeitem' (both interactive) but comes from a
     dynamic prop, so the a11y linter can't verify that statically. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	data-cover-host
	class="pane-row group relative flex items-center {sizeClasses()} {disabled
		? 'opacity-50 cursor-not-allowed'
		: 'cursor-pointer'} {stateClasses()} {className}"
	style={rowStyle()}
	use:tooltip={titleTooltip}
	data-pane-row
	data-disabled={disabled ? '' : undefined}
	{role}
	aria-selected={selected}
	aria-expanded={expandable ? expanded : undefined}
	aria-disabled={disabled}
	tabindex={disabled ? -1 : 0}
	onclick={handleClick}
	onkeydown={handleKeydown}
>
	{#if navTree}
		{#each { length: depth } as _, level}
			<span
				class="pointer-events-none absolute -top-px -bottom-px w-px bg-line-strong/65"
				style="left: -{level * NAV_INDENT + 4}px"
				aria-hidden="true"
			></span>
		{/each}
	{:else if depth > 0}
		<span class="flex-shrink-0" style="width: {indentPx}px" aria-hidden="true"></span>
	{/if}

	{#if showChevronSlot}
		{#if loading}
			<span class="flex-shrink-0 flex items-center justify-center w-5 h-5">
				<span
					class="w-3 h-3 rounded-full border border-line-strong border-t-transparent animate-spin"
				></span>
			</span>
		{:else if expandable}
			<button
				type="button"
				class="flex-shrink-0 flex items-center justify-center rounded hover:bg-line-hover/50 transition-colors {nav
					? 'w-[19px] h-[30px] text-fg-muted'
					: 'w-5 h-5'}"
				aria-label={nav ? `${expanded ? 'Collapse' : 'Expand'} ${title ?? ''}`.trim() : expanded ? 'Collapse' : 'Expand'}
				onclick={handleToggleClick}
				tabindex="-1"
			>
				<svg
					class="transition-transform {expanded ? 'rotate-90' : ''} {size === 'md' ? 'w-3.5 h-3.5' : 'w-3 h-3'}"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
				</svg>
			</button>
		{:else}
			<span class="flex-shrink-0 {nav ? 'w-[19px] h-[30px]' : 'w-5 h-5'}" aria-hidden="true"></span>
		{/if}
	{/if}

	{#if checkable}
		<span class="relative inline-flex flex-shrink-0 {nav ? 'w-[15px] h-[15px] ml-[3px] mr-[7px]' : 'w-3.5 h-3.5 mr-1.5'}">
			<input
				type="checkbox"
				class="peer absolute inset-0 opacity-0 cursor-pointer"
				aria-label={nav && title ? `Select ${title}` : undefined}
				{checked}
				onclick={handleCheckClick}
				onchange={handleCheckChange}
			/>
			<span
				class="pointer-events-none absolute inset-0 rounded border border-line-hover bg-surface-2 transition-colors peer-checked:border-accent peer-checked:bg-accent peer-focus-visible:ring-2 peer-focus-visible:ring-signal peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-canvas"
			></span>
			<svg
				class="pointer-events-none absolute inset-0 h-full w-full text-accent-contrast opacity-0 transition-opacity peer-checked:opacity-100"
				viewBox="0 0 15 15"
				fill="none"
				aria-hidden="true"
			>
				<path
					d="M3 7.5L6.5 11L12 4.5"
					stroke="currentColor"
					stroke-width="1.5"
					stroke-linecap="round"
					stroke-linejoin="round"
				/>
			</svg>
		</span>
	{:else if checkboxSpacer}
		<span class="flex-shrink-0 w-4 mr-1" aria-hidden="true"></span>
	{/if}

	<div class="flex items-center flex-1 min-w-0 {nav ? (navTree ? 'gap-2' : 'gap-2.5') : 'gap-3'}">
		{#if leading}
			{@render leading()}
		{:else if leadingKind === 'icon'}
			<span class="flex-shrink-0 {inactive ? 'opacity-50' : ''}">
				<Icon
					name={icon as string}
					className="{size === 'sm' ? 'w-3.5 h-3.5' : nav && !navTree ? 'w-[17px] h-[17px]' : 'w-4 h-4'} {selected
						? 'text-signal'
						: navTree
							? 'text-fg-muted'
							: nav
								? ''
								: 'text-fg-subtle'}"
					strokeWidth={nav ? 1.75 : 2}
				/>
			</span>
		{:else if leadingKind === 'dot'}
			<span
				class="flex-shrink-0 w-2 h-2 rounded-full {inactive ? 'opacity-50' : ''}"
				style="background-color: {dot};"
				aria-hidden="true"
			></span>
		{:else if leadingKind === 'thumbnail'}
			<span
				class="flex-shrink-0 rounded overflow-hidden border border-line flex items-center justify-center {thumbnail
					? 'bg-surface-2'
					: thumbFallback
						? ''
						: 'bg-surface-3'} {thumbSizeClasses[thumbSize ?? 'md']} {inactive ? 'opacity-50' : ''}"
				style={!thumbnail && thumbFallback ? placeholderTint(thumbFallback) : undefined}
			>
				{#if thumbnail}
					<img src={thumbnail} alt="" class="w-full h-full object-cover" />
				{:else}
					<Icon name="image" className="{thumbIconSizeClasses[thumbSize ?? 'md']} text-fg-subtle" />
				{/if}
			</span>
		{/if}

		<div class="flex-1 min-w-0 {inactive ? 'opacity-50' : ''}">
			{#if children}
				{@render children()}
			{:else}
				<div class="flex items-center gap-1.5 min-w-0">
					{#if title !== undefined}
						<span class="truncate {size === 'md' ? 'font-medium text-fg' : ''}">{title}</span>
					{/if}
					{@render badges?.()}
					{#if inactive && inactiveBadge}
						<Badge size="sm">{inactiveBadge}</Badge>
					{/if}
				</div>
				{#if subtitle !== undefined}
					<div class="truncate text-fg-subtle {subtitleMono ? 'font-mono text-2xs' : 'text-xs mt-0.5'}">
						{subtitle}
					</div>
				{/if}
				{@render meta?.()}
			{/if}
		</div>
	</div>

	{#if nav}
		{#if actions}
			<div
				class="ml-0.5 grid h-[30px] w-[34px] flex-shrink-0 place-items-center max-md:h-[38px] max-md:w-16 max-md:grid-cols-[1fr_32px] max-md:gap-[3px] [@media(hover:none)]:h-[38px] [@media(hover:none)]:w-16 [@media(hover:none)]:grid-cols-[1fr_32px] [@media(hover:none)]:gap-[3px]"
			>
				{#if count !== undefined}
					<span
						class="font-mono text-sm tabular-nums {countClass} {revealActions
							? 'hidden max-md:inline [@media(hover:none)]:inline'
							: 'group-hover:hidden group-focus-within:hidden max-md:!inline [@media(hover:none)]:!inline'}"
						>{count}</span
					>
				{/if}
				<div
					class="items-center justify-center {revealActions
						? 'flex'
						: 'hidden group-hover:flex group-focus-within:flex max-md:flex [@media(hover:none)]:flex'}"
				>
					{@render actions()}
				</div>
			</div>
		{:else if count !== undefined}
			<span class="ml-2 flex-shrink-0 font-mono text-sm tabular-nums {countClass}">{count}</span>
		{/if}
	{:else if count !== undefined}
		<span class="flex-shrink-0 ml-2 font-mono text-2xs tabular-nums text-fg-subtle">{count}</span>
	{/if}

	{@render trailing?.()}

	{#if actions && !nav}
		<div
			class="flex-shrink-0 flex items-center gap-1 ml-2 {revealActions
				? 'opacity-100'
				: 'opacity-0 group-hover:opacity-100 focus-within:opacity-100'}"
		>
			{@render actions()}
		</div>
	{/if}

	{#if preview}
		<div
			class="absolute left-full top-0 ml-2 z-50 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity rounded-lg shadow-floating"
		>
			{@render preview()}
		</div>
	{/if}
</div>

<style>
	.pane-row:focus {
		outline: none;
	}

	:global(:where(html:not([data-input='pointer']))) .pane-row:focus-visible {
		box-shadow: inset 0 0 0 2px rgb(var(--signal) / 0.2);
	}
</style>
