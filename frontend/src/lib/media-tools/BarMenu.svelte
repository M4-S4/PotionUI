<script lang="ts">
	import { tick, type Snippet } from 'svelte';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import dismiss, { type DismissReason } from '$lib/actions/dismiss';
	import menuNav, { focusFirstItem } from '$lib/actions/menuNav';
	import IconButton from '$lib/components/ui/IconButton.svelte';
	import { trackMenuPosition, type FlippedMenuPosition } from '$lib/utils/menuPosition';

	const ROW_SELECTOR = '[role="menuitem"]:not([aria-disabled="true"])';

	let {
		open = $bindable(false),
		label,
		sheet = false,
		align = 'left',
		width = 288,
		subtitle = '',
		onbarnav,
		trigger,
		children
	}: {
		open?: boolean;
		label: string;
		sheet?: boolean;
		align?: 'left' | 'right';
		width?: number;
		subtitle?: string;
		onbarnav?: (direction: -1 | 1) => void;
		trigger: Snippet<[{ open: boolean; toggle: () => void }]>;
		children: Snippet<[{ close: (restoreFocus?: boolean) => void }]>;
	} = $props();

	let triggerEl = $state<HTMLElement>();
	let menuEl = $state<HTMLElement>();
	let position = $state<FlippedMenuPosition | null>(null);
	let focused = false;

	export function toggle() {
		if (open) close(true);
		else open = true;
	}

	export function triggerButton(): HTMLElement | null {
		return triggerEl?.querySelector<HTMLElement>('button') ?? null;
	}

	function close(restoreFocus = false) {
		if (!open) return;
		open = false;
		position = null;
		focused = false;
		if (restoreFocus) void tick().then(() => triggerButton()?.focus());
	}

	function onDismiss(reason: DismissReason) {
		close(reason === 'escape');
	}

	function onMenuKeydown(event: KeyboardEvent) {
		if (event.key === 'Tab') {
			triggerButton()?.focus();
			close(false);
			return;
		}
		if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
			event.preventDefault();
			event.stopPropagation();
			const direction = event.key === 'ArrowRight' ? 1 : -1;
			triggerButton()?.focus();
			close(false);
			onbarnav?.(direction);
		}
	}

	function menuStyle(pos: FlippedMenuPosition | null): string {
		if (!pos) return `visibility: hidden; top: 0; left: 0; width: ${width}px;`;
		const vertical = pos.top !== undefined ? `top: ${pos.top}px;` : `bottom: ${pos.bottom}px;`;
		return `${vertical} left: ${pos.left}px; width: ${width}px; max-height: ${pos.maxHeight}px;`;
	}

	$effect(() => {
		if (!open || sheet || !triggerEl || !menuEl) return;
		const menu = menuEl;
		return trackMenuPosition(
			triggerEl,
			() => ({ width, heightEstimate: menu.offsetHeight, gap: 4, align, preferred: 'down' }),
			(next) => (position = next)
		);
	});

	$effect(() => {
		if (!open || !menuEl || focused) return;
		if (!sheet && !position) return;
		focused = true;
		const menu = menuEl;
		void tick().then(() => focusFirstItem(menu, ROW_SELECTOR));
	});
</script>

<span bind:this={triggerEl} class="inline-flex">
	{@render trigger({ open, toggle })}
</span>

{#if open}
	{#if sheet}
		<div use:portal use:overlayLayer class="fixed inset-0 flex flex-col justify-end" data-bar-sheet-root>
			<div class="absolute inset-0 bg-black/60" aria-hidden="true"></div>
			<div
				bind:this={menuEl}
				use:dismiss={{ onDismiss, ignore: [triggerEl] }}
				use:menuNav={ROW_SELECTOR}
				onkeydown={onMenuKeydown}
				role="menu"
				tabindex="-1"
				aria-label={label}
				class="relative flex max-h-[80dvh] flex-col overflow-y-auto rounded-t-xl border-t border-line-strong bg-surface-1 p-2 shadow-overlay"
				data-bar-menu
				data-bar-sheet
			>
				<div class="mx-auto mb-2 mt-1 h-1 w-9 rounded bg-line-strong" aria-hidden="true"></div>
				<div class="flex items-center gap-3 px-3 pb-2" data-bar-sheet-header>
					<div class="min-w-0 flex-1">
						<p class="truncate font-mono text-sm text-fg">{label}</p>
						{#if subtitle}
							<p class="truncate font-mono text-[12px] uppercase tracking-[0.07em] text-fg-subtle">{subtitle}</p>
						{/if}
					</div>
					<IconButton icon="close" label="Close menu" size="sm" onclick={() => close(true)} />
				</div>
				{@render children({ close })}
			</div>
		</div>
	{:else}
		<div
			bind:this={menuEl}
			use:portal
			use:overlayLayer
			use:dismiss={{ onDismiss, ignore: [triggerEl] }}
			use:menuNav={ROW_SELECTOR}
			onkeydown={onMenuKeydown}
			role="menu"
			tabindex="-1"
			aria-label={label}
			class="fixed flex flex-col overflow-y-auto rounded-xl border border-line-strong bg-surface-1 p-1 shadow-overlay"
			style={menuStyle(position)}
			data-bar-menu
		>
			{@render children({ close })}
		</div>
	{/if}
{/if}
