<script lang="ts">
	import { tick } from 'svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { IconButton } from '$lib/components/ui';
	import portal from '$lib/actions/portal';
	import overlayLayer from '$lib/actions/overlayLayer';
	import dismiss, { type DismissReason } from '$lib/actions/dismiss';
	import { computeFlippedMenuPosition, type FlippedMenuPosition } from '$lib/utils/menuPosition';
	import { MAX_BATCH_IMAGES, promptTitle } from '$lib/generation/promptBatch';
	import type { PromptTabData } from '$lib/types/tabs';

	const MENU_WIDTH = 360;
	const LOCKED_COPY =
		'This preset runs one prompt at a time.';

	let {
		prompts,
		active,
		perPrompt,
		canBatch,
		overCap,
		maxPrompts,
		onSelect,
		onAdd,
		onDuplicate,
		onDelete,
		onRename
	}: {
		prompts: PromptTabData[];
		active: number;
		perPrompt: number;
		canBatch: boolean;
		overCap: boolean;
		maxPrompts: number;
		onSelect: (index: number) => void;
		onAdd: () => void;
		onDuplicate: (index: number) => void;
		onDelete: (index: number) => void;
		onRename: (index: number, name: string) => void;
	} = $props();

	let open = $state(false);
	let triggerEl = $state<HTMLButtonElement | null>(null);
	let menuEl = $state<HTMLDivElement | null>(null);
	let position = $state<FlippedMenuPosition>({ left: 0, top: 0, maxHeight: 400 });
	let renaming = $state<number | null>(null);
	let renameDraft = $state('');

	const count = $derived(prompts.length);
	const multiple = $derived(count > 1);
	const activeTitle = $derived(prompts[active] ? promptTitle(prompts[active], active) : '');
	const showTitle = $derived(activeTitle !== '' && activeTitle !== `Prompt ${active + 1}`);
	const activeOverflows = $derived(overCap && active >= maxPrompts);
	const atLimit = $derived(canBatch && count >= maxPrompts);
	const addDisabled = $derived(!canBatch || atLimit);
	const imageWord = $derived(perPrompt === 1 ? 'image' : 'images');

	function reposition() {
		if (!triggerEl) return;
		position = computeFlippedMenuPosition(triggerEl, { width: MENU_WIDTH, heightEstimate: 360 });
	}

	async function openMenu() {
		reposition();
		open = true;
		await tick();
		reposition();
		menuEl?.querySelector<HTMLElement>('[data-current="true"]')?.focus();
	}

	function closeMenu(returnFocus = true) {
		open = false;
		renaming = null;
		if (returnFocus) triggerEl?.focus();
	}

	function toggle() {
		if (open) closeMenu(false);
		else openMenu();
	}

	function pick(index: number) {
		onSelect(index);
		closeMenu();
	}

	function startRename(index: number) {
		renaming = index;
		renameDraft = prompts[index]?.name ?? '';
		tick().then(() => menuEl?.querySelector<HTMLInputElement>('input[data-rename]')?.select());
	}

	function commitRename() {
		if (renaming === null) return;
		const index = renaming;
		renaming = null;
		onRename(index, renameDraft);
	}

	function handleDismiss(reason: DismissReason) {
		if (reason === 'outside') closeMenu(false);
		else if (renaming !== null) renaming = null;
		else closeMenu();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (!open) return;
		if (renaming !== null) return;
		if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
		const rows = [...(menuEl?.querySelectorAll<HTMLElement>('[data-row-main]') ?? [])];
		if (rows.length === 0) return;
		event.preventDefault();
		const at = rows.indexOf(document.activeElement as HTMLElement);
		const next = event.key === 'ArrowDown' ? at + 1 : at - 1;
		rows[(next + rows.length) % rows.length].focus();
	}

	function rowOverflows(index: number): boolean {
		return overCap && index >= maxPrompts;
	}

	function rowDimmed(index: number): boolean {
		return !canBatch && index > 0;
	}
</script>

<svelte:window onkeydown={handleKeydown} onresize={() => open && reposition()} onscrollcapture={() => open && reposition()} />

<button
	bind:this={triggerEl}
	type="button"
	class="inline-flex h-6 min-w-0 max-w-[16rem] items-center gap-1.5 overflow-hidden whitespace-nowrap rounded border px-2 text-sm {multiple ? 'flex-shrink' : 'flex-shrink-0'} transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal
		{activeOverflows
		? 'border-danger/40 bg-danger/10 text-danger'
		: open
			? 'border-line-hover bg-surface-3 text-fg'
			: 'border-line text-fg-muted hover:border-line-hover hover:bg-surface-2 hover:text-fg'}"
	aria-haspopup="dialog"
	aria-expanded={open}
	aria-label={multiple ? `Prompt ${active + 1} of ${count}` : 'Prompts'}
	onclick={toggle}
	data-prompt-picker
	data-toolbar-shrink="132"
>
	{#if multiple}
		<span class="flex-shrink-0 font-mono tabular-nums">Prompt {active + 1} of {count}</span>
		{#if showTitle}
			<span class="min-w-0 truncate {activeOverflows ? '' : 'text-fg-subtle'}">{activeTitle}</span>
		{/if}
	{:else}
		<span class="flex-shrink-0 whitespace-nowrap text-fg-subtle">1 prompt</span>
	{/if}
	<Icon name="chevron-down" className="h-3 w-3 flex-shrink-0 transition-transform {open ? 'rotate-180' : ''}" />
</button>

{#if open}
	<div
		use:portal
		use:overlayLayer
		bind:this={menuEl}
		use:dismiss={{ onDismiss: handleDismiss, ignore: [triggerEl] }}
		role="dialog"
		aria-label="Prompts"
		class="fixed flex flex-col overflow-hidden rounded-xl border border-line-hover bg-surface-2 shadow-overlay"
		style="left: {position.left}px; {position.top !== undefined ? `top: ${position.top}px` : `bottom: ${position.bottom}px`}; width: {MENU_WIDTH}px; max-width: calc(100vw - 16px); max-height: {position.maxHeight}px;"
	>
		<div class="flex items-center justify-between gap-3 border-b border-line px-3 py-2">
			<span class="font-mono text-sm tabular-nums text-fg-muted">{count} {count === 1 ? 'prompt' : 'prompts'}</span>
			{#if multiple}
				<span class="font-mono text-sm tabular-nums text-fg-subtle">Images each: {perPrompt}</span>
			{/if}
		</div>

		<ul class="min-h-0 flex-1 overflow-y-auto">
			{#each prompts as entry, index (index)}
				{@const current = index === active}
				<li
					class="group flex items-center gap-1 border-t border-line pr-2 first:border-t-0 {current ? 'border-l-2 border-l-signal bg-signal/5' : 'border-l-2 border-l-transparent hover:bg-surface-3'}"
				>
					{#if renaming === index}
						<input
							data-rename
							class="m-2 min-w-0 flex-1 rounded border border-signal bg-surface-1 px-2 py-1 text-sm text-fg focus:outline-none"
							type="text"
							aria-label="Prompt name"
							placeholder="Prompt name"
							bind:value={renameDraft}
							onkeydown={(event) => {
								if (event.key === 'Enter') {
									event.preventDefault();
									commitRename();
								}
							}}
							onblur={commitRename}
						/>
					{:else}
						<button
							type="button"
							data-row-main
							data-current={current}
							class="flex min-w-0 flex-1 items-center gap-3 px-3 py-2 text-left focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal {rowDimmed(index) ? 'opacity-50' : ''}"
							aria-current={current ? 'true' : undefined}
							onclick={() => pick(index)}
						>
							<span class="w-5 flex-shrink-0 text-right font-mono text-sm tabular-nums text-fg-subtle">{index + 1}</span>
							<span class="min-w-0 flex-1 truncate text-sm {rowOverflows(index) ? 'text-fg-subtle line-through' : current ? 'text-fg' : 'text-fg-muted'}">{promptTitle(entry, index)}</span>
							<span class="flex-shrink-0 font-mono text-sm tabular-nums {rowOverflows(index) ? 'text-danger' : 'text-fg-subtle'}">
								{rowOverflows(index) ? 'will not run' : rowDimmed(index) ? 'kept' : `${perPrompt} ${imageWord}`}
							</span>
						</button>
						<div class="flex flex-shrink-0 items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 {current ? 'opacity-100' : ''}">
							<Tooltip text="Rename prompt" position="top" delay={150}>
								<IconButton icon="edit" label="Rename prompt" size="xs" onclick={() => startRename(index)} />
							</Tooltip>
							<Tooltip text={addDisabled ? 'Cannot duplicate right now' : 'Duplicate prompt'} position="top" delay={150}>
								<IconButton icon="copy" label="Duplicate prompt" size="xs" disabled={addDisabled} onclick={() => onDuplicate(index)} />
							</Tooltip>
							{#if multiple}
								<Tooltip text="Delete prompt" position="top" delay={150}>
									<IconButton icon="trash" label="Delete prompt" size="xs" onclick={() => onDelete(index)} />
								</Tooltip>
							{/if}
						</div>
					{/if}
				</li>
			{/each}
		</ul>

		<div class="border-t border-line p-2">
			<button
				type="button"
				class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-fg-muted"
				disabled={addDisabled}
				onclick={() => {
					onAdd();
					closeMenu();
				}}
			>
				<Icon name="plus" className="h-3.5 w-3.5" />
				<span>Add prompt</span>
			</button>
			<p class="mt-1 px-2 text-sm text-fg-subtle">
				{#if !canBatch}
					{LOCKED_COPY}
				{:else if atLimit}
					A run can make up to {MAX_BATCH_IMAGES} images. At {perPrompt} {imageWord} each, that is {maxPrompts} prompts.
				{:else}
					Add prompt starts empty. Duplicate copies every segment.
				{/if}
			</p>
		</div>
	</div>
{/if}
