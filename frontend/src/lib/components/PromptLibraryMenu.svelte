<script module lang="ts">
	export interface FoldedRow {
		key: string;
		icon: string;
		label: string;
		description: string;
		state?: string;
		run: () => void;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import PromptMenuRow from '$lib/components/PromptMenuRow.svelte';
	import { Button, Menu } from '$lib/components/ui';

	let {
		negative = false,
		library = true,
		folded = [],
		hasContent,
		copied,
		showDot = false,
		onInsertPrompt,
		onInsertSegment,
		onInsertTemplate,
		onSavePrompt,
		onSaveTemplate,
		onCopy
	}: {
		negative?: boolean;
		library?: boolean;
		folded?: FoldedRow[];
		hasContent: boolean;
		copied: boolean;
		showDot?: boolean;
		onInsertPrompt: () => void;
		onInsertSegment: () => void;
		onInsertTemplate: () => void;
		onSavePrompt: () => void;
		onSaveTemplate: () => void;
		onCopy: () => void;
	} = $props();

	let viewportWidth = $state(1280);
	const menuWidth = $derived(Math.min(384, viewportWidth - 16));
	const triggerLabel = $derived(library ? 'Library' : negative ? 'More negative prompt actions' : 'More prompt actions');
	const menuLabel = $derived(library && negative ? 'Library (negative)' : triggerLabel);
	const tip = $derived(
		library
			? negative
				? 'Insert into or save the negative prompt'
				: 'Insert from or save to your library'
			: triggerLabel
	);
</script>

<svelte:window bind:innerWidth={viewportWidth} />

<Menu label={triggerLabel} {menuLabel} width={menuWidth} align="right">
	{#snippet trigger(menu)}
		<Tooltip text={tip} position="top">
			{#if library}
				<Button
					size="sm"
					variant="secondary"
					icon="composer-library"
					class="relative aria-expanded:bg-line-hover"
					ariaLabel="Library"
					ariaHaspopup="menu"
					ariaExpanded={menu.open}
					onclick={menu.toggle}
				>
					<span class="header-label">Library</span>
					<Icon name="chevron-down" className="h-[13px] w-[13px] text-fg-subtle transition-transform {menu.open ? 'rotate-180' : ''}" />
					{#if showDot}
						<span class="absolute right-[3px] top-[3px] h-1.5 w-1.5 rounded-full bg-signal" data-testid="library-state-dot"></span>
					{/if}
				</Button>
			{:else}
				<Button
					size="sm"
					variant="secondary"
					icon="more"
					class="header-icon-button aria-expanded:bg-line-hover"
					ariaLabel={triggerLabel}
					ariaHaspopup="menu"
					ariaExpanded={menu.open}
					onclick={menu.toggle}
				/>
			{/if}
		</Tooltip>
	{/snippet}
	{#snippet children(menu)}
		{#if folded.length > 0}
			<div role="group" aria-label="Folded actions">
				{#each folded as row (row.key)}
					<PromptMenuRow
						icon={row.icon}
						label={row.label}
						description={row.description}
						state={row.state}
						onclick={() => {
							menu.close(true);
							row.run();
						}}
					/>
				{/each}
			</div>
			{#if library || hasContent}
				<div class="mx-2 my-1 h-px bg-line" role="separator"></div>
			{/if}
		{/if}
		{#if library}
			<div role="group" aria-label="Insert">
				<div class="px-3 pb-1 pt-2 font-mono text-sm uppercase tracking-[0.05em] text-fg-subtle" aria-hidden="true">Insert</div>
				<PromptMenuRow
					icon="document"
					label="Prompts"
					description="Append or replace with a saved prompt"
					onclick={() => {
						menu.close(true);
						onInsertPrompt();
					}}
				/>
				<PromptMenuRow
					icon="list"
					label="Segments"
					description="Insert one saved segment"
					onclick={() => {
						menu.close(true);
						onInsertSegment();
					}}
				/>
				<PromptMenuRow
					icon="layout-template"
					label="Templates"
					description="Append or replace with a segment template"
					onclick={() => {
						menu.close(true);
						onInsertTemplate();
					}}
				/>
			</div>
			<div class="mx-2 my-1 h-px bg-line" role="separator"></div>
		{/if}
		{#if library || hasContent}
			<div role="group" aria-label="Save">
				{#if library}
					<div class="px-3 pb-1 pt-2 font-mono text-sm uppercase tracking-[0.05em] text-fg-subtle" aria-hidden="true">Save</div>
					<PromptMenuRow
						icon="composer-save"
						label="Save as Prompt"
						description="Keep these segments in your prompt library"
						onclick={() => {
							menu.close(true);
							onSavePrompt();
						}}
					/>
					{#if hasContent}
						<PromptMenuRow
							icon="layout-template"
							label="Save as Segment Template"
							description="Keep this segment layout as a template"
							onclick={() => {
								menu.close(true);
								onSaveTemplate();
							}}
						/>
					{/if}
				{/if}
				{#if hasContent}
					<PromptMenuRow
						icon={copied ? 'composer-check' : 'composer-copy'}
						label={copied ? 'Copied' : 'Copy prompt'}
						description="Copy the text to the clipboard"
						onclick={() => {
							menu.close(true);
							onCopy();
						}}
					/>
				{/if}
			</div>
		{/if}
	{/snippet}
</Menu>
