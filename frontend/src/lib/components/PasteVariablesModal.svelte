<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import ConfirmDialog from './modals/ConfirmDialog.svelte';
	import { findNameCollision } from '$lib/utils/variableNames';
	import { parseVariablesImport, mergeVariablesDetailed, remapNote, type ParsedVariablesImport } from '$lib/utils/variablesTransfer';
	import type { VariablesMap } from '$lib/utils/variableDefs';

	export let isOpen = false;
	export let existing: VariablesMap = {};

	const dispatch = createEventDispatcher<{ close: void; confirm: { variables: VariablesMap; count: number } }>();

	let text = '';
	let mergeRule: 'keep' | 'replace' = 'replace';
	let previousOpen = false;
	let dialog: ReturnType<typeof ConfirmDialog>;

	$: if (isOpen !== previousOpen) {
		previousOpen = isOpen;
		if (isOpen) {
			text = '';
			mergeRule = 'replace';
			prefillFromClipboard();
		}
	}

	async function prefillFromClipboard() {
		try {
			const clip = await navigator.clipboard.readText();
			if (clip) text = clip;
		} catch {
			return;
		}
	}

	$: parsed = text.trim()
		? parseVariablesImport(text, existing)
		: ({ variables: {}, summary: { variables: 0, conditions: 0 }, errors: [], notes: [] } as ParsedVariablesImport);
	$: merge = mergeVariablesDetailed(existing, parsed.variables, mergeRule);
	$: clashes = Object.keys(parsed.variables).filter((name) => findNameCollision(existing, name) !== null);
	$: showMergeControl = clashes.length > 0;
	$: canConfirm = text.trim().length > 0 && parsed.errors.length === 0 && parsed.summary.variables > 0;

	function pluralize(count: number, noun: string): string {
		return `${count} ${noun}${count === 1 ? '' : 's'}`;
	}

	$: summaryLine = [
		pluralize(parsed.summary.variables, 'variable'),
		...(parsed.summary.conditions > 0 ? [pluralize(parsed.summary.conditions, 'condition')] : []),
		...(clashes.length > 0 ? [`${pluralize(clashes.length, 'name')} already exist${clashes.length === 1 ? 's' : ''}`] : [])
	].join(' · ');

	function confirmImport() {
		if (!canConfirm) return false;
		dispatch('confirm', { variables: merge.variables, count: parsed.summary.variables });
		return true;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (!isOpen || event.key !== 'Enter' || !(event.metaKey || event.ctrlKey)) return;
		event.preventDefault();
		dialog.confirm();
	}
</script>

<svelte:window on:keydown|capture={handleKeydown} />

<ConfirmDialog
	bind:this={dialog}
	{isOpen}
	title="Paste variables"
	size="sm"
	confirmLabel="Import"
	confirmDisabled={!canConfirm}
	onConfirm={confirmImport}
	onCancel={() => dispatch('close')}
>
	<div class="space-y-3 p-4 sm:p-6">
		<textarea
			class="input w-full resize-none font-mono text-xs leading-relaxed"
			rows="8"
			placeholder={'{ "mood": { "type": "text", "value": "moody" } }'}
			bind:value={text}
			data-autofocus
			aria-label="Variables JSON"
		></textarea>

		{#if parsed.errors.length > 0}
			<ul class="space-y-1">
				{#each parsed.errors as error}
					<li class="font-mono text-xs text-danger">{error}</li>
				{/each}
			</ul>
		{:else if text.trim()}
			<p class="font-mono text-xs text-fg-subtle">{summaryLine}</p>
			{#if parsed.notes.length > 0 || merge.remapped.length > 0}
				<ul class="space-y-1" data-testid="paste-notes">
					{#each parsed.notes as note}
						<li class="font-mono text-xs text-warning">{note}</li>
					{/each}
					{#each merge.remapped as remap}
						<li class="font-mono text-xs text-warning">Matched by name: {remapNote(remap)}</li>
					{/each}
				</ul>
			{/if}
		{/if}

		{#if showMergeControl}
			<div class="inline-flex rounded bg-surface-2 p-0.5" role="tablist" aria-label="On name clash">
				<button
					type="button"
					role="tab"
					aria-selected={mergeRule === 'keep'}
					class="rounded px-2.5 py-1 text-xs font-medium transition-colors {mergeRule === 'keep'
						? 'bg-surface-1 text-fg shadow-raised'
						: 'text-fg-muted hover:text-fg'}"
					on:click={() => (mergeRule = 'keep')}
				>
					Keep mine
				</button>
				<button
					type="button"
					role="tab"
					aria-selected={mergeRule === 'replace'}
					class="rounded px-2.5 py-1 text-xs font-medium transition-colors {mergeRule === 'replace'
						? 'bg-signal/15 text-signal'
						: 'text-fg-muted hover:text-fg'}"
					on:click={() => (mergeRule = 'replace')}
				>
					Replace
				</button>
			</div>
		{/if}
	</div>
</ConfirmDialog>
