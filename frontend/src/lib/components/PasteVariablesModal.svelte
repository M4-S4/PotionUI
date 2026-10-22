<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import SegmentModal from './modals/SegmentModal.svelte';
	import SegmentFormRow from './modals/SegmentFormRow.svelte';
	import { findNameCollision } from '$lib/utils/variableNames';
	import { parseVariablesImport, mergeVariablesDetailed, remapNote, type ParsedVariablesImport } from '$lib/utils/variablesTransfer';
	import type { VariablesMap } from '$lib/utils/variableDefs';

	export let isOpen = false;
	export let existing: VariablesMap = {};

	const dispatch = createEventDispatcher<{ close: void; confirm: { variables: VariablesMap; count: number } }>();

	let text = '';
	let mergeRule: 'keep' | 'replace' = 'replace';
	let previousOpen = false;

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
</script>

<SegmentModal
	{isOpen}
	tier="M"
	icon="clipboard-list"
	title="Paste variables"
	context={`Into Prompt variables · ${Object.keys(existing).length} now`}
	primaryLabel="Import"
	primaryDisabled={!canConfirm}
	submit="ctrl"
	onConfirm={confirmImport}
	onCancel={() => dispatch('close')}
>
	<div class="space-y-4 p-5 max-md:p-4">
		<SegmentFormRow label="Variables JSON" forId="paste-variables-json">
			<textarea
				id="paste-variables-json"
				class="input w-full resize-y font-mono text-sm leading-5"
				rows="9"
				placeholder={'{ "mood": { "type": "text", "value": "moody" } }'}
				bind:value={text}
				data-autofocus
				aria-label="Variables JSON"
			></textarea>
		</SegmentFormRow>

		{#if parsed.errors.length > 0}
			<ul class="space-y-1">
				{#each parsed.errors as error}
					<li class="font-mono text-sm text-danger">{error}</li>
				{/each}
			</ul>
		{:else if text.trim()}
			<p class="font-mono text-sm text-fg-subtle">{summaryLine}</p>
			{#if parsed.notes.length > 0 || merge.remapped.length > 0}
				<ul class="space-y-1" data-testid="paste-notes">
					{#each parsed.notes as note}
						<li class="font-mono text-sm text-warning">{note}</li>
					{/each}
					{#each merge.remapped as remap}
						<li class="font-mono text-sm text-warning">Matched by name: {remapNote(remap)}</li>
					{/each}
				</ul>
			{/if}
		{/if}

		{#if showMergeControl}
			<div class="space-y-1.5">
				<span class="block text-sm font-medium text-fg-muted">When a name already exists</span>
				<div class="inline-grid grid-cols-2 gap-0.5 rounded border border-line bg-surface-2 p-0.5" role="tablist" aria-label="On name clash">
					{#each [['keep', 'Keep mine'], ['replace', 'Replace']] as [rule, label] (rule)}
						<button
							type="button"
							role="tab"
							aria-selected={mergeRule === rule}
							class="rounded-sm px-2.5 py-1 text-sm font-medium transition-colors max-md:py-2.5 {mergeRule === rule
								? 'bg-surface-3 text-fg shadow-raised'
								: 'text-fg-muted hover:text-fg'}"
							on:click={() => (mergeRule = rule as 'keep' | 'replace')}
						>
							{label}
						</button>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</SegmentModal>
