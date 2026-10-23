<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { api } from '$lib/services/api';
	import type { PromptUsageHint, Segment } from '$lib/types/segments';
	import { flattenRichSegments, toRichSegment } from '$lib/utils/richSegments';
	import type { VariablesMap } from '$lib/utils/variableDefs';
	import { summarizeVariables } from '$lib/utils/variablesTransfer';
	import { toasts } from '$lib/stores/toast';
	import SegmentModal from './SegmentModal.svelte';
	import SegmentFormRow from './SegmentFormRow.svelte';

	export let isOpen = false;
	export let segments: Segment[] = [];
	export let usageHint: PromptUsageHint = 'positive';
	export let variables: VariablesMap = {};
	const dispatch = createEventDispatcher<{ close: void; saved: { id: string } }>();
	let name = '';
	let saving = false;
	let previousOpen = false;
	$: if (isOpen !== previousOpen) { previousOpen = isOpen; if (isOpen) { name = ''; } }
	$: preview = flattenRichSegments(segments);
	$: variablesSummary = summarizeVariables(variables);
	$: variablesSummaryLine =
		variablesSummary.variables > 0
			? `Includes ${variablesSummary.variables} variable${variablesSummary.variables === 1 ? '' : 's'} · ${variablesSummary.conditions} condition${variablesSummary.conditions === 1 ? '' : 's'}`
			: '';

	async function save() {
		saving = true;
		try {
			const response = await api.createPrompt({
				name: name.trim() || null,
				usage_hint: usageHint,
				segments: segments.map(toRichSegment),
				variables: Object.keys(variables).length ? variables : null
			});
			if (!response.success || !response.data) throw new Error(response.error || 'Failed to save Prompt');
			toasts.success('New detached Prompt saved');
			dispatch('saved', { id: response.data.id });
			return true;
		} catch (error) {
			toasts.error(error instanceof Error ? error.message : 'Failed to save Prompt');
			return false;
		} finally {
			saving = false;
		}
	}
</script>

<SegmentModal
	{isOpen}
	tier="M"
	icon="save"
	title="Save as Prompt"
	context="{segments.length} segment{segments.length === 1 ? '' : 's'} · {usageHint} prompt"
	primaryLabel="Save"
	busy={saving}
	onConfirm={save}
	onCancel={() => dispatch('close')}
>
	<div class="space-y-4 p-5 max-md:p-4">
		<SegmentFormRow label="Name" forId="saved-prompt-name" optional>
			<input id="saved-prompt-name" class="input" bind:value={name} placeholder="The text preview is used when unnamed" data-autofocus />
		</SegmentFormRow>
		<SegmentFormRow label="Composition" helper="{segments.length} segment{segments.length === 1 ? '' : 's'} · {usageHint} usage hint">
			<div class="max-h-36 overflow-y-auto rounded-lg border border-line bg-surface-2 p-3 text-base text-fg-muted">{preview || 'Blank prompt'}</div>
		</SegmentFormRow>
		{#if variablesSummaryLine}
			<p class="font-mono text-sm text-fg-subtle">{variablesSummaryLine}</p>
		{/if}
		<p class="text-sm text-fg-subtle">Only this segment list is saved. Preset, mode, form values, session, backend, seed, tags, and generation settings are not included.</p>
	</div>
</SegmentModal>
