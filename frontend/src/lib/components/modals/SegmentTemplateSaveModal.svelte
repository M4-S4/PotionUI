<script lang="ts">
	import { untrack } from 'svelte';
	import SegmentModal from './SegmentModal.svelte';
	import SegmentFormRow from './SegmentFormRow.svelte';
	import { api } from '$lib/services/api';
	import type { Segment } from '$lib/types/segments';
	import { flattenRichSegments, toRichSegment } from '$lib/utils/richSegments';
	import { toasts } from '$lib/stores/toast';

	let {
		isOpen,
		segments,
		onClose,
		onSaved
	}: {
		isOpen: boolean;
		segments: Segment[];
		onClose: () => void;
		onSaved: () => void;
	} = $props();

	let name = $state('');
	let description = $state('');
	let saving = $state(false);
	let nameMissing = $state(false);

	let preview = $derived(flattenRichSegments(segments));
	let sourceNames = $derived([...new Set(segments.map((s) => s.template?.name).filter(Boolean))]);

	$effect(() => {
		if (!isOpen) return;
		untrack(() => {
			name = sourceNames.length === 1 ? (sourceNames[0] as string) : '';
			description = '';
			nameMissing = false;
		});
	});

	async function save() {
		if (!name.trim()) {
			nameMissing = true;
			return false;
		}
		saving = true;
		try {
			const response = await api.createSegmentTemplate({
				name: name.trim(),
				description: description.trim() || null,
				tags: [],
				segments: segments.map(toRichSegment)
			});
			if (!response.success || !response.data) throw new Error(response.error || 'Save failed');
			toasts.success('Segment Template saved');
			onSaved();
			return true;
		} catch (error) {
			toasts.error(error instanceof Error ? error.message : 'Failed to save Segment Template');
			return false;
		} finally {
			saving = false;
		}
	}
</script>

<SegmentModal
	{isOpen}
	tier="M"
	icon="layout-template"
	title="Save as Segment Template"
	context="{segments.length} segment{segments.length === 1 ? '' : 's'}"
	primaryLabel="Save"
	busy={saving}
	onConfirm={save}
	onCancel={onClose}
>
	<div class="space-y-4 p-5 max-md:p-4">
		<SegmentFormRow label="Name" forId="segment-template-name" problem={nameMissing ? 'Give it a name to save it.' : ''}>
			<input
				id="segment-template-name"
				class="input {nameMissing ? 'border-warning/60 ring-1 ring-warning/40' : ''}"
				bind:value={name}
				placeholder="How you'll find it later"
				aria-invalid={nameMissing ? 'true' : undefined}
				data-autofocus
				oninput={() => (nameMissing = false)}
			/>
		</SegmentFormRow>
		<SegmentFormRow label="Description" forId="segment-template-description" optional>
			<input id="segment-template-description" class="input" bind:value={description} placeholder="What this structure is for" />
		</SegmentFormRow>
		<SegmentFormRow label="Composition" helper="{segments.length} segment{segments.length === 1 ? '' : 's'}, disabled ones included.">
			<div class="max-h-36 overflow-y-auto rounded-lg border border-line bg-surface-2 p-3 text-base text-fg-muted">{preview || 'Blank segments'}</div>
		</SegmentFormRow>
	</div>
</SegmentModal>
