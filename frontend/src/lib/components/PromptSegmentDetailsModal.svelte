<script lang="ts">
	import { untrack } from 'svelte';
	import SegmentModal from './modals/SegmentModal.svelte';
	import type { Segment } from '$lib/types/segments';
	import PromptSegmentMetadataEditor from './PromptSegmentMetadataEditor.svelte';

	let {
		isOpen,
		segment,
		onClose,
		onSave
	}: {
		isOpen: boolean;
		segment: Segment;
		onClose: () => void;
		onSave: (updates: {
			name?: string;
			color?: string;
			description?: string;
			prefix?: string;
			suffix?: string;
		}) => void;
	} = $props();

	let draft = $state<Segment>(untrack(() => segment));

	const context = $derived(segment.name || segment.title || 'Untitled segment');

	$effect(() => {
		if (!isOpen) return;
		untrack(() => {
			draft = { ...segment };
		});
	});

	function handleChange(
		event: CustomEvent<Partial<Pick<Segment, 'name' | 'color' | 'description' | 'prefix' | 'suffix'>>>
	) {
		draft = { ...draft, ...event.detail };
	}

	function save() {
		onSave({
			name: draft.name ?? undefined,
			color: draft.color ?? undefined,
			description: draft.description ?? undefined,
			prefix: draft.prefix ?? undefined,
			suffix: draft.suffix ?? undefined
		});
		onClose();
	}
</script>

<SegmentModal {isOpen} tier="M" icon="pencil" title="Segment details" {context} primaryLabel="Save" onConfirm={save} onCancel={onClose}>
	<div class="p-5 max-md:p-4">
		<PromptSegmentMetadataEditor segment={draft} on:change={handleChange} />
		{#if segment.template}
			<p class="mt-3 text-sm text-fg-subtle">
				From template {segment.template.name}, slot {segment.template.slot}
			</p>
		{/if}
	</div>
</SegmentModal>
