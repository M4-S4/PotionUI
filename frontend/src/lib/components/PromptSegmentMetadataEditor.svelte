<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import SegmentFormRow from './modals/SegmentFormRow.svelte';
	import SegmentColorField from './SegmentColorField.svelte';
	import type { Segment } from '$lib/types/segments';

	export let segment: Segment;

	const dispatch = createEventDispatcher();

	function update(field: 'name' | 'color' | 'description' | 'prefix' | 'suffix', value: string) {
		dispatch('change', { [field]: value || undefined });
	}
</script>

<div class="grid grid-cols-2 gap-4 max-md:grid-cols-1">
	<SegmentFormRow label="Name" forId="segment-details-name" optional>
		<input
			id="segment-details-name"
			type="text"
			class="input"
			value={segment.name || segment.title || ''}
			placeholder="Optional segment name"
			data-autofocus
			on:input={(event) => update('name', event.currentTarget.value)}
		/>
	</SegmentFormRow>

	<SegmentFormRow label="Colour" optional>
		<SegmentColorField value={segment.color || ''} onchange={(color) => update('color', color)} />
	</SegmentFormRow>

	<SegmentFormRow label="Prefix" forId="segment-details-prefix">
		<input
			id="segment-details-prefix"
			type="text"
			class="input font-mono"
			value={segment.prefix || ''}
			placeholder="Joined before the text"
			on:input={(event) => update('prefix', event.currentTarget.value)}
		/>
	</SegmentFormRow>

	<SegmentFormRow label="Suffix" forId="segment-details-suffix">
		<input
			id="segment-details-suffix"
			type="text"
			class="input font-mono"
			value={segment.suffix || ''}
			placeholder="Joined after the text"
			on:input={(event) => update('suffix', event.currentTarget.value)}
		/>
	</SegmentFormRow>

	<SegmentFormRow label="Description" forId="segment-details-description" optional class="col-span-2 max-md:col-span-1">
		<textarea
			id="segment-details-description"
			class="input resize-y"
			rows="3"
			value={segment.description || ''}
			placeholder="Optional notes about this segment"
			on:input={(event) => update('description', event.currentTarget.value)}
		></textarea>
	</SegmentFormRow>
</div>
