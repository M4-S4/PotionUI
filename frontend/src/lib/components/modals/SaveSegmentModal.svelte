<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { api } from '$lib/services/api';
	import type { Segment, SegmentCategory } from '$lib/types/segments';
	import { toRichSegment } from '$lib/utils/richSegments';
	import { toasts } from '$lib/stores/toast';
	import { resolveResourceMarkers, textHasResourceMarkers, type PromptResourceSpec } from '$lib/utils/promptResources';
	import SegmentModal from './SegmentModal.svelte';
	import SegmentFormRow from './SegmentFormRow.svelte';
	import SegmentColorField from '$lib/components/SegmentColorField.svelte';

	export let isOpen = false;
	export let segment: Segment;
	export let promptResources: PromptResourceSpec[] = [];
	export let resourceFieldValues: Record<string, unknown> = {};
	const dispatch = createEventDispatcher<{ close: void; saved: void }>();
	let categories: SegmentCategory[] = [];
	let name = '';
	let categoryId = '';
	let description = '';
	let color = '';
	let tags = '';
	let saving = false;
	let previousOpen = false;
	let convertReferences = true;
	let nameMissing = false;

	$: hasReferences = textHasResourceMarkers(segment?.content);

	$: if (isOpen !== previousOpen) {
		previousOpen = isOpen;
		if (isOpen) {
			initialize();
		}
	}

	async function initialize() {
		name = segment.name || segment.title || '';
		description = segment.description || '';
		color = segment.color || '';
		tags = '';
		convertReferences = true;
		nameMissing = false;
		try {
			categories = (await api.listSegmentCategories()).data?.categories || [];
			categoryId = categories[0]?.id || '';
		} catch {
			categories = [];
		}
	}

	async function save() {
		if (!name.trim()) {
			nameMissing = true;
			return false;
		}
		if (!categoryId) return false;
		saving = true;
		try {
			const shouldConvert = hasReferences && convertReferences;
			const content = shouldConvert
				? resolveResourceMarkers(segment.content, promptResources, resourceFieldValues)
				: segment.content;
			const rich = toRichSegment({
				...segment,
				content,
				resources: shouldConvert ? {} : segment.resources,
				name: name.trim(),
				description,
				color
			});
			const response = await api.createSavedSegment({
				...rich,
				name: name.trim(),
				category_id: categoryId,
				color: color || null,
				description: description.trim() || null,
				tags: tags.split(',').map((tag) => tag.trim()).filter(Boolean)
			});
			if (!response.success) throw new Error(response.error || 'Failed to save Segment');
			toasts.success('Segment saved to library');
			dispatch('saved');
			return true;
		} catch (error) {
			toasts.error(error instanceof Error ? error.message : 'Failed to save Segment');
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
	title="Save as Segment"
	context={segment?.name || segment?.title || 'Untitled segment'}
	primaryLabel="Save"
	busy={saving}
	primaryDisabled={!categoryId}
	onConfirm={save}
	onCancel={() => dispatch('close')}
>
	<div class="grid grid-cols-2 gap-4 p-5 max-md:grid-cols-1 max-md:p-4">
		<SegmentFormRow label="Name" forId="saved-segment-name" problem={nameMissing ? 'Give it a name to save it.' : ''} class="col-span-2 max-md:col-span-1">
			<input
				id="saved-segment-name"
				class="input {nameMissing ? 'border-warning/60 ring-1 ring-warning/40' : ''}"
				bind:value={name}
				placeholder="How you'll find it in the library"
				aria-invalid={nameMissing ? 'true' : undefined}
				data-autofocus
				on:input={() => (nameMissing = false)}
			/>
		</SegmentFormRow>
		<SegmentFormRow label="Category" forId="saved-segment-category" helper={categories.length === 0 ? 'Create a category in the Prompts library first.' : ''}>
			<select id="saved-segment-category" class="input" bind:value={categoryId} disabled={categories.length === 0}>
				{#each categories as category (category.id)}<option value={category.id}>{category.name}</option>{/each}
			</select>
		</SegmentFormRow>
		<SegmentFormRow label="Tags" forId="saved-segment-tags" optional>
			<input id="saved-segment-tags" class="input" bind:value={tags} placeholder="comma, separated" />
		</SegmentFormRow>
		<SegmentFormRow label="Colour" optional class="col-span-2 max-md:col-span-1">
			<SegmentColorField value={color} onchange={(next) => (color = next)} />
		</SegmentFormRow>
		<SegmentFormRow label="Description" forId="saved-segment-description" optional class="col-span-2 max-md:col-span-1">
			<textarea id="saved-segment-description" class="input resize-y" rows="2" bind:value={description}></textarea>
		</SegmentFormRow>
		{#if hasReferences}
			<div class="col-span-2 space-y-2 rounded-lg border border-warning/40 bg-warning/10 p-3 max-md:col-span-1">
				<p class="text-sm text-fg">
					This segment references media uploaded on this tab. Those references only resolve here, so a library copy needs its own choice.
				</p>
				<label class="flex items-start gap-2 text-sm text-fg">
					<input type="radio" name="reference-handling" checked={convertReferences} on:change={() => (convertReferences = true)} class="mt-0.5" />
					<span>Convert to plain text now (“Picture 2”). Always valid, never dangling.</span>
				</label>
				<label class="flex items-start gap-2 text-sm text-fg">
					<input type="radio" name="reference-handling" checked={!convertReferences} on:change={() => (convertReferences = false)} class="mt-0.5" />
					<span>Keep the references. Only works on a tab with the same uploads.</span>
				</label>
			</div>
		{/if}
		<p class="col-span-2 text-sm text-fg-subtle max-md:col-span-1">Creates a detached copy; later edits to either copy don't sync.</p>
	</div>
</SegmentModal>
