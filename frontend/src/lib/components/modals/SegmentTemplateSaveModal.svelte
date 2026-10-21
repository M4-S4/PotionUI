<script lang="ts">
	import { untrack } from 'svelte';
	import ConfirmDialog from './ConfirmDialog.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { Input } from '$lib/components/ui';
	import { api } from '$lib/services/api';
	import type { Segment } from '$lib/types/segments';
	import { flattenRichSegments, toRichSegment } from '$lib/utils/richSegments';
	import { toasts } from '$lib/stores/toast';

	// Forks the editor's current cards into the user's own Segment Template
	// library. A preset-declared template has no database row, so this is the
	// only way to keep an edited copy of one.
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

	let preview = $derived(flattenRichSegments(segments));
	let sourceNames = $derived([...new Set(segments.map((s) => s.template?.name).filter(Boolean))]);

	$effect(() => {
		if (!isOpen) return;
		untrack(() => {
			name = sourceNames.length === 1 ? (sourceNames[0] as string) : '';
			description = '';
		});
	});

	async function save() {
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

<ConfirmDialog
	{isOpen}
	title="Save as Segment Template"
	size="md"
	busy={saving}
	confirmLabel="Save Template"
	confirmDisabled={!name.trim()}
	onConfirm={save}
	onCancel={onClose}
>
	{#snippet headerIconContent()}
		<Icon name="layout-template" className="h-5 w-5 text-fg-muted" />
	{/snippet}

	<div class="space-y-4 p-4 sm:p-6">
		<div>
			<label class="mb-1.5 block text-sm font-medium text-fg" for="segment-template-name">Name</label>
			<Input id="segment-template-name" bind:value={name} placeholder="How you will find it later" />
		</div>

		<div>
			<label class="mb-1.5 block text-sm font-medium text-fg" for="segment-template-description">
				Description <span class="font-normal text-fg-subtle">(optional)</span>
			</label>
			<Input id="segment-template-description" bind:value={description} placeholder="What this structure is for" />
		</div>

		<div>
			<span class="mb-1.5 block text-sm font-medium text-fg">Composition</span>
			<div class="max-h-36 overflow-y-auto rounded-lg border border-line bg-surface-2 p-3 text-sm text-fg-muted">
				{preview || 'Blank segments'}
			</div>
			<p class="mt-1.5 text-xs text-fg-subtle">
				{segments.length} segment{segments.length === 1 ? '' : 's'}, disabled ones included.
			</p>
		</div>
	</div>
</ConfirmDialog>
