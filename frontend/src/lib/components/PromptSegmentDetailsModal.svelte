<script lang="ts">
	import { untrack } from 'svelte';
	import ConfirmDialog from './modals/ConfirmDialog.svelte';
	import { Kbd } from '$lib/components/ui';
	import type { Segment } from '$lib/types/segments';
	import PromptSegmentMetadataEditor from './PromptSegmentMetadataEditor.svelte';

	// Replaces the old inline reveal under the card/content: name, colour,
	// description and the prefix/suffix affixes now edit in a draft that only
	// lands on the real segment when Save (or Enter) commits it — Cancel (or
	// Esc) discards it untouched.
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

	let draft = $state<Segment>(segment);
	let dialog: ReturnType<typeof ConfirmDialog>;

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

<ConfirmDialog
	bind:this={dialog}
	{isOpen}
	title="Segment details"
	size="lg"
	confirmLabel="Save details"
	onConfirm={save}
	onCancel={onClose}
>
	<div class="segment-composer">
		<div class="p-4 sm:p-6">
			<PromptSegmentMetadataEditor segment={draft} on:change={handleChange} />

			{#if segment.template}
				<p class="mt-1 text-xs text-fg-subtle">
					From template {segment.template.name}, slot {segment.template.slot}
				</p>
			{/if}
		</div>
	</div>

	{#snippet footerContent()}
		<div class="segment-composer details-footer w-full">
			<footer class="modal-foot">
				<button type="button" class="small-button" onclick={() => dialog.cancel()}>
					Cancel
					<Kbd keys="Esc" />
				</button>
				<button type="button" class="small-button primary" onclick={() => dialog.confirm()}>
					Save details
					<Kbd keys="Enter" />
				</button>
			</footer>
		</div>
	{/snippet}
</ConfirmDialog>

<style>
	/* The Save button used to be autofocused via ConfirmFooter's `initialFocus`
	   (a focusTrap requestAnimationFrame `.focus()`), which browsers treat as
	   keyboard-origin focus and ring accordingly — now the mock's own footer
	   buttons, so that autofocus is gone with it; Escape/Enter still reach the
	   same confirm/cancel path via the window keydown handler above regardless
	   of where focus sits. */
	.details-footer :global(button:focus-visible) {
		outline: 2px solid rgb(var(--signal));
		outline-offset: 2px;
	}
</style>
