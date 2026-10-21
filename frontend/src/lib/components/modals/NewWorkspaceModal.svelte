<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import ConfirmDialog from './ConfirmDialog.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { Button } from '$lib/components/ui';

	export let isOpen: boolean = false;
	/** Whether a tab OTHER than the one "Save" will act on also has unsaved
	 *  work — see utils/newWorkspace.ts `hasUnsavedWorkOutsideTab`. */
	export let otherTabsAlsoDirty: boolean = false;
	/** Disables all three actions and shows a spinner on Save while it's in flight. */
	export let busy: boolean = false;

	const dispatch = createEventDispatcher<{ save: void; discard: void; cancel: void }>();
	let dialog: ReturnType<typeof ConfirmDialog>;

	const titleId = `new-workspace-title-${Math.random().toString(36).slice(2, 9)}`;
</script>

<ConfirmDialog
	bind:this={dialog}
	{isOpen}
	size="md"
	hideCloseButton
	{busy}
	dialogRole="alertdialog"
	labelledBy={titleId}
	confirmLabel={busy ? 'Saving…' : 'Save & create new'}
	onConfirm={() => dispatch('save')}
	onCancel={() => dispatch('cancel')}
>
	<div class="p-7">
		<div class="flex items-start gap-4">
			<div class="w-11 h-11 bg-warning/10 rounded-full flex items-center justify-center flex-shrink-0">
				<Icon name="warning" className="w-5 h-5 text-warning" strokeWidth={1.5} />
			</div>
			<div class="min-w-0 pt-0.5">
				<h3 id={titleId} class="text-base font-semibold text-fg mb-1.5 break-words">Unsaved changes</h3>
				<p class="text-sm leading-relaxed text-fg-muted whitespace-pre-line break-words">
					This workspace has unsaved changes. Creating a new workspace closes every open tab.
					{#if otherTabsAlsoDirty}
						Saving keeps only the active tab's changes — other tabs' unsaved edits will be discarded either way.
					{/if}
				</p>
			</div>
		</div>
	</div>
	{#snippet leftActions()}
		<Button variant="danger" disabled={busy} onclick={() => dialog.settle(() => dispatch('discard'))}>Discard &amp; create new</Button>
	{/snippet}
</ConfirmDialog>
