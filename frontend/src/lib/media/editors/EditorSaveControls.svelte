<script lang="ts">
	import Tooltip from '$lib/components/Tooltip.svelte';
	/**
	 * The save choice, in one place for all four editors.
	 *
	 * An edit is two different decisions wearing one button: replacing keeps the
	 * resource - and so its tags and its place in every collection - and swaps
	 * the file behind it, while saving as new leaves the original alone. Neither
	 * is safe to guess, so both are offered wherever both are possible, and the
	 * non-destructive one leads.
	 *
	 * Both consumers show both: a form field edits the same library resources the
	 * Library page does, and hiding the choice in one of them would mean the same
	 * media behaves differently depending on which screen it was opened from.
	 */
	import { getContext } from 'svelte';
	import { readable } from 'svelte/store';
	import { Button, Spinner } from '$lib/components/ui';
	import { EDITOR_RUN_CONTEXT_KEY, type EditorRunContext } from './editorRunContext';
	import type { MediaEditorSaveMode } from './types';

	export let busy: boolean = false;
	/** Why saving is unavailable, or null when it is. Shown, never swallowed. */
	export let blockedReason: string | null = null;
	/**
	 * Why the last attempt failed. Shown here rather than as a toast: the
	 * refusal is about the selection still on screen, and a toast would be gone
	 * before the user finished re-reading it.
	 */
	export let failureMessage: string | null = null;
	/** False when this editor's result can only ever be a new resource. */
	export let allowReplace: boolean = true;
	/** The single-choice label, for an editor that cannot replace anything. */
	export let applyLabel: string = 'Save as new';
	export let onCancel: () => void;
	export let onSave: (mode: MediaEditorSaveMode) => void;

	const run = getContext<EditorRunContext | undefined>(EDITOR_RUN_CONTEXT_KEY);
	const progress = run?.progress ?? readable<number | null>(null);

	$: disabled = busy || blockedReason !== null;
	$: percent = $progress === null ? null : Math.round(Math.min(1, Math.max(0, $progress)) * 100);
</script>

{#if failureMessage}
	<p class="min-w-0 flex-1 text-xs text-danger">{failureMessage}</p>
{:else if blockedReason}
	<p class="min-w-0 flex-1 text-xs text-warning">{blockedReason}</p>
{:else if busy}
	<span class="inline-flex items-center gap-2 text-xs text-fg-muted">
		<Spinner size="sm" />
		Working…
		{#if percent !== null}
			<span class="font-mono tabular-nums" data-editor-progress>{percent}%</span>
		{/if}
	</span>
{/if}

<div class="ml-auto flex items-center gap-2">
	{#if busy && run}
		<Button variant="secondary" size="sm" onclick={() => run.cancel()}>Cancel edit</Button>
	{:else}
		<Button variant="ghost" size="sm" onclick={onCancel} disabled={busy}>Cancel</Button>
	{/if}

	{#if allowReplace}
		<Tooltip text="Swap the file behind this item, keeping its tags and collections">
			<Button variant="secondary" size="sm" {disabled} onclick={() => onSave('replace')}>
				Replace original
			</Button>
		</Tooltip>
	{/if}

	<Button variant="primary" size="sm" icon="check" {disabled} onclick={() => onSave('new')}>
		{allowReplace ? 'Save as new' : applyLabel}
	</Button>
</div>
