<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import IconButton from '$lib/components/ui/IconButton.svelte';
	import { describeSource } from './readout';
	import type { MediaToolSource } from './types';

	let {
		clips,
		min,
		max,
		onMove,
		onRemove
	}: {
		clips: MediaToolSource[];
		min: number;
		max: number;
		onMove: (from: number, to: number) => void;
		onRemove: (index: number) => void;
	} = $props();

	let dragging = $state<number | null>(null);
	let over = $state<number | null>(null);

	function handleGripKey(event: KeyboardEvent, index: number) {
		if (!event.altKey) return;
		const delta = event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0;
		if (delta === 0) return;
		event.preventDefault();
		const target = index + delta;
		if (target >= 0 && target < clips.length) onMove(index, target);
	}

	function handleDrop(event: DragEvent, index: number) {
		event.preventDefault();
		if (dragging !== null && dragging !== index) onMove(dragging, index);
		dragging = null;
		over = null;
	}

	let hint = $derived(
		Number.isFinite(max) ? `Joins ${min} to ${max} clips.` : `Joins ${min} or more clips.`
	);
</script>

<section class="flex flex-col gap-2" data-clips>
	<div class="flex items-center gap-1.5">
		<Icon name="video" className="h-3.5 w-3.5 text-fg-subtle" />
		<span class="font-mono text-sm uppercase tracking-[0.07em] text-fg-muted">Clips</span>
		<span class="font-mono text-sm tabular-nums text-fg-subtle" data-clip-count>{clips.length}</span>
		<span class="h-px flex-1 bg-line"></span>
	</div>

	<ol class="flex flex-col gap-1.5">
		{#each clips as clip, index (clip.id)}
			<li
				draggable="true"
				data-clip={index}
				class="flex min-w-0 items-center gap-2 rounded-lg border bg-surface-1 p-1.5 transition-colors {over === index
					? 'border-signal'
					: 'border-line-strong'} {dragging === index ? 'opacity-45' : ''}"
				ondragstart={() => (dragging = index)}
				ondragend={() => {
					dragging = null;
					over = null;
				}}
				ondragover={(event) => {
					event.preventDefault();
					over = index;
				}}
				ondrop={(event) => handleDrop(event, index)}
			>
				<button
					type="button"
					class="grid h-8 w-5 shrink-0 cursor-grab place-items-center rounded text-fg-subtle hover:text-fg"
					aria-label="Reorder clip {index + 1}, hold Alt and use the arrow keys"
					onkeydown={(event) => handleGripKey(event, index)}
				>
					<Icon name="grip" className="h-3.5 w-3.5" />
				</button>
				<span class="w-4 text-right font-mono text-sm tabular-nums text-fg-subtle">{index + 1}</span>
				<span class="h-10 w-10 shrink-0 overflow-hidden rounded bg-surface-3">
					{#if clip.kind === 'video'}
						<video
							src={clip.url}
							muted
							playsinline
							preload="metadata"
							class="h-full w-full object-cover"
							aria-hidden="true"
						></video>
					{:else}
						<span class="grid h-full w-full place-items-center text-fg-subtle">
							<Icon name={clip.kind === 'audio' ? 'audio' : 'image'} className="h-4 w-4" />
						</span>
					{/if}
				</span>
				<span class="flex min-w-0 flex-1 flex-col gap-0.5">
					<span class="truncate text-sm text-fg">{clip.filename}</span>
					<span class="truncate font-mono text-sm tabular-nums text-fg-subtle">{describeSource(clip)}</span>
				</span>
				<IconButton
					icon="close"
					size="sm"
					label="Remove {clip.filename}"
					disabled={clips.length <= min}
					onclick={() => onRemove(index)}
				/>
			</li>
		{/each}
	</ol>

	<p class="text-sm text-fg-muted">Drag to reorder, or Alt and the arrow keys. {hint}</p>
</section>
