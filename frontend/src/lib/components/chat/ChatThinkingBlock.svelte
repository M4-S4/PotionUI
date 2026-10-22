<script lang="ts">
	import { slide } from 'svelte/transition';
	import type { ThinkingSegment } from '$lib/types/chat';
	import Icon from '$lib/components/Icon.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { formatThinkingElapsed } from '$lib/chat/thinkingTime';

	let {
		segments = [],
		isStreaming = false
	}: { segments?: ThinkingSegment[]; isStreaming?: boolean } = $props();

	let userOpen = $state<boolean | null>(null);
	let now = $state(Date.now());
	let bodyEl = $state<HTMLDivElement | null>(null);

	const shown = $derived(segments.filter((segment) => segment.text.trim().length > 0));
	const live = $derived(isStreaming && segments.some((segment) => segment.open));
	const open = $derived(userOpen ?? live);

	const elapsedMs = $derived.by(() => {
		let known = false;
		let total = 0;
		for (const segment of segments) {
			if (segment.open && segment.started_at) {
				total += Math.max(0, now - segment.started_at);
				known = true;
			} else if (typeof segment.duration_ms === 'number') {
				total += segment.duration_ms;
				known = true;
			}
		}
		return known ? total : null;
	});

	const label = $derived.by(() => {
		if (live) return 'Thinking…';
		return elapsedMs === null ? 'Thought' : `Thought for ${formatThinkingElapsed(elapsedMs)}`;
	});

	$effect(() => {
		if (!live) return;
		now = Date.now();
		const handle = setInterval(() => (now = Date.now()), 250);
		return () => clearInterval(handle);
	});

	const pinnedGrowth = $derived(
		live && open ? shown.reduce((sum, segment) => sum + segment.text.length, 0) : -1
	);

	$effect(() => {
		if (pinnedGrowth >= 0 && bodyEl) bodyEl.scrollTop = bodyEl.scrollHeight;
	});

	function toggle() {
		userOpen = !open;
	}
</script>

{#if shown.length > 0}
	<div
		class="thinking-block mb-3 w-[min(100%,500px)] overflow-hidden rounded-lg border border-line bg-surface-1"
		data-testid="thinking-block"
		data-state={live ? 'live' : 'done'}
	>
		<button
			type="button"
			class="flex w-full items-center gap-2 px-3 py-2 text-left text-fg-muted hover:bg-surface-2"
			aria-expanded={open}
			data-testid="thinking-toggle"
			onclick={toggle}
		>
			{#if live}
				<Spinner size="sm" />
			{/if}
			<span class="text-sm font-normal text-fg-muted">{label}</span>
			{#if live && elapsedMs !== null}
				<span class="font-mono text-sm font-normal tabular-nums text-fg-subtle" data-testid="thinking-elapsed"
					>{formatThinkingElapsed(elapsedMs)}</span
				>
			{/if}
			<span class="ml-auto flex shrink-0 text-fg-subtle transition-transform" class:rotate-180={open}>
				<Icon name="chat-chevron" className="h-3 w-3" strokeWidth={1.8} />
			</span>
		</button>
		{#if open}
			<div
				bind:this={bodyEl}
				class="max-h-60 overflow-y-auto border-t border-line px-3 py-2"
				role="region"
				aria-label="Model thinking"
				data-testid="thinking-body"
				transition:slide={{ duration: 150 }}
			>
				{#each shown as segment, index}
					<p
						class="m-0 whitespace-pre-wrap break-words text-sm text-fg-muted"
						class:mt-2={index > 0}
						class:border-t={index > 0}
						class:border-line={index > 0}
						class:pt-2={index > 0}
						data-testid="thinking-segment"
					>
						{segment.text}
					</p>
				{/each}
			</div>
		{/if}
	</div>
{/if}
