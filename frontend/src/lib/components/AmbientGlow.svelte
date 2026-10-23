<script lang="ts">
	import { fade } from 'svelte/transition';

	let {
		src,
		kind = 'image',
		variant = 'glow',
		keyed = true,
		crossfade = 0,
		reducedMotion = false,
		class: className = '',
		style = ''
	}: {
		src: string | null | undefined;
		kind?: 'image' | 'video';
		variant?: 'glow' | 'fill';
		keyed?: boolean;
		crossfade?: number;
		reducedMotion?: boolean;
		class?: string;
		style?: string;
	} = $props();

	let outro = $derived({ duration: reducedMotion ? 0 : crossfade });
</script>

{#snippet media()}
	{#if src && kind === 'video'}
		<video {src} class="ambient-media" autoplay muted loop playsinline preload="auto" out:fade|global={outro}></video>
	{:else if src}
		<img {src} alt="" class="ambient-media" out:fade|global={outro} />
	{/if}
{/snippet}

<div class="ambient ambient--{variant} {className}" {style} aria-hidden="true">
	{#if keyed}
		{#key src}
			{@render media()}
		{/key}
	{:else}
		{@render media()}
	{/if}
</div>

<style>
	.ambient {
		position: absolute;
		inset: 0;
		z-index: -1;
		pointer-events: none;
	}

	.ambient--fill {
		overflow: hidden;
	}

	.ambient-media {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: cover;
		filter: blur(var(--ambient-blur, 90px)) saturate(var(--ambient-saturate, 1.4));
		opacity: var(--ambient-opacity, 0.6);
		animation: ambient-in var(--ambient-fade, 0.5s) var(--ease-out-quart, ease-out);
	}

	.ambient--fill .ambient-media {
		transform: scale(1.35);
		filter: blur(var(--ambient-blur, 56px)) saturate(var(--ambient-saturate, 1.35));
		opacity: var(--ambient-opacity, 0.55);
		will-change: transform;
	}

	@keyframes ambient-in {
		from {
			opacity: 0;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.ambient-media {
			animation: none;
		}
	}
</style>
