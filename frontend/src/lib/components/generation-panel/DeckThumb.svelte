<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import PresetCoverMedia from '$lib/components/preset/PresetCoverMedia.svelte';
	import { fallbackIconForCategory } from '$lib/utils/presetMedia';
	import { placeholderTint } from '$lib/utils/placeholderTint';
	import { DECK_THUMB_ICONS, formatClock, kindForCategory, type DeckThumbView } from './deckThumb';

	let { view }: { view: DeckThumbView } = $props();

	let failed = $state(false);

	const source = $derived(view.tile === 'media' || view.tile === 'live' ? view.src : view.tile === 'cover' ? view.cover : '');
	const fallbackKind = $derived(
		view.tile === 'media' ? view.kind : view.tile === 'icon' ? view.kind : view.tile === 'live' ? 'image' : null
	);
	const label = $derived(
		failed && fallbackKind
			? `${fallbackKind}-missing`
			: view.tile === 'media'
				? view.kind
				: view.tile === 'icon'
					? `${view.kind}${view.pending ? '-pending' : ''}`
					: view.tile === 'cover' && failed
						? 'family'
						: view.tile
	);
	const familyPreset = $derived(view.tile === 'cover' || view.tile === 'family' ? view.preset : null);
	const familyIcon = $derived(fallbackIconForCategory(familyPreset?.category));
	const badges = $derived(view.tile === 'media' || (view.tile === 'icon' && !view.pending) ? view : null);
	const WAVE = [6, 12, 18, 9, 14, 20, 11, 16, 7, 13, 5];

	$effect(() => {
		void source;
		failed = false;
	});
</script>

<div class="deck-thumb" data-thumb={label} aria-hidden="true">
	{#if view.tile === 'live' && !failed}
		<img src={view.src} alt="" onerror={() => (failed = true)} />
	{:else if view.tile === 'media' && !failed}
		{#if view.kind === 'video'}
			<video src={view.src} muted playsinline preload="metadata" onerror={() => (failed = true)}></video>
		{:else}
			<img src={view.src} alt="" onerror={() => (failed = true)} />
		{/if}
	{:else if view.tile === 'cover' && !failed}
		<div class="deck-thumb-cover">
			<PresetCoverMedia presetId={view.preset.id} presetName={view.preset.name} cover={view.cover} zoom={false} onfail={() => (failed = true)} />
		</div>
	{:else if view.tile === 'cover' || view.tile === 'family'}
		<span class="deck-thumb-fill" style={placeholderTint(familyPreset?.name || familyPreset?.id || 'preset')}>
			<Icon name={familyIcon} className="deck-thumb-icon" strokeWidth={1.5} />
		</span>
	{:else if view.tile === 'icon' && view.kind === 'audio' && !view.pending}
		<span class="deck-thumb-fill is-type">
			<svg class="deck-thumb-wave" viewBox="0 0 36 20" fill="currentColor">
				{#each WAVE as height, index}
					<rect x={index * 3.3} y={(20 - height) / 2} width="2" height={height} rx="1" />
				{/each}
			</svg>
		</span>
	{:else}
		<span class="deck-thumb-fill is-type">
			<Icon name={DECK_THUMB_ICONS[fallbackKind ?? kindForCategory(null)]} className="deck-thumb-icon" strokeWidth={1.5} />
		</span>
	{/if}
	{#if badges}
		{#if badges.kind === 'video' && !failed && badges.tile === 'media'}
			<span class="deck-thumb-glyph"><Icon name="composer-play" className="deck-thumb-glyph-icon" /></span>
		{/if}
		{#if badges.duration !== null && (badges.kind === 'video' || badges.kind === 'audio')}
			<span class="deck-thumb-badge is-duration">{formatClock(badges.duration)}</span>
		{/if}
		{#if badges.count > 1}
			<span class="deck-thumb-badge is-count">×{badges.count}</span>
		{/if}
	{/if}
</div>

<style>
	.deck-thumb {
		position: relative;
		display: grid;
		place-items: center;
		width: 60px;
		height: 60px;
		flex-shrink: 0;
		overflow: hidden;
		border: 1px solid rgb(var(--line));
		border-radius: 6px;
		background: rgb(var(--surface-2));
		color: rgb(var(--fg-muted));
	}

	.deck-thumb img,
	.deck-thumb video,
	.deck-thumb-cover {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.deck-thumb-cover :global(img),
	.deck-thumb-cover :global(video) {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.deck-thumb-fill {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
	}

	.deck-thumb-fill.is-type {
		background: rgb(var(--surface-3));
	}

	.deck-thumb :global(.deck-thumb-icon) {
		width: 24px;
		height: 24px;
	}

	.deck-thumb-wave {
		width: 36px;
		height: 20px;
		margin-bottom: 12px;
	}

	.deck-thumb-glyph {
		position: absolute;
		top: 2px;
		left: 2px;
		display: grid;
		place-items: center;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: rgb(var(--scrim) / 0.72);
		color: rgb(var(--on-scrim));
	}

	.deck-thumb-glyph :global(.deck-thumb-glyph-icon) {
		width: 8px;
		height: 8px;
	}

	.deck-thumb-badge {
		position: absolute;
		right: 2px;
		bottom: 2px;
		padding: 0 3px;
		border-radius: 4px;
		background: rgb(var(--scrim) / 0.72);
		color: rgb(var(--on-scrim));
		font: 600 12px/16px 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		font-variant-numeric: tabular-nums;
	}

	.deck-thumb-badge.is-count {
		top: 2px;
		bottom: auto;
	}
</style>
