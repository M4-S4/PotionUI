<script lang="ts">
	import Spinner from '$lib/components/ui/Spinner.svelte';

	let {
		beforeUrl,
		beforeKind,
		afterUrl,
		loading = false,
		error = null,
		aspect = 4 / 3
	}: {
		beforeUrl: string;
		beforeKind: 'image' | 'video';
		afterUrl: string | null;
		loading?: boolean;
		error?: string | null;
		aspect?: number;
	} = $props();

	let split = $state(50);
	let afterAspect = $state<number | null>(null);

	let shown = $derived(Math.min(2.2, Math.max(0.5, afterAspect ?? aspect)));

	function handleAfterLoad(event: Event) {
		const image = event.currentTarget as HTMLImageElement;
		if (image.naturalWidth > 0 && image.naturalHeight > 0) {
			afterAspect = image.naturalWidth / image.naturalHeight;
		}
	}
</script>

<section class="flex flex-col gap-2" data-preview aria-label="Preview">
	<div class="flex items-center gap-2">
		<span class="label !mb-0">Preview</span>
		<span class="font-mono text-sm text-fg-subtle">low resolution</span>
		{#if loading}
			<span
				class="ml-auto inline-flex items-center gap-1.5 font-mono text-sm text-fg-subtle"
				data-preview-loading
			>
				<Spinner size="sm" />
				Updating preview
			</span>
		{/if}
	</div>

	<div
		class="relative mx-auto overflow-hidden rounded-lg bg-scrim"
		style="width: min(100%, {shown * 280}px); aspect-ratio: {shown};"
		data-before-after
	>
		<div class="absolute inset-0 flex items-center justify-center overflow-hidden">
			{#if beforeKind === 'video'}
				<video
					src={beforeUrl}
					muted
					playsinline
					preload="metadata"
					class="h-full w-full object-contain"
					aria-label="Before"
				></video>
			{:else}
				<img src={beforeUrl} alt="Before" class="h-full w-full object-contain" />
			{/if}
		</div>

		<div
			class="checkerboard absolute inset-0 flex items-center justify-center overflow-hidden"
			style="clip-path: inset(0 0 0 {split}%);"
		>
			{#if afterUrl}
				<img src={afterUrl} alt="After" class="h-full w-full object-contain" onload={handleAfterLoad} />
			{:else if error}
				<p class="px-4 text-center text-sm text-fg-muted" role="alert">{error}</p>
			{/if}
		</div>

		<span
			class="pointer-events-none absolute inset-y-0 z-[3] w-0.5 -translate-x-1/2 bg-scrim-on"
			style="left: {split}%;"
		></span>
		<span
			class="pointer-events-none absolute left-2 top-2 z-[3] rounded bg-scrim/70 px-1.5 py-0.5 font-mono text-sm uppercase tracking-[0.07em] text-scrim-on backdrop-blur-sm"
		>
			Before
		</span>
		<span
			class="pointer-events-none absolute right-2 top-2 z-[3] rounded bg-scrim/70 px-1.5 py-0.5 font-mono text-sm uppercase tracking-[0.07em] text-scrim-on backdrop-blur-sm"
		>
			After
		</span>
		<input
			type="range"
			min="0"
			max="100"
			bind:value={split}
			aria-label="Before and after split"
			class="absolute inset-0 z-[4] m-0 h-full w-full cursor-ew-resize opacity-0"
		/>
	</div>
</section>

<style>
	.checkerboard {
		background-image:
			linear-gradient(45deg, rgb(var(--surface-3)) 25%, transparent 25%),
			linear-gradient(-45deg, rgb(var(--surface-3)) 25%, transparent 25%),
			linear-gradient(45deg, transparent 75%, rgb(var(--surface-3)) 75%),
			linear-gradient(-45deg, transparent 75%, rgb(var(--surface-3)) 75%);
		background-size: 16px 16px;
		background-position:
			0 0,
			0 8px,
			8px -8px,
			-8px 0;
		background-color: rgb(var(--surface-2));
	}
</style>
