<script lang="ts">
	import { goto } from '$app/navigation';
	import { Badge, Button } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import {
		formatEnergyKept,
		formatModelFileSizes,
		hasQualitySummary,
		isLoraFile,
		modelFileHref,
		modelFileTypeLabel,
		normalizeLabel,
		qualityNote,
		sizeReductionPercent,
		splitFileName,
		type ModelFileData
	} from '$lib/generation/artifacts/modelFile';

	let { artifact }: { artifact: { artifact_data: ModelFileData } } = $props();

	let data = $derived(artifact.artifact_data ?? {});
	let summary = $derived(data.summary ?? {});
	let lora = $derived(isLoraFile(data));
	let href = $derived(modelFileHref(data));
	let sizes = $derived(formatModelFileSizes(data.source_size_bytes, data.size_bytes));
	let reduction = $derived(sizeReductionPercent(data.source_size_bytes, data.size_bytes));
	let energy = $derived(formatEnergyKept(summary.energy_kept));
	let worstLayers = $derived(summary.worst_layers ?? []);
	let sources = $derived(summary.sources ?? []);
	let showQuality = $derived(hasQualitySummary(data.summary));
	let note = $derived(qualityNote(data.summary));
	let fileName = $derived(data.file_name || 'Model file');
	let nameParts = $derived(splitFileName(fileName));
	let showFormat = $derived(
		!!data.format && normalizeLabel(data.format ?? '') !== normalizeLabel(modelFileTypeLabel(data))
	);
	let worstOpen = $state(false);
</script>

<div class="space-y-3" data-model-file-card data-kind={lora ? 'lora' : 'model'}>
	<div class="flex flex-wrap items-center gap-2">
		<Icon name="box" className="h-4 w-4 shrink-0 text-fg-muted" />
		<Tooltip text={fileName} position="top" wrapperClass="flex min-w-0 flex-1 items-center">
			<span class="flex min-w-0 font-mono text-sm font-semibold text-fg" data-model-file-name>
				<span class="min-w-0 truncate">{nameParts.stem}</span><span class="shrink-0"
					>{nameParts.ext}</span
				>
			</span>
		</Tooltip>
		<Badge variant="neutral" size="md">{modelFileTypeLabel(data)}</Badge>
		{#if showFormat}
			<Badge variant="signal" size="md">{(data.format ?? '').replace(/_/g, ' ')}</Badge>
		{/if}
	</div>

	{#if sizes}
		<div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
			<span class="font-mono text-sm font-semibold tabular-nums text-fg" data-model-file-sizes
				>{sizes}</span
			>
			{#if reduction !== null && reduction > 0}
				<span class="font-mono text-xs tabular-nums text-success">-{reduction}%</span>
			{/if}
		</div>
	{/if}

	{#if showQuality}
		<dl class="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3" data-model-file-quality>
			{#if typeof summary.layers_quantized === 'number'}
				<div>
					<dt class="text-fg-subtle">Layers quantized</dt>
					<dd class="font-mono tabular-nums text-fg">{summary.layers_quantized}</dd>
				</div>
			{/if}
			{#if typeof summary.layers_kept === 'number'}
				<div>
					<dt class="text-fg-subtle">Layers kept</dt>
					<dd class="font-mono tabular-nums text-fg">{summary.layers_kept}</dd>
				</div>
			{/if}
			{#if typeof summary.layers_shape_kept === 'number'}
				<div>
					<dt class="text-fg-subtle">Kept for shape</dt>
					<dd class="font-mono tabular-nums text-fg" data-model-file-shape-kept>{summary.layers_shape_kept}</dd>
				</div>
			{/if}
			{#if lora && typeof summary.rank === 'number'}
				<div>
					<dt class="text-fg-subtle">Rank</dt>
					<dd class="font-mono tabular-nums text-fg" data-model-file-rank>{summary.rank}</dd>
				</div>
			{/if}
			{#if energy}
				<div>
					<dt class="text-fg-subtle">Energy kept</dt>
					<dd class="font-mono tabular-nums text-fg" data-model-file-energy>{energy}</dd>
				</div>
			{/if}
		</dl>
	{/if}

	{#if note}
		<p class="text-xs text-warning" data-model-file-quality-note>{note}</p>
	{/if}

	{#if worstLayers.length > 0}
		<div>
			<button
				type="button"
				class="flex min-h-8 items-center gap-1.5 text-xs font-medium text-fg-muted transition-colors hover:text-fg"
				aria-expanded={worstOpen}
				aria-controls="worst-layers-{data.index ?? 0}"
				onclick={() => (worstOpen = !worstOpen)}
			>
				<Icon
					name="chevron-down"
					className="h-3.5 w-3.5 transition-transform {worstOpen ? 'rotate-180' : ''}"
				/>
				Worst layers
				<span class="font-mono tabular-nums text-fg-subtle">{worstLayers.length}</span>
			</button>
			{#if worstOpen}
				<ul
					id="worst-layers-{data.index ?? 0}"
					class="mt-1 space-y-1 rounded bg-surface-1 px-2.5 py-2"
					data-model-file-worst
				>
					{#each worstLayers as layer, i (i)}
						<li class="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 text-xs">
							<span class="min-w-0 break-all font-mono text-fg-muted">{layer.key}</span>
							<span class="shrink-0 font-mono tabular-nums text-fg-subtle">
								{#if typeof layer.rel_l2 === 'number'}rel L2 {layer.rel_l2.toFixed(4)}{/if}
								{#if typeof layer.cosine === 'number'}
									{typeof layer.rel_l2 === 'number' ? ' · ' : ''}cos {layer.cosine.toFixed(4)}
								{/if}
							</span>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	{/if}

	{#if sources.length > 0}
		<div data-model-file-sources>
			<div class="mb-1 text-xs text-fg-subtle">Sources</div>
			<ul class="space-y-1">
				{#each sources as source, i (i)}
					<li class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
						<span class="min-w-0 break-all font-mono text-fg">{source.name}</span>
						{#if typeof source.strength === 'number'}
							<Badge variant="neutral" size="sm" class="font-mono tabular-nums"
								>{source.strength}x</Badge
							>
						{/if}
						{#if source.sha256}
							<Tooltip text={source.sha256} position="top">
								<span class="font-mono tabular-nums text-fg-subtle">{source.sha256.slice(0, 8)}</span>
							</Tooltip>
						{/if}
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if summary.licence_note}
		<p class="text-xs text-fg-muted" data-model-file-licence>{summary.licence_note}</p>
	{/if}

	{#if data.indexed === false && data.index_warning}
		<p class="text-xs text-warning" data-model-file-index-warning>
			Saved, but not added to the library: {data.index_warning}
		</p>
	{/if}

	{#if href}
		<div class="flex flex-wrap items-center gap-2 pt-1">
			<Button size="xs" variant="secondary" icon="box" onclick={() => goto(href)}>Open in Models</Button>
		</div>
	{/if}
</div>
