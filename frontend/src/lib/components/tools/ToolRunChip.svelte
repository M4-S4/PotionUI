<script lang="ts">
	import { page } from '$app/stores';
	import { IconButton } from '$lib/components/ui';
	import { mediaToolRuns } from '$lib/media-tools/runs';
	import { activeRuns, isRunChipRoute } from '$lib/tools/activeRuns';

	let runs = $derived(activeRuns($mediaToolRuns));
	let visible = $derived(runs.length > 0 && isRunChipRoute($page.url.pathname));

	function percent(progress: number): number {
		return Math.round(Math.min(1, Math.max(0, progress)) * 100);
	}
</script>

{#if visible}
	<div
		class="fixed bottom-[calc(var(--tab-bar-height)_+_0.75rem)] left-3 z-toast flex max-h-[50vh] w-64 flex-col gap-2 overflow-y-auto md:bottom-4 md:left-[4.25rem]"
		data-tool-run-chips
	>
		{#each runs as run (run.runId)}
			{@const value = percent(run.progress)}
			<div
				class="grid grid-cols-[1fr_auto] items-center gap-x-2 gap-y-1.5 rounded-lg border border-line-strong bg-surface-2 p-2.5 shadow-floating"
				data-tool-run-chip
			>
				<span class="min-w-0 truncate text-sm text-fg" data-tool-run-chip-label>{run.label}</span>
				<span class="flex items-center gap-1.5">
					<span
						class="font-mono text-sm tabular-nums text-fg-muted"
						role="progressbar"
						aria-label="{run.label} progress"
						aria-valuemin={0}
						aria-valuemax={100}
						aria-valuenow={value}
						data-tool-run-chip-percent>{value}%</span
					>
					<IconButton
						icon="close"
						label={run.cancelling ? 'Cancelling' : `Cancel ${run.label}`}
						size="xs"
						disabled={run.cancelling}
						onclick={() => mediaToolRuns.cancelRun(run.runId)}
					/>
				</span>
				<div class="relative col-span-2 h-1 overflow-hidden rounded-sm bg-surface-3">
					<div
						class="absolute inset-y-0 left-0 bg-signal transition-[width] duration-200 ease-linear"
						style="width: {value}%"
					></div>
				</div>
			</div>
		{/each}
	</div>
{/if}
