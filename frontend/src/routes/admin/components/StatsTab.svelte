<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/services/api/index';
	import type {
		StatsOverview,
		StatsTimeseries,
		StatsDurations,
		StatsStorage,
		StatsBreakdown,
		DurationBucket,
		PresetTimingItem,
		PresetResourcesItem
	} from '$lib/services/api/stats';
	import CloudSpendSection from './CloudSpendSection.svelte';
	import type { CloudSpend } from '$lib/utils/cloudCost';
	import { Button, Input, Spinner, EmptyState, Alert } from '$lib/components/ui';
	import { DetailBody, DetailSection } from '$lib/components/detail';
	import LibraryShell from '$lib/components/library/LibraryShell.svelte';
	import type { LibrarySectionMeta } from '$lib/components/library/librarySection';
	import {
		StatTile,
		ChartCard,
		TimeSeriesChart,
		RankBarChart,
		HistogramChart,
		RowLimitSelect
	} from '$lib/components/charts';
	import { formatBytes, formatDuration, formatCount } from '$lib/utils/format';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { statsRowLimits } from '$lib/stores/statsRowLimits';
	import { BREAKDOWNS, BREAKDOWN_BY_SECTION, type BreakdownConfig, type BreakdownSection } from './statsBreakdowns';

	function formatMb(mb: number | null): string {
		return mb === null ? '—' : formatBytes(mb * 1024 * 1024);
	}

	let dateFrom = $state('');
	let dateTo = $state('');

	let loading = $state(true);
	let error = $state<string | null>(null);

	let overview = $state<StatsOverview | null>(null);
	let timeseries = $state<StatsTimeseries | null>(null);
	let durations = $state<StatsDurations | null>(null);
	let storage = $state<StatsStorage | null>(null);
	let breakdowns = $state<Record<BreakdownSection, StatsBreakdown | null>>(
		Object.fromEntries(BREAKDOWNS.map((b) => [b.section, null])) as Record<BreakdownSection, StatsBreakdown | null>
	);
	let presetTiming = $state<PresetTimingItem[] | null>(null);
	let presetResources = $state<PresetResourcesItem[] | null>(null);
	let spend = $state<CloudSpend | null>(null);

	function currentRange() {
		return {
			from: dateFrom || undefined,
			to: dateTo || undefined
		};
	}

	async function load() {
		loading = true;
		error = null;
		const range = currentRange();
		const limits = $statsRowLimits;
		try {
			const [
				overviewRes,
				timeseriesRes,
				durationsRes,
				storageRes,
				breakdownRes,
				presetTimingRes,
				presetResourcesRes,
				spendRes
			] = await Promise.all([
				api.getStatsOverview(range),
				api.getStatsTimeseries('count', 'day', range),
				api.getStatsDurations(range),
				api.getStatsStorage(range, 'day', limits.storage),
				Promise.all(BREAKDOWNS.map((b) => api.getStatsBreakdown(b.dimension, limits[b.section], range))),
				api.getStatsPresetTiming(limits.presetTiming),
				api.getStatsPresetResources(limits.presetResources),
				api.getStatsSpend(range).catch(() => null)
			]);

			overview = overviewRes.success ? overviewRes.data ?? null : null;
			timeseries = timeseriesRes.success ? timeseriesRes.data ?? null : null;
			durations = durationsRes.success ? durationsRes.data ?? null : null;
			storage = storageRes.success ? storageRes.data ?? null : null;
			BREAKDOWNS.forEach((b, index) => {
				const res = breakdownRes[index];
				breakdowns[b.section] = res.success ? (res.data ?? null) : null;
			});
			presetTiming = presetTimingRes.success ? presetTimingRes.data?.items ?? null : null;
			presetResources = presetResourcesRes.success ? presetResourcesRes.data?.items ?? null : null;

			spend = spendRes?.success ? spendRes.data ?? null : null;

			if (!overviewRes.success) {
				error = overviewRes.message || 'Failed to load statistics';
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load statistics';
		} finally {
			loading = false;
		}
	}

	async function reloadBreakdown(b: BreakdownConfig, limit: number) {
		const res = await api.getStatsBreakdown(b.dimension, limit, currentRange());
		breakdowns[b.section] = res.success ? (res.data ?? null) : null;
	}

	async function reloadStorage(limit: number) {
		const res = await api.getStatsStorage(currentRange(), 'day', limit);
		storage = res.success ? (res.data ?? null) : null;
	}

	async function reloadPresetTiming(limit: number) {
		const res = await api.getStatsPresetTiming(limit);
		presetTiming = res.success ? (res.data?.items ?? null) : null;
	}

	async function reloadPresetResources(limit: number) {
		const res = await api.getStatsPresetResources(limit);
		presetResources = res.success ? (res.data?.items ?? null) : null;
	}

	onMount(load);

	function durationBucketLabel(b: DurationBucket): string {
		return b.upper_s === null ? `${b.lower_s}s+` : `${b.lower_s}-${b.upper_s}s`;
	}

	let durationHistogramData = $derived(
		(durations?.buckets ?? []).map((b) => ({ label: durationBucketLabel(b), count: b.count }))
	);

	function bucketLabelForMs(ms: number | null): string | null {
		if (ms === null || !durations) return null;
		const s = ms / 1000;
		const match = durations.buckets.find((b) =>
			b.upper_s === null ? s >= b.lower_s : s >= b.lower_s && s < b.upper_s
		);
		return match ? durationBucketLabel(match) : null;
	}

	let durationMarkers = $derived(
		[
			{ label: 'p50', at: bucketLabelForMs(durations?.p50_ms ?? null) },
			{ label: 'p95', at: bucketLabelForMs(durations?.p95_ms ?? null) }
		].filter((m): m is { label: string; at: string } => m.at !== null)
	);

	let durationSubtitle = $derived(
		[
			`p50 ${durations?.p50_ms != null ? formatDuration(durations.p50_ms) : '—'}`,
			`p95 ${durations?.p95_ms != null ? formatDuration(durations.p95_ms) : '—'}`,
			...(durations?.unknown ? [`${formatCount(durations.unknown)} with unknown duration`] : [])
		].join(' · ')
	);

	let storageMaxTotal = $derived(
		Math.max(
			1,
			...((storage?.over_time ?? []).map((d) => d.image_bytes + d.video_bytes) ?? [0])
		)
	);

	const PARAMETER_BREAKDOWNS = BREAKDOWNS.filter((b) => b.group === 'parameters');

	const STATS_SECTIONS: readonly LibrarySectionMeta<'overview'>[] = [{ id: 'overview', label: 'Overview', icon: 'gauge' }];

	function clearRange() {
		dateFrom = '';
		dateTo = '';
		load();
	}
</script>

{#snippet breakdownCard(b: BreakdownConfig)}
	{@const data = breakdowns[b.section]}
	{#if data}
		<ChartCard
			title={b.title}
			subtitle={b.subtitle}
			tableData={data.items}
			tableColumns={[
				{ key: 'label', label: b.label },
				...(b.groupLabel ? [{ key: 'group', label: b.groupLabel }] : []),
				{ key: 'count', label: 'Count', align: 'right', format: (v) => formatCount(Number(v)) }
			]}
		>
			{#snippet headerExtra()}
				<RowLimitSelect
					section={b.section}
					value={$statsRowLimits[b.section]}
					onchange={(limit) => reloadBreakdown(b, limit)}
				/>
			{/snippet}
			<RankBarChart data={data.items} colorBy={b.colorBy} valueFormat={b.valueFormat} />
		</ChartCard>
	{/if}
{/snippet}

<LibraryShell
	title="Stats"
	persistKey="admin-stats-library"
	heightClass="h-full"
	sections={STATS_SECTIONS}
	section="overview"
	onSelectSection={() => {}}
>
	{#snippet toolbar()}
		<div class="flex items-end gap-2 flex-wrap">
			<div>
				<span class="block text-xs font-medium text-fg-muted mb-1">From</span>
				<Input type="date" bind:value={dateFrom} class="w-40" aria-label="From date" />
			</div>
			<div>
				<span class="block text-xs font-medium text-fg-muted mb-1">To</span>
				<Input type="date" bind:value={dateTo} class="w-40" aria-label="To date" />
			</div>
			<Button variant="primary" size="sm" onclick={load} disabled={loading}>Apply</Button>
			{#if dateFrom || dateTo}
				<Button variant="ghost" size="sm" icon="close" onclick={clearRange}>Clear</Button>
			{/if}
		</div>
	{/snippet}

	{#snippet primary()}
		{#if loading}<Spinner size="sm" />{/if}
	{/snippet}

<DetailBody>
<div class="space-y-4">
	{#if error}
		<Alert variant="danger" density="compact" live="polite">{error}</Alert>
	{/if}

	{#if overview}
		<DetailSection label="Overview">
			<div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
				<StatTile label="Generations" value={formatCount(overview.total_generations)} hero />
				<StatTile
					label="Completed"
					value={formatCount(overview.completed)}
					hint="{overview.total_generations
						? Math.round((overview.completed / overview.total_generations) * 100)
						: 0}% of total"
				/>
				<StatTile
					label="Failed"
					value={formatCount(overview.failed)}
					hint="{overview.total_generations
						? Math.round((overview.failed / overview.total_generations) * 100)
						: 0}% of total"
				/>
				<StatTile label="Distinct models" value={formatCount(overview.distinct_models)} />
				<StatTile
					label="Outputs"
					value={formatCount(overview.total_outputs)}
					hint={formatBytes(overview.total_bytes)}
				/>
				<StatTile
					label="Median duration"
					value={overview.median_duration_ms !== null
						? formatDuration(overview.median_duration_ms)
						: '—'}
					hint={overview.p95_duration_ms !== null
						? `p95 ${formatDuration(overview.p95_duration_ms)}`
						: undefined}
				/>
			</div>
		</DetailSection>
	{/if}

	<CloudSpendSection {spend} />

	<DetailSection label="Activity">
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		{#if timeseries}
			<ChartCard
				title="Generations over time"
				subtitle="Daily count"
				tableData={timeseries.points}
				tableColumns={[
					{ key: 'bucket', label: 'Date' },
					{ key: 'value', label: 'Count', align: 'right', format: (v) => formatCount(Number(v)) }
				]}
			>
				<TimeSeriesChart
					data={timeseries.points}
					label="Generations"
					valueFormat={(n) => formatCount(Math.round(n))}
				/>
			</ChartCard>
		{/if}

		{#if durations}
			<ChartCard
				title="Duration distribution"
				subtitle={durationSubtitle}
				tableData={durationHistogramData}
				tableColumns={[
					{ key: 'label', label: 'Bucket' },
					{ key: 'count', label: 'Count', align: 'right', format: (v) => formatCount(Number(v)) }
				]}
			>
				<HistogramChart
					data={durationHistogramData}
					markers={durationMarkers}
					valueFormat={(n) => formatCount(Math.round(n))}
				/>
			</ChartCard>
		{/if}

		{@render breakdownCard(BREAKDOWN_BY_SECTION.presets)}
		{@render breakdownCard(BREAKDOWN_BY_SECTION.models)}

		{#if storage}
			<ChartCard
				title="Storage over time"
				subtitle="Image vs. video bytes — the list the row limit controls"
				tableData={storage.over_time}
				tableColumns={[
					{ key: 'bucket', label: 'Date' },
					{
						key: 'image_bytes',
						label: 'Image',
						align: 'right',
						format: (v) => formatBytes(Number(v))
					},
					{
						key: 'video_bytes',
						label: 'Video',
						align: 'right',
						format: (v) => formatBytes(Number(v))
					}
				]}
			>
				{#snippet headerExtra()}
					<RowLimitSelect
						section="storage"
						value={$statsRowLimits.storage}
						onchange={(limit) => reloadStorage(limit)}
					/>
				{/snippet}
				<div class="flex flex-col gap-1">
					<div class="mb-2 flex items-center gap-3 text-xs text-fg-muted">
						<span class="flex items-center gap-1.5">
							<span class="inline-block h-2 w-2 rounded-full" style="background: rgb(var(--viz-1));"
							></span>
							Image
						</span>
						<span class="flex items-center gap-1.5">
							<span class="inline-block h-2 w-2 rounded-full" style="background: rgb(var(--viz-2));"
							></span>
							Video
						</span>
					</div>
					{#each storage.over_time as bucket (bucket.bucket)}
						{@const total = bucket.image_bytes + bucket.video_bytes}
						<div class="flex items-center gap-2">
							<div class="w-20 shrink-0 truncate text-xs text-fg-muted">{bucket.bucket}</div>
							<div class="relative h-4 flex-1 min-w-0 flex overflow-hidden rounded">
								<div
									style="width: {(bucket.image_bytes / storageMaxTotal) * 100}%; background: rgb(var(--viz-1));"
								></div>
								<div
									style="width: {(bucket.video_bytes / storageMaxTotal) * 100}%; background: rgb(var(--viz-2));"
								></div>
							</div>
							<div class="w-16 shrink-0 text-right font-mono text-xs tabular-nums text-fg">
								{formatBytes(total)}
							</div>
						</div>
					{/each}
				</div>
			</ChartCard>
		{/if}

		{@render breakdownCard(BREAKDOWN_BY_SECTION.resolutions)}
	</div>
	</DetailSection>

	<DetailSection label="Generation parameters">
	<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
		{#each PARAMETER_BREAKDOWNS as b (b.section)}
			{@render breakdownCard(b)}
		{/each}
	</div>
	</DetailSection>

	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<DetailSection label="Cold vs. warm start">
			{#snippet headerExtra()}
				<RowLimitSelect
					section="presetTiming"
					value={$statsRowLimits.presetTiming}
					onchange={(limit) => reloadPresetTiming(limit)}
				/>
			{/snippet}
			<p class="mb-3 text-xs text-fg-muted">
				Cold = at least one model had to load from disk. Warm = every model came from cache. Durable —
				keeps reporting after the generation is deleted.
			</p>
			{#if presetTiming && presetTiming.length > 0}
				<div class="overflow-x-auto">
					<table class="w-full text-xs">
						<thead>
							<tr class="border-b border-line">
								<th class="py-1.5 px-2 text-left font-medium text-fg-subtle">Preset</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Runs</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Cold</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Warm</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Avg cold</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Avg warm</th>
							</tr>
						</thead>
						<tbody>
							{#each presetTiming as item (item.preset_id)}
								<tr class="border-b border-line/50 last:border-0">
									<td class="py-1.5 px-2 text-fg max-w-40">
										<Tooltip text={item.preset_id} wrapperClass="block">
											<span class="block truncate">{item.preset_name}</span>
										</Tooltip>
									</td>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg"
										>{formatCount(item.total_runs)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg"
										>{formatCount(item.cold_runs)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg"
										>{formatCount(item.warm_runs)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg-muted">
										{item.avg_cold_duration_ms !== null
											? formatDuration(item.avg_cold_duration_ms)
											: '—'}
									</td>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg-muted">
										{item.avg_warm_duration_ms !== null
											? formatDuration(item.avg_warm_duration_ms)
											: '—'}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<EmptyState
					icon="hourglass"
					title="No stats yet"
					description="This fills in as generations complete."
					compact
				/>
			{/if}
		</DetailSection>

		<DetailSection label="Resources per preset">
			{#snippet headerExtra()}
				<RowLimitSelect
					section="presetResources"
					value={$statsRowLimits.presetResources}
					onchange={(limit) => reloadPresetResources(limit)}
				/>
			{/snippet}
			<p class="mb-3 text-xs text-fg-muted">Peak and average VRAM/RAM/CPU. Durable — keeps reporting after the generation is deleted.</p>
			{#if presetResources && presetResources.length > 0}
				<div class="overflow-x-auto">
					<table class="w-full text-xs">
						<thead>
							<tr class="border-b border-line">
								<th class="py-1.5 px-2 text-left font-medium text-fg-subtle">Preset</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Peak VRAM</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Avg VRAM</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Peak RAM</th>
								<th class="py-1.5 px-2 text-right font-medium text-fg-subtle">Avg CPU</th>
							</tr>
						</thead>
						<tbody>
							{#each presetResources as item (item.preset_id)}
								<tr class="border-b border-line/50 last:border-0">
									<td class="py-1.5 px-2 text-fg max-w-40">
										<Tooltip text={item.preset_id} wrapperClass="block">
											<span class="block truncate">{item.preset_name}</span>
										</Tooltip>
									</td>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg"
										>{formatMb(item.peak_vram_mb)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg-muted"
										>{formatMb(item.avg_vram_mb)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg"
										>{formatMb(item.peak_ram_mb)}</td
									>
									<td class="py-1.5 px-2 text-right font-mono tabular-nums text-fg-muted">
										{item.avg_cpu_percent !== null ? `${Math.round(item.avg_cpu_percent)}%` : '—'}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<EmptyState
					icon="cpu"
					title="No stats yet"
					description="This fills in as generations complete."
					compact
				/>
			{/if}
		</DetailSection>
	</div>
</div>
</DetailBody>
</LibraryShell>
