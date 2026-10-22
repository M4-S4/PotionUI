<script lang="ts">
	import { Alert, Badge, Button, SegmentedControl, Spinner } from '$lib/components/ui';
	import { DetailSection } from '$lib/components/detail';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { isMobile } from '$lib/stores/viewport';
	import { toasts } from '$lib/stores/toast';
	import { formatBytes } from '$lib/utils/format';
	import type { IndexingLevel } from '$lib/services/api/models';
	import IndexingDisclosure from './IndexingDisclosure.svelte';
	import IndexingFact from './IndexingFact.svelte';
	import IndexingStatusPanel from './IndexingStatusPanel.svelte';
	import { indexingStatusStore } from './indexingStatus.svelte';
	import {
		comparedFilesText,
		digestBytesDone,
		digestBytesTotal,
		digestFilesTotal,
		digestPercent,
		digestPhase,
		formatEtaAbout,
		formatRate,
		indexingDigests,
		indexingIsVisible,
		indexingScanHasContent,
		levelDescription,
		levelEstimateLine,
		levelLabel,
		levelProfile,
		notIndexedTitle
	} from './indexingDisplay';

	const FALLBACK_LEVELS: IndexingLevel[] = ['low', 'normal', 'max'];

	const status = $derived(indexingStatusStore.status);
	const digests = $derived(indexingDigests(status));
	const phase = $derived(digestPhase(status));
	const busy = $derived(indexingStatusStore.busy);

	let requestedLevel = $state<IndexingLevel | null>(null);

	const level = $derived(requestedLevel ?? digests?.level ?? 'low');
	const profile = $derived(levelProfile(digests, level));
	const levelItems = $derived(
		(digests?.levels.length ? digests.levels.map((entry) => entry.id) : FALLBACK_LEVELS).map((id) => ({
			id,
			label: levelLabel(id)
		}))
	);

	const hasQueue = $derived(phase === 'running' || phase === 'paused' || phase === 'waiting');
	const showProgress = $derived(hasQueue || phase === 'up_to_date');
	const dimBar = $derived(phase === 'paused' || phase === 'waiting');
	const percent = $derived(digests ? digestPercent(digests) : 0);
	const reading = $derived(digests?.current ?? []);
	const remainingBytes = $derived.by(() => {
		if (!digests) return 0;
		if (digests.queued === 0) return digests.pending_bytes;
		const done = digests.current.reduce((sum, file) => sum + file.done, 0);
		return Math.max(0, digests.queued_bytes - done);
	});
	const estimate = $derived(levelEstimateLine(remainingBytes, profile, digests && digests.queued > 0 ? 'left' : 'rest'));
	const maxWhileBusy = $derived(hasQueue && !!digests?.busy && !!profile && !profile.pauses_for_generation);
	const failures = $derived(digests?.failed ?? []);
	const visible = $derived(!!digests || indexingIsVisible(status));

	const TITLES: Record<string, string> = {
		choice: 'Choose how to index your models',
		empty: 'No models to index yet',
		up_to_date: 'Every model is indexed',
		paused: 'Indexing is paused',
		waiting: 'Indexing is waiting for your generation to finish',
		running: 'Indexing in the background'
	};

	const title = $derived(phase === 'not_started' && digests ? notIndexedTitle(digests) : (TITLES[phase] ?? ''));
	const subtitle = $derived(
		phase === 'empty' || phase === 'none'
			? ''
			: hasQueue
				? 'Models are listed and ready to use while this runs.'
				: 'Models are listed and ready to use.'
	);

	const speedText = $derived(phase === 'running' ? formatRate(digests?.throughput_bps) : '0 MB/s');
	const etaText = $derived.by(() => {
		if (phase === 'paused') return 'Paused';
		if (phase === 'waiting') return 'Waiting';
		return formatEtaAbout(digests?.eta_seconds) || 'Calculating';
	});

	async function changeLevel(next: string) {
		if (busy || next === level) return;
		const id = next as IndexingLevel;
		requestedLevel = id;
		const ok = await indexingStatusStore.setLevel(id);
		requestedLevel = null;
		if (ok) toasts.success(`Resource use set to ${levelLabel(id)}`);
		else toasts.error(indexingStatusStore.actionError ?? 'The change could not be saved.');
	}

	async function togglePause() {
		const ok = phase === 'paused' ? await indexingStatusStore.resume() : await indexingStatusStore.pause();
		if (!ok) toasts.error(indexingStatusStore.actionError ?? 'The change could not be saved.');
	}

	async function indexAll() {
		const ok = await indexingStatusStore.indexAll();
		if (ok) toasts.success('Indexing started in the background');
		else toasts.error(indexingStatusStore.actionError ?? 'Indexing could not be started.');
	}
</script>

{#if visible}
	<DetailSection label="Model indexing">
		{#snippet headerExtra()}
			{#if phase === 'choice'}
				<Badge size="lg">Waiting for your choice</Badge>
			{:else if phase === 'not_started'}
				<Badge size="lg">Not started</Badge>
			{:else if phase === 'up_to_date'}
				<Badge size="lg" variant="success" dot>Up to date</Badge>
			{:else if phase === 'paused'}
				<Badge size="lg" variant="warning" dot>Paused</Badge>
			{:else if phase === 'waiting'}
				<Badge size="lg" variant="info" dot>Waiting for generation</Badge>
			{:else if phase === 'running'}
				<Badge size="lg" variant="signal" dot>Indexing</Badge>
			{/if}
		{/snippet}

		{#if indexingScanHasContent(status) && status}
			<div class="mb-4">
				<IndexingStatusPanel {status} />
			</div>
		{/if}

		{#if digests}
			<div class="flex flex-wrap items-center gap-3">
				<div class="min-w-0 flex-1 max-sm:basis-full">
					<div class="text-base font-medium text-fg">{title}</div>
					{#if subtitle}
						<p class="mt-0.5 text-sm text-fg-muted">{subtitle}</p>
					{/if}
				</div>
				{#if hasQueue}
					<Button
						variant="secondary"
						size="sm"
						icon={phase === 'paused' ? 'play' : 'pause'}
						class="max-sm:min-h-11 max-sm:w-full"
						disabled={busy}
						onclick={togglePause}
					>
						{phase === 'paused' ? 'Resume' : 'Pause'}
					</Button>
				{:else if phase === 'choice' || phase === 'not_started'}
					<div class="flex flex-wrap gap-2 max-sm:w-full max-sm:flex-col-reverse">
						<Button
							variant="secondary"
							size="sm"
							class="max-sm:min-h-11 max-sm:w-full"
							disabled={busy}
							onclick={() => indexingStatusStore.requestPick()}
						>
							Pick models
						</Button>
						<Button variant="primary" size="sm" class="max-sm:min-h-11 max-sm:w-full" disabled={busy} onclick={indexAll}>
							Index all in the background
						</Button>
					</div>
				{/if}
			</div>

			{#if showProgress}
				<div class="mt-4">
					<div
						class="h-1.5 overflow-hidden rounded-sm bg-surface-3"
						role="progressbar"
						aria-label="Indexing progress"
						aria-valuemin={0}
						aria-valuemax={100}
						aria-valuenow={percent}
					>
						<div class="h-full transition-all duration-300 {dimBar ? 'bg-fg-subtle' : 'bg-signal-solid'}" style="width: {percent}%"></div>
					</div>
					<div class="mt-2 flex flex-wrap justify-between gap-x-3 font-mono text-sm tabular-nums text-fg-muted">
						<span>{digests.ready} of {digestFilesTotal(digests)} indexed</span>
						<span>{formatBytes(digestBytesDone(digests), 1)} of {formatBytes(digestBytesTotal(digests), 1)}</span>
					</div>
				</div>
			{/if}

			{#if hasQueue}
				<div class="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
					<IndexingFact label="Speed" value={speedText} />
					<IndexingFact label="Time left" value={etaText} />
					<IndexingFact label="Workers" value={String(digests.workers)} wide />
				</div>

				{#if phase === 'running' && reading.length}
					<div class="mt-3 flex min-w-0 items-center gap-2 text-sm text-fg-muted">
						<Spinner size="sm" />
						<span>Reading</span>
						<Tooltip text={reading[0].filename} position="bottom" truncateOnly wrapperClass="min-w-0 flex">
							<span class="truncate font-mono text-fg">{reading[0].filename}</span>
						</Tooltip>
						<span class="whitespace-nowrap font-mono tabular-nums text-fg-subtle">
							{formatBytes(reading[0].done, 1)} of {formatBytes(reading[0].size, 1)}
						</span>
						{#if reading.length > 1}
							<span class="whitespace-nowrap text-fg-subtle">and {reading.length - 1} more</span>
						{/if}
					</div>
				{/if}
				{#if phase === 'running' && digests.duplicate_checks > 0}
					<p class="mt-2 text-sm text-fg-subtle">{comparedFilesText(digests.duplicate_checks)}</p>
				{/if}

				{#if phase === 'paused'}
					<Alert variant="neutral" density="compact" icon="pause" class="mt-3">
						Paused. Progress is kept, even if you restart PotionUI.
					</Alert>
				{:else if phase === 'waiting'}
					<Alert variant="info" density="compact" icon="info" class="mt-3">
						Paused while a generation is running. Indexing carries on by itself when it finishes.
					</Alert>
				{/if}
				{#if maxWhileBusy}
					<Alert variant="warning" density="compact" icon="info" class="mt-3">
						A generation is running and Max does not pause for it. Switch to Low or Normal to keep generations fast.
					</Alert>
				{/if}
			{:else if phase === 'choice' || phase === 'not_started'}
				<Alert variant="neutral" density="compact" icon="info" class="mt-4">
					Indexing lets PotionUI find duplicates, look models up on CivitAI and remember what each file is. Without it,
					everything still works, and a file is read the moment something needs it.
				</Alert>
			{/if}

			{#if failures.length}
				<div class="mt-4">
					<IndexingDisclosure
						count={failures.length}
						description={failures.length === 1 ? 'file could not be read' : 'files could not be read'}
						variant="warning"
						dot
					>
						{#each failures as failure (failure.path)}
							<li class="flex min-w-0 items-start gap-2">
								<Tooltip text={failure.path} position="bottom" truncateOnly wrapperClass="min-w-0 flex">
									<span class="inline-block max-w-[18rem] truncate align-bottom font-mono text-fg-muted">{failure.path}</span>
								</Tooltip>
								<span class="truncate text-danger">{failure.error}</span>
							</li>
						{/each}
					</IndexingDisclosure>
				</div>
			{/if}

			<div class="my-4 h-px bg-line"></div>

			<div class="mb-2 flex items-baseline justify-between gap-2">
				<span class="font-medium text-fg">Resource use</span>
				<span class="text-sm text-fg-subtle">Applies right away</span>
			</div>
			<SegmentedControl
				variant="toggle"
				ariaLabel="Resource use"
				items={levelItems}
				selected={level}
				fullWidth={$isMobile}
				onSelect={changeLevel}
			/>
			<div class="mt-3 grid gap-1 text-sm text-fg-muted">
				{#if profile}
					<span>{levelDescription(profile)}</span>
				{/if}
				{#if estimate}
					<span class="font-mono text-fg">{estimate}</span>
				{/if}
			</div>
		{/if}
	</DetailSection>
{/if}
