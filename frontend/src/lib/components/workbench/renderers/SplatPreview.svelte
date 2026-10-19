<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import { Alert, Spinner, EmptyState, IconButton } from '$lib/components/ui';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { logger } from '$lib/utils/logger';
	import {
		resolveSplatUrl,
		resolveSplatMetadata,
		resolveSplatFormat,
		type SplatFileLike
	} from './splatUrl';
	import { SPLAT_BACKGROUND_CSS, SplatLoadError, splatErrorMessage } from './splat/splatSupport';
	import type { SplatViewer, SplatViewerOptions } from './splat/splatViewer';

	export let file: SplatFileLike | null | undefined = null;

	type Status = 'empty' | 'loading' | 'ready' | 'error';

	const LOAD_TIMEOUT_MS = 60000;

	$: splatUrl = resolveSplatUrl(file);
	$: splatMeta = resolveSplatMetadata(file);
	$: splatFormat = resolveSplatFormat(file);
	$: splatCount = splatMeta.splatCount ?? loadedCount;
	$: hasStats = splatCount != null || !!splatMeta.filename || !!splatFormat;

	let status: Status = splatUrl ? 'loading' : 'empty';
	let errorMessage = '';
	let loadedCount: number | null = null;
	let containerEl: HTMLElement | null = null;
	let viewer: SplatViewer | null = null;
	let createViewer: ((options: SplatViewerOptions) => SplatViewer) | null = null;
	let moduleReady = false;
	let autoRotateOn = false;
	let activeUrl: string | null = null;
	let loadController: AbortController | null = null;
	let loadTimeoutId: ReturnType<typeof setTimeout> | null = null;

	function formatCount(n: number | null): string {
		return n == null ? '—' : n.toLocaleString();
	}

	function fail(message: string) {
		status = 'error';
		errorMessage = message;
	}

	onMount(async () => {
		if (!browser) return;
		try {
			createViewer = (await import('./splat/splatViewer')).createSplatViewer;
			moduleReady = true;
		} catch (err) {
			logger.error('[SplatPreview] failed to load the 3D viewer module', err);
			fail('The 3D viewer failed to load.');
		}
	});

	function clearLoadTimeout() {
		if (loadTimeoutId) {
			clearTimeout(loadTimeoutId);
			loadTimeoutId = null;
		}
	}

	function cancelLoad() {
		clearLoadTimeout();
		const controller = loadController;
		loadController = null;
		controller?.abort();
	}

	function startViewer() {
		if (!containerEl || !createViewer) return;
		try {
			viewer = createViewer({
				container: containerEl,
				onContextLost: () => {
					cancelLoad();
					fail('The graphics context was lost. Reload the page to view this scene.');
				}
			});
		} catch (err) {
			logger.error('[SplatPreview] could not start the 3D viewer', err);
			fail(splatErrorMessage(err));
		}
	}

	function syncLoad(url: string | null, format: ReturnType<typeof resolveSplatFormat>) {
		cancelLoad();
		activeUrl = url;
		loadedCount = null;
		if (!url) {
			viewer?.clear();
			status = 'empty';
			return;
		}
		status = 'loading';
		errorMessage = '';

		const controller = new AbortController();
		loadController = controller;
		let timedOut = false;
		loadTimeoutId = setTimeout(() => {
			timedOut = true;
			controller.abort();
		}, LOAD_TIMEOUT_MS);

		viewer!.load(url, format, controller.signal).then(
			(result) => {
				if (controller !== loadController) return;
				clearLoadTimeout();
				loadedCount = result.splatCount;
				status = 'ready';
			},
			(err) => {
				if (controller !== loadController) return;
				clearLoadTimeout();
				logger.error('[SplatPreview] failed to display the splat', err);
				fail(splatErrorMessage(timedOut ? new SplatLoadError('timeout') : err));
			}
		);
	}

	$: if (browser && moduleReady && containerEl && splatUrl && !viewer) startViewer();

	$: if (viewer && splatUrl !== activeUrl) syncLoad(splatUrl, splatFormat);

	$: if (!viewer && status !== 'error') status = splatUrl ? 'loading' : 'empty';

	function resetView() {
		viewer?.resetView();
	}

	function toggleAutoRotate() {
		autoRotateOn = !autoRotateOn;
		viewer?.setAutoRotate(autoRotateOn);
	}

	onDestroy(() => {
		cancelLoad();
		viewer?.dispose();
		viewer = null;
	});
</script>

<div
	class="relative w-full h-full overflow-hidden"
	style:background-color={status === 'loading' || status === 'ready' ? SPLAT_BACKGROUND_CSS : null}
>
	<div
		bind:this={containerEl}
		class="splat-viewer absolute inset-0"
		style="opacity: {status === 'ready' ? 1 : 0}; pointer-events: {status === 'ready' ? 'auto' : 'none'}"
	></div>

	{#if status === 'empty'}
		<div class="absolute inset-0 flex items-center justify-center">
			<EmptyState icon="splat" title="No scene" description="This generation has no splat to display." compact />
		</div>
	{:else if status === 'error'}
		<div class="absolute inset-0 flex items-center justify-center">
			<div class="max-w-sm px-6">
				<Alert variant="danger" icon title="Couldn't display this scene">
					{errorMessage}
				</Alert>
			</div>
		</div>
	{:else if status === 'loading'}
		<div class="absolute inset-0 flex flex-col items-center justify-center gap-3 z-10 pointer-events-none">
			<Spinner size="lg" />
			<span class="text-2xs font-mono uppercase tracking-[0.08em] text-white/70">Loading scene…</span>
		</div>
	{:else}
		<div class="absolute bottom-3 right-3 z-30 flex items-center gap-1 bg-black/50 backdrop-blur-sm rounded-lg p-1 shadow-lg">
			<Tooltip text={autoRotateOn ? 'Stop auto-rotate' : 'Auto-rotate'} position="top" delay={150}>
				<IconButton
					icon="globe"
					label="Toggle auto-rotate"
					size="sm"
					active={autoRotateOn}
					ariaPressed={autoRotateOn}
					onclick={toggleAutoRotate}
				/>
			</Tooltip>
			<Tooltip text="Reset view" position="top" delay={150}>
				<IconButton icon="refresh" label="Reset view" size="sm" onclick={resetView} />
			</Tooltip>
		</div>

		{#if hasStats}
			<div class="absolute bottom-3 left-3 z-10 flex flex-col items-start gap-1.5 pointer-events-none">
				{#if splatMeta.filename}
					<span class="bg-black/70 backdrop-blur-sm text-white px-3 py-1 rounded-full text-2xs font-mono shadow-lg">
						{splatMeta.filename}
					</span>
				{/if}
				<span class="bg-black/70 backdrop-blur-sm text-white px-3 py-1.5 rounded-full shadow-lg">
					<span class="font-mono text-2xs uppercase tracking-[0.06em] tabular-nums">
						{formatCount(splatCount)} splats{splatFormat ? ` · ${splatFormat}` : ''}
					</span>
				</span>
			</div>
		{/if}
	{/if}
</div>

<style>
	.splat-viewer {
		transition: opacity 150ms var(--ease-out-quart, ease-out);
	}
</style>
