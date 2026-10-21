<script lang="ts">
	import { sessionDrafts } from '$lib/stores/tabDraftSync';
	import { logger } from '$lib/utils/logger';
	import { onMount, onDestroy } from 'svelte';
	import { storage } from '$lib/utils/storage';
	import { api } from '$lib/services/api/index';
	import { tabsStore } from '$lib/stores/tabs';
	import type { PresetModeVariant } from '$lib/types/api';
	import { sortVariants } from '$lib/utils/variants';
	import { createSessionController } from '$lib/session/sessionController';
	import { createSessionDialogs } from '$lib/session/sessionDialogs';
	import { loadFieldLabels } from '$lib/session/sessionFieldLabels';
	import type { FieldLabels } from '$lib/session/sessionDrawerModel';
	import { toasts } from '$lib/stores/toast';
	import SessionControl from '$lib/components/session/SessionControl.svelte';
	import SessionHost from '$lib/components/session/SessionHost.svelte';

	// The session half of the old PresetSessionBar, re-homed as the
	// compact pill in the tabs row (top-right). Preset selection/mode/variant
	// state stays in PresetHeader - this component only knows presetId,
	// currentMode and tabId, all sourced from the active tab. The session
	// workflow itself lives in the shared controller
	// (lib/session/sessionController.ts), driven identically by the console
	// bar's generation-panel/SessionCluster.svelte.
	export let presetId: string | null = null;
	export let currentMode: string | null = null;
	export let tabId: string;
	// Live preset's version, for the saved-session drift warning and the
	// stamp written into new saves - the full preset list itself belongs to
	// PresetHeader, not here.
	export let presetVersion: string | undefined = undefined;
	// Only used to resolve a restored session's variant against what the
	// current mode actually offers (falls back to the mode's default variant
	// if the saved one no longer exists) - the variant *selector* itself is
	// PresetHeader's.
	export let availableModes: Array<{ id: string; variants?: PresetModeVariant[] }> = [];

	const controller = createSessionController({ api, tabs: tabsStore, storage, toasts, logger, drafts: sessionDrafts });
	const session = controller.state;
	const dialogs = createSessionDialogs(controller);

	let isClient = false;
	let fieldLabels: FieldLabels = {};

	$: refreshFieldLabels(presetId, currentMode);

	async function refreshFieldLabels(preset: string | null, mode: string | null) {
		const labels = await loadFieldLabels(preset, mode);
		if (preset === presetId && mode === currentMode) fieldLabels = labels;
	}

	$: currentModeVariants = sortVariants(availableModes.find((mode) => mode.id === currentMode)?.variants);

	$: controller.setContext({
		tabId,
		presetId,
		currentMode,
		presetVersion,
		modeVariants: currentModeVariants
	});

	onMount(() => {
		isClient = true;
		controller.start();
	});

	onDestroy(() => {
		controller.destroy();
	});
</script>

{#if isClient}
	<SessionControl {fieldLabels} compact session={$session} actions={dialogs.actions} />
{/if}

<SessionHost {controller} {dialogs} />
