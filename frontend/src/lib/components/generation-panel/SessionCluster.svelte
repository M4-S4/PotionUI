<script lang="ts">
	import { sessionDrafts } from '$lib/stores/tabDraftSync';
	import { logger } from '$lib/utils/logger';
	import { onMount, onDestroy } from 'svelte';
	import { storage } from '$lib/utils/storage';
	import { api } from '$lib/services/api/index';
	import { tabsStore } from '$lib/stores/tabs';
	import { shortcutLabels } from '$lib/stores/keybindings';
	import type { PresetModeVariant } from '$lib/types/api';
	import { sortVariants } from '$lib/utils/variants';
	import { createSessionController } from '$lib/session/sessionController';
	import { createSessionDialogs } from '$lib/session/sessionDialogs';
	import { loadFieldLabels } from '$lib/session/sessionFieldLabels';
	import type { FieldLabels } from '$lib/session/sessionDrawerModel';
	import { activeWorkspaceSaveRequest, settleWorkspaceSaveRequest } from '$lib/stores/workspaceSaveRequest';
	import { activeWorkspaceDirtyQuery, answerWorkspaceDirtyQuery } from '$lib/stores/workspaceDirtyQuery';
	import { toasts } from '$lib/stores/toast';
	import { timeAgo } from '$lib/utils/relativeTime';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import SessionDrawer from '$lib/components/session/SessionDrawer.svelte';
	import { sideDrawer } from '$lib/stores/sideDrawer';
	import SessionHost from '$lib/components/session/SessionHost.svelte';

	export let presetId: string | null = null;
	export let currentMode: string | null = null;
	export let tabId: string;
	export let presetVersion: string | undefined = undefined;
	export let availableModes: Array<{ id: string; variants?: PresetModeVariant[] }> = [];

	const controller = createSessionController({ api, tabs: tabsStore, storage, toasts, logger, drafts: sessionDrafts });
	const session = controller.state;
	const dialogs = createSessionDialogs(controller);

	$: currentModeVariants = sortVariants(availableModes.find((mode) => mode.id === currentMode)?.variants);

	$: controller.setContext({
		tabId,
		presetId,
		currentMode,
		presetVersion,
		modeVariants: currentModeVariants
	});

	let open = false;
	$: if ($sideDrawer === 'formulas') open = false;
	let fieldLabels: FieldLabels = {};

	$: refreshFieldLabels(presetId, currentMode);

	async function refreshFieldLabels(preset: string | null, mode: string | null) {
		const labels = await loadFieldLabels(preset, mode);
		if (preset === presetId && mode === currentMode) fieldLabels = labels;
	}

	// "New workspace" (TabBar) asks whichever tab owns this mounted instance to
	// run its real save action before wiping. See stores/workspaceSaveRequest.ts.
	$: if ($activeWorkspaceSaveRequest && $activeWorkspaceSaveRequest.tabId === tabId) {
		void handleWorkspaceSaveRequest($activeWorkspaceSaveRequest.id);
	}

	async function handleWorkspaceSaveRequest(requestId: number) {
		if (!$session.sessionControlsEnabled || !$session.currentSession || !$session.hasUnsavedChanges) {
			// Nothing saveable on this tab right now (no session, or already
			// clean) - nothing to do, so this is a success from the caller's
			// point of view.
			settleWorkspaceSaveRequest(requestId, true);
			return;
		}
		await dialogs.quickSave();
		settleWorkspaceSaveRequest(requestId, !$session.error);
	}

	// "New workspace" also needs to know, at the moment of the click, whether
	// THIS tab (the only one with a live `hasUnsavedChanges`) is currently
	// dirty. A one-shot answer to a one-shot query, unlike the sessionDirty
	// push this replaced: continuously writing a dirty flag onto the shared
	// tabsStore on every `hasUnsavedChanges` transition churned the tabs array
	// during normal typing and intermittently stole focus from the segment
	// editor mid-keystroke (regressed session-tab-switch-preserves-draft.spec.ts).
	// A query answered only when asked has no such steady-state cost.
	$: if ($activeWorkspaceDirtyQuery && $activeWorkspaceDirtyQuery.tabId === tabId) {
		answerWorkspaceDirtyQuery($activeWorkspaceDirtyQuery.id, $session.hasUnsavedChanges);
	}

	$: saveCellText = !$session.sessionControlsEnabled
		? 'Unavailable'
		: !$session.currentSession
			? 'Save as new'
			: $session.isQuickSaving
				? 'Saving…'
				: $session.hasUnsavedChanges
					? $session.autoSaveEnabled
						? `Unsaved · ${$session.autoSaveInterval / 1000}s`
						: 'Unsaved changes'
					: `Saved ${timeAgo($session.lastSavedTime?.toISOString())}`;
	$: saveCellClass = !$session.sessionControlsEnabled
		? 'text-fg-disabled'
		: !$session.currentSession
			? 'text-signal'
			: $session.hasUnsavedChanges || $session.isQuickSaving
				? 'text-warning'
				: 'text-fg-subtle';
	$: saveCellAriaLabel = !$session.sessionControlsEnabled
		? 'Session save unavailable'
		: !$session.currentSession
			? 'Save as a new session'
			: $session.isQuickSaving
				? 'Saving session'
				: $session.hasUnsavedChanges
					? 'Save session'
					: 'Session saved';

	function toggleOpen() {
		if (!$session.sessionControlsEnabled) return;
		open = !open;
		if (open) sideDrawer.open('sessions');
	}

	function handleSaveCellClick() {
		if (!$session.sessionControlsEnabled) return;
		if (!$session.currentSession) {
			dialogs.openSaveAs();
		} else {
			dialogs.quickSave();
		}
	}

	onMount(() => {
		controller.start();
	});

	onDestroy(() => {
		controller.destroy();
	});
</script>

<div class="session-control">
	<!-- Ported literally from generation-panel-concept.html's `.session-control`
	     > `.session-button` (lines 186-187, 443-450): the name+chevron line
	     AND the passive status line (dirty-dot + save status text) both live
	     inside this one button, which opens the picker — aria-label stays the
	     constant "Session" (real contract: button[aria-label="Session"] in
	     session-edit-survives-tab-switch.spec.ts / sessionClusterTabSwitchClobber.test.ts). -->
	<button
		type="button"
		class="session-button"
		disabled={!$session.sessionControlsEnabled}
		aria-label="Session"
		aria-haspopup="dialog"
		aria-expanded={open}
		on:click={toggleOpen}
	>
		<span class="cell-copy">
			<span class="session-name-line">
				<span class="session-name">{$session.currentSession ? $session.currentSession.name : 'None'}</span>
				<span class="session-chevron"><svg class="icon"><use href="#i-chevron" /></svg></span>
			</span>
			<span class="session-state">
				<span class="dirty-dot {$session.hasUnsavedChanges ? '' : 'is-saved'}" aria-hidden="true"></span>
				<span class={saveCellClass}>{saveCellText}</span>
			</span>
		</span>
	</button>

	<!-- The mock's own icon-only `.session-save-button` (32px) — the actual
	     save action. Its aria-label is the dynamic one ("Save session" /
	     "Session saved" / "Save as a new session" / "Session save
	     unavailable"): real contract (session-edit-survives-tab-switch.spec.ts,
	     button[aria-label="Save session"] etc.) matches on the attribute, not
	     on visible text, so an icon-only button satisfies it unchanged. -->
	<Tooltip text={saveCellAriaLabel} kbd={$shortcutLabels['save_session']} position="top" delay={150}>
		<button
			type="button"
			class="session-save-button"
			disabled={!$session.sessionControlsEnabled}
			aria-label={saveCellAriaLabel}
			on:click={handleSaveCellClick}
		>
			<svg class="icon"><use href="#i-save" /></svg>
		</button>
	</Tooltip>

	{#if open}
		<SessionDrawer {fieldLabels} session={$session} actions={dialogs.actions} onClose={() => (open = false)} />
	{/if}
</div>

<SessionHost {controller} {dialogs} />
