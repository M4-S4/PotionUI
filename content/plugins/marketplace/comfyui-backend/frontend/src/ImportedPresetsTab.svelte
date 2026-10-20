<script>
	// Registered as this plugin's `admin_tabs[]` "Imported presets" tab (see
	// manifest.yml) - mounted with `{ pluginId, plugin }` props. Lists every
	// preset under content/presets/local this plugin's importer created (see
	// backend/api.py's GET /presets/imported) and offers row actions: reload
	// (re-emit from the stored source), edit (reopens the wizard, prefilled),
	// delete, and Open in Presets.
	//
	// Edit hands off to the sibling "Import workflow" tab - a SEPARATELY
	// mounted plugin dist bundle, so there is no shared module state between
	// them (esbuild inlines a shared .js import into each bundle
	// independently). `sessionStorage` plus a `window` CustomEvent is this
	// plugin's own small cross-tab bridge: sessionStorage carries which
	// preset to edit (just the id - the wizard re-fetches everything else
	// fresh), the event asks the host (PluginsTab.svelte's generic
	// `potionui:switch-plugin-tab` listener) to switch to that tab.
	import { onMount } from 'svelte';
	import HostUi from '../../../../sdk/HostUi.svelte';
	import { call, messageOf } from './api.js';
	import { deletePreset } from './deletePreset.js';

	let { pluginId = 'comfyui-backend', plugin = null } = $props();

	const API_BASE = `/api/plugins/${pluginId}`;
	const EDIT_STORAGE_KEY = 'comfyui-import-edit-preset-id';

	let loading = $state(true);
	let loadError = $state('');
	let presets = $state([]);

	let reloadingId = $state(null);
	let reloadResultById = $state({});
	let deletingId = $state(null);

	async function load() {
		loading = true;
		loadError = '';
		try {
			const { ok, status, payload } = await call(`${API_BASE}/presets/imported`);
			if (!ok) {
				loadError = messageOf(payload, `Failed to load (${status})`);
				return;
			}
			presets = payload?.presets ?? [];
		} catch (e) {
			loadError = 'Could not reach the server.';
		} finally {
			loading = false;
		}
	}

	onMount(load);

	function requirementsChip(summary) {
		if (!summary) return { text: '—', variant: 'neutral' };
		const total = summary.ok + summary.missing + summary.unknown + summary.optional_missing;
		if (summary.missing > 0) return { text: `${summary.ok}/${total}`, variant: 'danger' };
		if (summary.unknown > 0 || summary.optional_missing > 0) return { text: `${summary.ok}/${total}`, variant: 'warning' };
		return { text: `${summary.ok}/${total}`, variant: 'success' };
	}

	function relativeTime(unixSeconds) {
		const deltaMs = Date.now() - unixSeconds * 1000;
		const minute = 60_000;
		const hour = 60 * minute;
		const day = 24 * hour;
		if (deltaMs < minute) return 'just now';
		if (deltaMs < hour) return `${Math.floor(deltaMs / minute)} min ago`;
		if (deltaMs < day) return `${Math.floor(deltaMs / hour)}h ago`;
		if (deltaMs < 30 * day) return `${Math.floor(deltaMs / day)}d ago`;
		return new Date(unixSeconds * 1000).toLocaleDateString();
	}

	async function reloadPreset(p) {
		if (reloadingId) return;
		reloadingId = p.preset_id;
		reloadResultById = { ...reloadResultById, [p.preset_id]: null };
		try {
			const { ok, status, payload } = await call(`${API_BASE}/presets/imported/${p.preset_id}/reload`, {
				method: 'POST',
				timeout: 0
			});
			if (!ok) {
				reloadResultById = {
					...reloadResultById,
					[p.preset_id]: { tone: 'danger', text: messageOf(payload, `Reload failed (${status})`) }
				};
				return;
			}
			const errCount = payload.lint.errors.length;
			const warnCount = payload.lint.warnings.length;
			const tone = errCount > 0 ? 'danger' : warnCount > 0 ? 'warning' : 'success';
			const text =
				errCount > 0
					? `${errCount} lint error${errCount === 1 ? '' : 's'}`
					: warnCount > 0
						? `Reloaded — ${warnCount} warning${warnCount === 1 ? '' : 's'}`
						: 'Reloaded — lint clean';
			reloadResultById = { ...reloadResultById, [p.preset_id]: { tone, text } };
			await load();
		} catch (e) {
			reloadResultById = { ...reloadResultById, [p.preset_id]: { tone: 'danger', text: 'Could not reach the server.' } };
		} finally {
			reloadingId = null;
		}
	}

	function editPreset(p) {
		try {
			sessionStorage.setItem(EDIT_STORAGE_KEY, p.preset_id);
		} catch (e) {
			// Best-effort - a private-window sessionStorage failure just means
			// the wizard opens on a blank Source step instead of prefilled.
		}
		window.dispatchEvent(
			new CustomEvent('potionui:switch-plugin-tab', { detail: { pluginId, tabId: 'import-workflow' } })
		);
	}

	async function removePreset(p) {
		if (deletingId) return;
		deletingId = p.preset_id;
		try {
			if (await deletePreset(API_BASE, p)) {
				presets = presets.filter((row) => row.preset_id !== p.preset_id);
			}
		} finally {
			deletingId = null;
		}
	}
</script>

<div class="ip-wrap">
	{#if loading}
		<div class="ip-loading"><HostUi name="Spinner" props={{ size: 'sm' }} />Loading imported presets…</div>
	{:else if loadError}
		<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true }}>{loadError}</HostUi>
	{:else if presets.length === 0}
		<div class="ip-empty">
			No presets have been imported yet. Use the <strong>Import workflow</strong> tab to bring one in from ComfyUI.
		</div>
	{:else}
		<div class="ip-table" data-imported-presets>
			<div class="ip-row ip-head">
				<span>Name</span>
				<span>Family / variant</span>
				<span>Format</span>
				<span>Requirements</span>
				<span>Created</span>
				<span></span>
			</div>
			{#each presets as p (p.preset_id)}
				{@const chip = requirementsChip(p.requirements_summary)}
				{@const reloadResult = reloadResultById[p.preset_id]}
				<div class="ip-row">
					<span class="ip-name">{p.name}</span>
					<span class="ip-fam mono">{p.family} · {p.variant}</span>
					<HostUi name="Badge" props={{ variant: 'neutral', size: 'sm' }}>{p.format}</HostUi>
					<HostUi name="Badge" props={{ variant: chip.variant, size: 'sm', class: 'font-mono' }}>{chip.text}</HostUi>
					<span class="ip-created mono">{relativeTime(p.created_at)}</span>
					<div class="ip-actions">
						<HostUi
							name="IconButton"
							props={{
								icon: 'refresh',
								label: 'Reload from source',
								size: 'sm',
								loading: reloadingId === p.preset_id,
								disabled: reloadingId === p.preset_id,
								onclick: () => reloadPreset(p)
							}}
						/>
						<HostUi name="IconButton" props={{ icon: 'edit', label: 'Edit', size: 'sm', onclick: () => editPreset(p) }} />
						<HostUi
							name="IconButton"
							props={{
								icon: 'trash',
								label: 'Delete',
								size: 'sm',
								loading: deletingId === p.preset_id,
								disabled: !!deletingId,
								onclick: () => removePreset(p)
							}}
						/>
						<HostUi
							name="Button"
							props={{
								icon: 'external-link',
								variant: 'ghost',
								size: 'sm',
								ariaLabel: 'Open in Presets',
								title: 'Open in Presets',
								href: `/admin?tab=presets&preset=${p.preset_id}`
							}}
						/>
					</div>
					{#if reloadResult}
						<div class="ip-reload-result">
							<HostUi name="Alert" props={{ variant: reloadResult.tone, density: 'compact' }}>{reloadResult.text}</HostUi>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.ip-wrap {
		width: 100%;
		max-width: none;
		padding: 4px 0;
	}
	.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-variant-numeric: tabular-nums;
	}
	.ip-loading {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 20px 0;
		color: rgb(var(--fg-muted, 169 174 184));
		font-size: 12px;
	}
	.ip-empty {
		padding: 20px;
		border: 1px solid rgb(var(--line, 36 38 44));
		border-radius: 6px;
		background: rgb(var(--surface-2, 31 33 38) / 0.4);
		color: rgb(var(--fg-subtle, 122 128 144));
		font-size: 12.5px;
		text-align: center;
	}
	.ip-table {
		border: 1px solid rgb(var(--line, 36 38 44));
		border-radius: 6px;
		background: rgb(var(--surface-1, 22 24 28));
	}
	.ip-row {
		display: grid;
		grid-template-columns: 2fr 1fr 90px 90px 100px 128px;
		align-items: center;
		gap: 14px;
		padding: 12px 16px;
	}
	.ip-row + .ip-row {
		border-top: 1px solid rgb(var(--line, 36 38 44));
	}
	.ip-head {
		border-radius: 6px 6px 0 0;
		background: rgb(var(--canvas, 12 13 15));
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 9.5px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: rgb(var(--fg-subtle, 122 128 144));
		padding-top: 8px;
		padding-bottom: 8px;
	}
	.ip-name {
		font-size: 12.5px;
		font-weight: 500;
		color: rgb(var(--fg, 232 234 237));
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.ip-fam {
		font-size: 11.5px;
		color: rgb(var(--fg-muted, 169 174 184));
	}
	.ip-created {
		font-size: 11px;
		color: rgb(var(--fg-subtle, 122 128 144));
	}
	.ip-reload-result {
		grid-column: 1 / -1;
		margin-top: 8px;
		width: fit-content;
	}
	.ip-actions {
		display: flex;
		align-items: center;
		gap: 4px;
		justify-content: flex-end;
	}
</style>
