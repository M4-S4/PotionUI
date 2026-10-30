<script lang="ts">
	import { onMount } from 'svelte';
	import * as adminApi from '$lib/services/admin-api';
	import { logger, getApiErrorMessage } from '$lib/utils/logger';
	import { timeAgo } from '$lib/utils/relativeTime';
	import { toasts } from '$lib/stores/toast';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { Alert, Badge, Button, IconButton, LoadErrorState, Spinner, Switch } from '$lib/components/ui';
	import {
		DetailBody,
		DetailFooter,
		DetailLayout,
		DetailSection,
		KVGrid,
		KVItem
	} from '$lib/components/detail';
	import type { LLMProfile, LLMProfileSchema } from '$lib/types/llm';
	import {
		SOURCE_LABEL,
		SOURCE_VARIANT,
		buildProfileGroups,
		clearProfileDraft,
		formatProfileValue,
		isOverridden,
		profileDraftCount,
		profileDraftPayload,
		profileHasProblems,
		profileLayers,
		seedRowLines,
		setProfileDraft,
		type ProfileDraft,
		type ProfileRow
	} from './llmProfile';

	let { configId, draft = $bindable({}) }: { configId: string; draft?: ProfileDraft } = $props();

	let profile = $state<LLMProfile | null>(null);
	let schema = $state<LLMProfileSchema | null>(null);
	let loading = $state(true);
	let loadError = $state<string | null>(null);
	let refreshing = $state(false);
	let saving = $state(false);

	const groups = $derived(profile && schema ? buildProfileGroups(profile, schema, draft) : []);
	const dirtyCount = $derived(profileDraftCount(draft));
	const blocked = $derived(profileHasProblems(groups));

	onMount(load);

	async function load() {
		loading = true;
		loadError = null;
		try {
			const [profileResponse, schemaResponse] = await Promise.all([
				adminApi.getLLMProfile(configId),
				adminApi.getLLMProfileSchema()
			]);
			if (profileResponse.success && profileResponse.data && schemaResponse.success && schemaResponse.data) {
				profile = profileResponse.data;
				schema = schemaResponse.data;
			} else {
				loadError = profileResponse.message || schemaResponse.message || 'Failed to load the capability profile';
			}
		} catch (error) {
			logger.error('Failed to load the capability profile:', error);
			loadError = getApiErrorMessage(error, 'Failed to load the capability profile');
		} finally {
			loading = false;
		}
	}

	async function refresh() {
		refreshing = true;
		try {
			const response = await adminApi.refreshLLMProfile(configId);
			if (response.success && response.data) {
				profile = response.data;
			} else {
				toasts.error(response.message || 'Failed to refresh the capability profile');
			}
		} catch (error) {
			logger.error('Failed to refresh the capability profile:', error);
			toasts.error(getApiErrorMessage(error, 'Failed to refresh the capability profile'));
		} finally {
			refreshing = false;
		}
	}

	async function save() {
		saving = true;
		try {
			const response = await adminApi.updateLLMProfile(configId, profileDraftPayload(draft));
			if (response.success && response.data) {
				profile = response.data;
				draft = {};
				toasts.success('Capability profile saved');
			} else {
				toasts.error(response.message || 'Failed to save the capability profile');
			}
		} catch (error) {
			logger.error('Failed to save the capability profile:', error);
			toasts.error(getApiErrorMessage(error, 'Failed to save the capability profile'));
		} finally {
			saving = false;
		}
	}

	function discard() {
		draft = {};
	}

	function edit(row: ProfileRow, next: unknown) {
		draft = setProfileDraft(draft, row, next);
	}

	function clear(row: ProfileRow) {
		draft = clearProfileDraft(draft, row);
	}

	function editNumber(row: ProfileRow, event: Event) {
		const raw = (event.currentTarget as HTMLInputElement).value.trim();
		const parsed = Number(raw);
		if (raw !== '' && Number.isFinite(parsed)) edit(row, parsed);
	}

	function restoreNumber(row: ProfileRow, event: Event) {
		(event.currentTarget as HTMLInputElement).value = typeof row.value === 'number' ? String(row.value) : '';
	}

	function editText(row: ProfileRow, event: Event) {
		const raw = (event.currentTarget as HTMLInputElement).value;
		edit(row, raw.trim() === '' ? null : raw);
	}

	function toggleOption(row: ProfileRow, option: string) {
		const current = Array.isArray(row.value) ? row.value.map(String) : [];
		const next = row.options.filter((entry) =>
			entry === option ? !current.includes(option) : current.includes(entry)
		);
		edit(row, next);
	}

	function enumOptions(row: ProfileRow): string[] {
		const current = row.value === null || row.value === undefined ? null : String(row.value);
		return current !== null && !row.options.includes(current) ? [...row.options, current] : row.options;
	}
</script>

{#snippet control(row: ProfileRow)}
	{@const id = `profile-${row.name}`}
	{#if row.kind === 'enum'}
		<select
			{id}
			class="input"
			value={row.value === null || row.value === undefined ? '' : String(row.value)}
			onchange={(event) => edit(row, event.currentTarget.value === '' ? null : event.currentTarget.value)}
		>
			{#if row.value === null || row.value === undefined}
				<option value="">Not set</option>
			{/if}
			{#each enumOptions(row) as option (option)}
				<option value={option}>{option}</option>
			{/each}
		</select>
	{:else if row.kind === 'set'}
		<div {id} class="flex flex-wrap gap-1.5" role="group" aria-label={row.label}>
			{#each row.options as option (option)}
				{@const on = Array.isArray(row.value) && row.value.map(String).includes(option)}
				<button
					type="button"
					class="rounded border px-2 py-1 font-mono text-sm transition-colors {on
						? 'border-signal/40 bg-signal/10 text-signal'
						: 'border-line text-fg-muted hover:bg-surface-2 hover:text-fg'}"
					aria-pressed={on}
					data-option={option}
					onclick={() => toggleOption(row, option)}
				>
					{option}
				</button>
			{/each}
		</div>
	{:else if row.kind === 'int' || row.kind === 'float'}
		<input
			{id}
			type="number"
			class="input font-mono tabular-nums {row.problem ? 'border-danger' : ''}"
			value={typeof row.value === 'number' ? row.value : ''}
			min={row.min}
			max={row.max}
			step={row.kind === 'int' ? 1 : 'any'}
			aria-invalid={row.problem ? 'true' : undefined}
			oninput={(event) => editNumber(row, event)}
			onblur={(event) => restoreNumber(row, event)}
		/>
	{:else if row.kind === 'bool'}
		<Switch {id} label={row.label} checked={row.value === true} onchange={(checked) => edit(row, checked)} />
	{:else if row.kind === 'string'}
		<input
			{id}
			type="text"
			class="input font-mono"
			value={typeof row.value === 'string' ? row.value : ''}
			oninput={(event) => editText(row, event)}
		/>
	{/if}
{/snippet}

{#snippet profileRow(row: ProfileRow)}
	{@const layers = profileLayers(row)}
	<div
		class="grid gap-x-6 gap-y-2 border-b border-line px-4 py-3 last:border-b-0 sm:px-5 md:grid-cols-[minmax(0,1fr)_minmax(0,19rem)] md:items-start"
		data-testid="profile-row"
		data-field={row.name}
		data-source={row.source}
	>
		<div class="min-w-0">
			<div class="flex flex-wrap items-center gap-2">
				<label for="profile-{row.name}" class="text-sm font-medium text-fg">{row.label}</label>
				<Badge size="sm" variant={SOURCE_VARIANT[row.source]} dot>{SOURCE_LABEL[row.source]}</Badge>
				{#if row.edited}
					<span class="font-mono text-sm text-warning" data-testid="profile-unsaved">Unsaved</span>
				{/if}
			</div>
			<p class="mt-0.5 font-mono text-sm text-fg-subtle">{row.name}</p>
			{#if layers.length > 0}
				<p class="mt-1 break-words font-mono text-sm tabular-nums text-fg-subtle" data-testid="profile-layers">
					{layers.join(' · ')}
				</p>
			{/if}
		</div>
		<div class="min-w-0">
			<div class="flex items-start gap-2">
				<div class="min-w-0 flex-1">
					{#if row.editable}
						{@render control(row)}
					{:else}
						<code class="block break-words font-mono text-sm text-fg-muted" data-testid="profile-readonly">{formatProfileValue(row.value)}</code>
					{/if}
				</div>
				<div class="w-8 shrink-0">
					{#if row.editable && (isOverridden(row) || row.edited)}
						<Tooltip text="Clear override">
							<IconButton icon="undo" label="Clear override for {row.label}" size="sm" onclick={() => clear(row)} />
						</Tooltip>
					{/if}
				</div>
			</div>
			{#if row.problem}
				<p class="mt-1 text-sm text-danger" role="alert" data-testid="profile-problem">{row.problem}</p>
			{/if}
		</div>
	</div>
{/snippet}

{#snippet refreshAction()}
	<Button variant="secondary" size="sm" icon="refresh" loading={refreshing} disabled={saving} onclick={refresh}>
		Refresh
	</Button>
{/snippet}

{#if loading}
	<DetailBody>
		<div class="flex flex-col items-center justify-center py-16" data-testid="profile-loading">
			<Spinner size="lg" />
			<p class="mt-4 text-sm text-fg-muted">Loading capability profile…</p>
		</div>
	</DetailBody>
{:else if loadError || !profile}
	<DetailBody>
		<LoadErrorState message={loadError ?? 'Failed to load the capability profile'} onRetry={load} retrying={loading} />
	</DetailBody>
{:else}
	{@const current = profile}
	{@const seedLines = seedRowLines(current)}
	<DetailBody>
		<DetailLayout>
			{#snippet main()}
				{#if current.discovery_error}
					<Alert variant="warning" density="compact" icon title="Discovery failed">
						<span data-testid="profile-discovery-error">{current.discovery_error}</span>
						<span class="block">Values the server could not discover fall back to the built-in seed.</span>
					</Alert>
				{/if}
				{#each groups as group (group.key)}
					<DetailSection label={group.label} padded={false}>
						<div data-testid="profile-group" data-group={group.key}>
							{#each group.rows as row (row.name)}
								{@render profileRow(row)}
							{/each}
						</div>
					</DetailSection>
				{/each}
			{/snippet}
			{#snippet aside()}
				<DetailSection label="Profile" headerExtra={refreshAction}>
					<KVGrid>
						<KVItem label="Adapter" mono full>{current.adapter}</KVItem>
						<KVItem label="Model" mono full>{current.model}</KVItem>
						<KVItem label="Discovered" mono full>
							{current.discovered_at ? timeAgo(current.discovered_at) : 'Never'}
						</KVItem>
						{#if seedLines.length > 0}
							<KVItem label="Seed rows" mono full>
								<ul class="space-y-0.5" data-testid="profile-seed-rows">
									{#each seedLines as line (line)}
										<li class="break-words font-mono text-sm text-fg-subtle" data-testid="profile-seed-row">{line}</li>
									{/each}
								</ul>
							</KVItem>
						{/if}
					</KVGrid>
				</DetailSection>
				<DetailSection label="Verification">
					{#if current.verified}
						<div class="mb-3" data-testid="profile-verified">
							<Badge variant="success" dot>Verified</Badge>
						</div>
						<KVGrid>
							<KVItem label="Suite" mono full>{current.verified.suite}</KVItem>
							<KVItem label="Passed" mono>{current.verified.passed_at}</KVItem>
							{#if current.verified.runtime}
								<KVItem label="Runtime" mono>{current.verified.runtime}</KVItem>
							{/if}
							{#if current.verified.gpu_tier}
								<KVItem label="GPU tier" mono>{current.verified.gpu_tier}</KVItem>
							{/if}
						</KVGrid>
					{:else}
						<p class="text-sm text-fg-muted" data-testid="profile-unverified">
							Not verified. A verified profile has passed the chat evaluation suite on a known runtime and GPU tier.
						</p>
					{/if}
				</DetailSection>
			{/snippet}
		</DetailLayout>
	</DetailBody>
	<DetailFooter
		dirtyCount={dirtyCount}
		mode="edit"
		{saving}
		canSave={!blocked}
		onSave={save}
		onDiscard={discard}
	/>
{/if}
