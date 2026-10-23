<script>
	import { untrack } from 'svelte';
	import HostUi from '../../../../sdk/HostUi.svelte';
	import {
		USER_MODE_ALL,
		USER_MODE_SELECTED,
		buildInitialFormState,
		clampLimit,
		buildFetchRequestBody,
		isFormValid,
		summarizeFetchResult
	} from './promptsFetch.js';

	let { context = {}, hookName = '', pluginId = 'civitai-provider' } = $props();

	let model = $derived(context.model || {});

	let open = $state(false);
	let phase = $state('form');
	let loadingModal = $state(false);
	let loadError = $state('');
	let submitError = $state('');
	let options = $state(null);
	let users = $state([]);
	let currentUserId = $state(null);
	let formState = $state(null);
	let result = $state(null);

	let modalHostEl = $state(null);
	let modalApi = $state.raw(null);
	let modalInstance = $state.raw(null);
	let bodyEl = $state(null);

	let userPickerEl = $state(null);
	let userPickerApi = $state.raw(null);
	let userPickerInstance = $state.raw(null);

	let usersById = $derived(Object.fromEntries(users.map((u) => [u.id, u])));

	function notify(level, message) {
		const notifications = window.__potionui?.notifications;
		if (notifications?.toast) notifications.toast(level, message);
	}

	function userLabel(userId) {
		return usersById[userId]?.username || userId;
	}

	async function fetchJson(url, init) {
		const response = await window.__potionui.api.fetch(url, init);
		const payload = await response.json().catch(() => null);
		if (!response.ok || payload?.success === false) {
			const message = payload?.message || payload?.error || `Request failed (${response.status})`;
			throw new Error(message);
		}
		return payload?.data ?? null;
	}

	async function loadModalData() {
		loadingModal = true;
		loadError = '';
		try {
			const [optionsData, meData, usersData] = await Promise.all([
				fetchJson(`/api/plugins/${pluginId}/prompts/options`),
				fetchJson('/api/auth/me'),
				fetchJson('/api/users')
			]);
			options = optionsData;
			users = Array.isArray(usersData) ? usersData : [];
			currentUserId = meData?.id ?? null;
			formState = buildInitialFormState(options, currentUserId);
		} catch (e) {
			loadError = e instanceof Error ? e.message : 'Could not load fetch options.';
		} finally {
			loadingModal = false;
		}
	}

	async function openModal(event) {
		event.stopPropagation();
		if (!model.id) return;
		if (!window.__potionui?.components?.PluginModal) {
			notify('error', 'Fetch prompts is unavailable right now.');
			return;
		}
		open = true;
		phase = 'form';
		submitError = '';
		result = null;
		await loadModalData();
	}

	function closeModal() {
		if (phase === 'submitting') return;
		open = false;
	}

	function pickerValueFromFormState(fs) {
		return fs.userMode === USER_MODE_ALL ? USER_MODE_ALL : fs.selectedUserIds;
	}

	function onUserPickerChange(next) {
		formState =
			next === USER_MODE_ALL
				? { ...formState, userMode: USER_MODE_ALL }
				: { ...formState, userMode: USER_MODE_SELECTED, selectedUserIds: next };
	}

	function userPickerProps() {
		return {
			users: users.map((u) => ({ id: u.id, username: u.username, email: u.email })),
			value: pickerValueFromFormState(formState),
			onChange: onUserPickerChange,
			disabled: phase === 'submitting'
		};
	}

	$effect(() => {
		const el = userPickerEl;
		if (!el) return;
		const api = window.__potionui?.components?.UserPicker;
		if (!api) return;
		userPickerApi = api;
		userPickerInstance = api.mount(el, untrack(userPickerProps));
		return () => {
			api.unmount(userPickerInstance);
			userPickerInstance = null;
		};
	});

	$effect(() => {
		if (!userPickerInstance || !userPickerApi || !formState) return;
		userPickerApi.update(userPickerInstance, userPickerProps());
	});

	function modalProps() {
		const base = {
			title: 'Fetch prompts',
			subtitle: model.name || model.filename || 'Model',
			onCancel: closeModal,
			mountBody: (el) => {
				el.appendChild(bodyEl);
			}
		};
		if (phase === 'result') {
			return {
				...base,
				confirmLabel: 'Close',
				confirmDisabled: false,
				busy: false,
				hideCancel: true,
				onConfirm: closeModal
			};
		}
		return {
			...base,
			confirmLabel: 'Fetch prompts',
			confirmDisabled: !formState || !isFormValid(formState) || loadingModal || !!loadError,
			busy: phase === 'submitting',
			onConfirm: submit
		};
	}

	$effect(() => {
		const el = modalHostEl;
		if (!el || !open) return;
		const api = window.__potionui?.components?.PluginModal;
		if (!api) return;
		modalApi = api;
		modalInstance = api.mount(el, untrack(modalProps));
		return () => {
			api.unmount(modalInstance);
			modalInstance = null;
		};
	});

	$effect(() => {
		if (!modalInstance || !modalApi) return;
		modalApi.update(modalInstance, modalProps());
	});

	function onLimitInput(e) {
		formState = { ...formState, limit: clampLimit(e.currentTarget.value, options?.limit_max) };
	}

	async function submit() {
		if (!formState || !isFormValid(formState) || phase === 'submitting') return;
		phase = 'submitting';
		submitError = '';

		const run = async () => {
			const body = buildFetchRequestBody(model.id, formState);
			const data = await fetchJson(`/api/plugins/${pluginId}/prompts/fetch`, {
				method: 'POST',
				timeout: 0,
				body: JSON.stringify(body)
			});
			result = summarizeFetchResult(data);
			phase = 'result';
			await context.refresh?.();
		};

		try {
			if (context.track) {
				await context.track(run);
			} else {
				await run();
			}
		} catch (e) {
			submitError = e instanceof Error ? e.message : 'Fetching prompts failed.';
			phase = 'form';
			notify('error', 'Fetching CivitAI prompts failed');
		}
	}

	function optionValue(o) {
		return typeof o === 'string' ? o : (o?.value ?? '');
	}

	function optionLabel(o) {
		return typeof o === 'string' ? o : (o?.label ?? String(o?.value ?? ''));
	}
</script>

{#if model.id}
	<span class="trigger">
		<HostUi name="IconButton" props={{ icon: 'document', label: 'Fetch prompts from CivitAI', size: 'sm', variant: 'secondary', onclick: openModal }} />
	</span>
{/if}

{#if open}
	<div bind:this={modalHostEl}></div>
	<div bind:this={bodyEl} class="body">
		{#if loadingModal}
		<div class="loading-row"><HostUi name="Spinner" props={{ size: 'sm' }} />Loading options&hellip;</div>
		{:else if loadError}
		<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true }}>{loadError}</HostUi>
		{:else if phase === 'result' && result}
		<div class="result-list">
			{#each result.models as m (m.modelId)}
				<div class="result-model">
					<div class="result-model-header">
						<HostUi name="Tooltip" props={{ text: m.modelName, wrapperClass: 'flex min-w-0' }}>
								<span class="result-model-name">{m.modelName}</span>
							</HostUi>
						{#if m.notLinked}
							<HostUi name="Badge" props={{ variant: 'warning', size: 'sm' }}>Not linked</HostUi>
						{:else if m.error}
							<HostUi name="Badge" props={{ variant: 'danger', size: 'sm' }}>Error</HostUi>
						{/if}
					</div>

					{#if m.notLinked}
						<p class="result-hint">Run Fetch CivitAI info first.</p>
					{:else if m.error}
						<p class="result-hint result-hint-danger">{m.error}</p>
					{:else}
						<div class="result-stats">
							<div class="stat">
								<span class="stat-label">Images seen</span>
								<span class="stat-value">{m.imagesSeen}</span>
							</div>
							<div class="stat">
								<span class="stat-label">With prompt</span>
								<span class="stat-value">{m.withPrompt}</span>
							</div>
							<div class="stat">
								<span class="stat-label">Created</span>
								<span class="stat-value">{m.created}</span>
							</div>
							<div class="stat">
								<span class="stat-label">Skipped duplicates</span>
								<span class="stat-value">{m.skippedDuplicates}</span>
							</div>
						</div>

						{#if m.perUserRows.length}
							<table class="per-user-table">
								<thead>
									<tr>
										<th>User</th>
										<th>Created</th>
										<th>Skipped</th>
									</tr>
								</thead>
								<tbody>
									{#each m.perUserRows as row (row.userId)}
										<tr>
											<td class="per-user-name">
												<HostUi name="Tooltip" props={{ text: userLabel(row.userId), wrapperClass: 'flex min-w-0' }}>
														<span class="per-user-label">{userLabel(row.userId)}</span>
													</HostUi>
											</td>
											<td>{row.created}</td>
											<td>{row.skippedDuplicates}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						{/if}
					{/if}
				</div>
			{/each}
		</div>
		{:else if formState}
		{#if submitError}
			<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true }}>{submitError}</HostUi>
		{/if}

		<div class="field-group">
			<span class="label">Users</span>
			<div bind:this={userPickerEl}></div>
		</div>

		<div class="field-grid">
			<div class="field-group">
				<span class="label">Sort</span>
				<select class="input" bind:value={formState.sort} disabled={phase === 'submitting'}>
					{#each options?.sorts || [] as o}
						<option value={optionValue(o)}>{optionLabel(o)}</option>
					{/each}
				</select>
			</div>
			<div class="field-group">
				<span class="label">Period</span>
				<select class="input" bind:value={formState.period} disabled={phase === 'submitting'}>
					{#each options?.periods || [] as o}
						<option value={optionValue(o)}>{optionLabel(o)}</option>
					{/each}
				</select>
			</div>
			<div class="field-group">
				<span class="label">NSFW level</span>
				<select class="input" bind:value={formState.nsfw} disabled={phase === 'submitting'}>
					<option value={null}>Any</option>
					{#each options?.nsfw || [] as o}
						<option value={optionValue(o)}>{optionLabel(o)}</option>
					{/each}
				</select>
			</div>
			<div class="field-group">
				<span class="label">Limit</span>
				<input
					class="input"
					type="number"
					min="1"
					max={options?.limit_max ?? 200}
					value={formState.limit}
					oninput={onLimitInput}
					disabled={phase === 'submitting'}
				/>
			</div>
		</div>

		<div class="field-group">
			<div class="toggle-row">
					<HostUi
						name="Switch"
						props={{
							checked: formState.includeShowcase,
							label: 'Include creator showcase images',
							size: 'sm',
							disabled: phase === 'submitting',
							onchange: (checked) => (formState = { ...formState, includeShowcase: checked })
						}}
					/>
					<span>Include creator showcase images</span>
				</div>
			<div class="toggle-row">
					<HostUi
						name="Switch"
						props={{
							checked: formState.includeNegative,
							label: 'Also save negative prompts',
							size: 'sm',
							disabled: phase === 'submitting',
							onchange: (checked) => (formState = { ...formState, includeNegative: checked })
						}}
					/>
					<span>Also save negative prompts</span>
				</div>
		</div>
		{/if}
	</div>
{/if}

<style>
	.body {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

	.label {
		display: block;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 12px;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.07em;
		color: rgb(var(--fg-subtle, 122 128 144));
		margin-bottom: 6px;
	}
	.field-group {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.field-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 14px;
	}
	.toggle-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 13px;
		color: rgb(var(--fg, 232 234 237));
	}
	.trigger {
		display: inline-flex;
		opacity: 0;
		transition: opacity 100ms;
	}
	:global(.group:hover) .trigger,
	.trigger:focus-within {
		opacity: 1;
	}
	.result-model-name,
	.per-user-label {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}

	.input {
		width: 100%;
		min-height: 32px;
		padding: 6px 10px;
		font-size: 13px;
		background: rgb(var(--field-bg, 31 33 38));
		border: 1px solid rgb(var(--field-border, 43 46 53));
		color: rgb(var(--fg, 232 234 237));
		border-radius: 4px;
	}
	.input:focus {
		outline: 2px solid rgb(var(--accent, 255 255 255));
		outline-offset: 1px;
	}
	.input:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.loading-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 20px 0;
		color: rgb(var(--fg-muted, 169 174 184));
		font-size: 12px;
	}
	.result-list {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.result-model {
		border: 1px solid rgb(var(--line, 36 38 44));
		border-radius: 6px;
		padding: 12px 14px;
		background: rgb(var(--surface-2, 31 33 38) / 0.4);
	}
	.result-model-header {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}
	.result-model-name {
		font-size: 13px;
		font-weight: 600;
		color: rgb(var(--fg, 232 234 237));
	}
	.result-hint {
		margin: 0;
		font-size: 12px;
		color: rgb(var(--fg-muted, 169 174 184));
	}
	.result-hint-danger {
		color: rgb(var(--danger, 255 138 138));
	}
	.result-stats {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px 16px;
	}
	.stat {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.stat-label {
		font-size: 12px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: rgb(var(--fg-subtle, 122 128 144));
	}
	.stat-value {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-variant-numeric: tabular-nums;
		font-size: 14px;
		color: rgb(var(--fg, 232 234 237));
	}
	.per-user-table {
		width: 100%;
		margin-top: 10px;
		border-collapse: collapse;
		font-size: 12px;
	}
	.per-user-table th {
		text-align: left;
		font-size: 12px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: rgb(var(--fg-subtle, 122 128 144));
		padding: 4px 8px 4px 0;
		border-bottom: 1px solid rgb(var(--line, 36 38 44));
	}
	.per-user-table td {
		padding: 5px 8px 5px 0;
		color: rgb(var(--fg-muted, 169 174 184));
		border-bottom: 1px solid rgb(var(--line, 36 38 44) / 0.5);
	}
	.per-user-table td.per-user-name {
		max-width: 12rem;
		color: rgb(var(--fg, 232 234 237));
	}
</style>
