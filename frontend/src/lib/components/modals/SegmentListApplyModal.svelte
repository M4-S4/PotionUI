<script lang="ts">
	import { createEventDispatcher, tick } from 'svelte';
	import dismiss from '$lib/actions/dismiss';
	import { api } from '$lib/services/api';
	import type { Prompt, SegmentTemplate } from '$lib/types/segments';
	import { flattenRichSegments, type SegmentApplyMode } from '$lib/utils/richSegments';
	import type { PresetSegmentTemplate } from '$lib/utils/presetSegmentTemplates';
	import { logger } from '$lib/utils/logger';
	import { debounce } from '$lib/utils/time';
	import { modelDisplayName } from '$lib/utils/modelDisplay';
	import { parseVariableUsageTokens } from '$lib/utils/promptVariables';
	import { timeAgo } from '$lib/utils/relativeTime';
	import SegmentModal from './SegmentModal.svelte';
	import ModelAssignmentModal from './ModelAssignmentModal.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { Badge, Button, Spinner, Alert } from '$lib/components/ui';
	import PromptFiltersPopover from '$lib/prompts/PromptFiltersPopover.svelte';
	import {
		DEFAULT_PROMPT_FILTERS,
		promptFilterChips,
		type PromptFilters,
		type PromptSortBy,
		promptFilters
	} from '$lib/prompts/promptFilters';
	import { queryPromptLibrary } from '$lib/prompts/promptLibraryQuery';
	import { promptCardTitle, summarizePromptVariables } from '$lib/prompts/promptCardDisplay';

	export let isOpen = false;
	export let kind: 'prompt' | 'template' = 'prompt';
	export let targetHasMeaningfulContent = false;
	export let presetTemplates: PresetSegmentTemplate[] = [];

	type LibraryItem = Prompt | SegmentTemplate | PresetSegmentTemplate;

	const dispatch = createEventDispatcher<{
		close: void;
		apply: { item: LibraryItem; mode: SegmentApplyMode };
	}>();

	const PAGE_SIZE = 48;
	const SORT_OPTIONS: ReadonlyArray<{ value: PromptSortBy; label: string }> = [
		{ value: 'last_used_at', label: 'Last used' },
		{ value: 'created_at', label: 'Created' },
		{ value: 'name', label: 'Name' },
		{ value: 'usage_count', label: 'Most used' }
	];

	let loadedItems: LibraryItem[] = [];
	let total = 0;
	let loading = false;
	let loadingMore = false;
	let error = '';
	let searchTerm = '';
	let selectedId: string | null = null;
	let applyMode: SegmentApplyMode = 'append';
	let previousOpen = false;
	let showReplaceConfirmation = false;
	const applyModes: { mode: SegmentApplyMode; label: string }[] = [
		{ mode: 'append', label: 'Append' },
		{ mode: 'prepend', label: 'Prepend' },
		{ mode: 'replace', label: 'Replace' }
	];

	let filters: PromptFilters = { ...DEFAULT_PROMPT_FILTERS };
	let filtersOpen = false;
	let showModelPicker = false;
	let semanticHitCount = 0;
	let promptRequestId = 0;
	const scheduleLoad = debounce(() => void loadPrompts(true), 250);
	let models: Array<{ id: string; filename?: string; name?: string; model_type?: string }> = [];
	let modelsLoaded = false;
	let listEl: HTMLDivElement | undefined;
	let filtersTriggerEl: HTMLDivElement | undefined;

	$: items = kind === 'template' ? [...presetTemplates, ...loadedItems] : loadedItems;
	$: normalizedSearch = searchTerm.trim().toLowerCase();
	$: visibleItems =
		kind === 'template'
			? items.filter((item) => {
					if (!normalizedSearch) return true;
					return [itemName(item), itemPreview(item), itemDescription(item), ...(item.tags || [])]
						.join(' ')
						.toLowerCase()
						.includes(normalizedSearch);
				})
			: items;
	$: selectedItem = visibleItems.find((item) => item.id === selectedId) || null;
	$: selectedPrompt = kind === 'prompt' && selectedItem ? (selectedItem as Prompt) : null;
	$: selectedVariables = selectedPrompt ? summarizePromptVariables(selectedPrompt.variables) : null;
	$: title = kind === 'prompt' ? 'Apply Prompt' : 'Apply Segment Template';
	$: context = kind === 'prompt' ? `${total} prompt${total === 1 ? '' : 's'} in your library` : `${visibleItems.length} template${visibleItems.length === 1 ? '' : 's'}`;
	$: hasMore = kind === 'prompt' && loadedItems.length < total;
	$: filterCount = promptFilters.activeCount(filters);
	$: modelLabel = filters.modelId
		? modelDisplayName(models.find((model) => model.id === filters.modelId)) || 'Selected model'
		: 'Any';
	$: filterChips = promptFilterChips(filters, modelLabel !== 'Any' ? modelLabel : undefined);
	$: searchHint = semanticHitCount > 0 ? `semantic · ${semanticHitCount} hits` : null;
	$: emptyReason = filters.q.trim() || filterCount > 0 ? 'No prompts match.' : 'No prompts yet.';

	$: if (isOpen !== previousOpen) {
		previousOpen = isOpen;
		if (isOpen) {
			resetSelection();
			resetLibrary();
			loadItems();
		}
	}

	function itemName(item: LibraryItem): string {
		if (kind === 'prompt') {
			const prompt = item as Prompt;
			return prompt.name || prompt.display_name || itemPreview(item) || 'Untitled Prompt';
		}
		return item.name || 'Untitled Segment Template';
	}

	function itemPreview(item: LibraryItem): string {
		if ('flattened_text' in item && item.flattened_text) return item.flattened_text;
		return flattenRichSegments(item.segments);
	}

	function itemDescription(item: LibraryItem): string {
		return 'description' in item ? item.description || '' : '';
	}

	function isPresetItem(item: LibraryItem): boolean {
		return 'origin' in item && item.origin === 'preset';
	}

	function promptSourceLabel(prompt: Prompt): string | null {
		return prompt.source_provider && prompt.source_provider !== 'manual' ? prompt.source_provider : null;
	}

	function promptUsageLabel(prompt: Prompt): string {
		return prompt.usage_count ? `used ${prompt.usage_count}×` : 'never used';
	}

	function promptRelativeLabel(prompt: Prompt): string {
		const when = prompt.last_used_at || prompt.created_at;
		return when ? timeAgo(when) : '';
	}

	function resetSelection() {
		searchTerm = '';
		selectedId = null;
		applyMode = 'append';
		showReplaceConfirmation = false;
	}

	function resetLibrary() {
		scheduleLoad.cancel();
		filters = { ...DEFAULT_PROMPT_FILTERS };
		filtersOpen = false;
		showModelPicker = false;
		semanticHitCount = 0;
		total = 0;
		error = '';
	}

	function loadItems() {
		if (kind === 'prompt') {
			if (!modelsLoaded) void loadModels();
			void loadPrompts(true);
		} else {
			void loadTemplates();
		}
	}

	async function loadModels() {
		modelsLoaded = true;
		try {
			const response = await api.getModels({ limit: 200, sort_by: 'filename', sort_order: 'asc' });
			models = response.data?.models || [];
		} catch {
			models = [];
		}
	}

	async function loadTemplates() {
		loading = true;
		error = '';
		try {
			const response = await api.listSegmentTemplates();
			if (!response.success) throw new Error(response.error || 'Failed to load Segment Templates');
			loadedItems = response.data?.templates || [];
		} catch (loadError) {
			logger.error('Failed to load template library:', loadError);
			error = loadError instanceof Error ? loadError.message : 'Failed to load template library';
			loadedItems = [];
		} finally {
			loading = false;
		}
	}

	async function loadPrompts(reset: boolean) {
		const requestId = ++promptRequestId;
		if (reset) loading = true;
		else loadingMore = true;
		error = '';
		try {
			const result = await queryPromptLibrary(api, filters, {
				limit: PAGE_SIZE,
				offset: reset ? 0 : loadedItems.length
			});
			if (requestId !== promptRequestId) return;
			loadedItems = reset ? result.rows : [...loadedItems, ...result.rows];
			total = result.total;
			semanticHitCount = (reset ? 0 : semanticHitCount) + result.semanticHits;
		} catch (loadError) {
			if (requestId !== promptRequestId) return;
			logger.error('Failed to load prompt library:', loadError);
			error = loadError instanceof Error ? loadError.message : 'Failed to load prompt library';
			if (reset) loadedItems = [];
		} finally {
			if (requestId === promptRequestId) {
				loading = false;
				loadingMore = false;
			}
		}
	}

	function updateFilters(next: PromptFilters) {
		filters = next;
		scheduleLoad();
	}

	function handleSearchInput() {
		if (kind !== 'prompt') return;
		updateFilters({ ...filters, q: searchTerm });
	}

	function handleSortChange(event: Event) {
		updateFilters({ ...filters, sortBy: (event.currentTarget as HTMLSelectElement).value as PromptSortBy });
	}

	function removeChip(key: string) {
		updateFilters(promptFilters.clearChip(filters, key));
	}

	function clearFilters() {
		updateFilters(promptFilters.clearAll(filters));
	}

	function selectModelForFilter(model: { id: string } | null) {
		showModelPicker = false;
		if (model && !models.some((entry) => entry.id === model.id)) models = [model as never, ...models];
		updateFilters({ ...filters, modelId: model?.id || '' });
	}

	function moveSelection(step: 1 | -1) {
		if (!visibleItems.length) return;
		const index = visibleItems.findIndex((item) => item.id === selectedId);
		const next = index < 0 ? 0 : Math.min(Math.max(index + step, 0), visibleItems.length - 1);
		selectedId = visibleItems[next].id;
		void tick().then(() => {
			const row = listEl?.querySelector('[aria-selected="true"]') as HTMLElement | null;
			row?.scrollIntoView?.({ block: 'nearest' });
		});
	}

	function handleListKeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
			event.preventDefault();
			moveSelection(event.key === 'ArrowDown' ? 1 : -1);
		} else if (event.key === 'Enter' && !event.defaultPrevented) {
			if (!selectedItem) return;
			event.preventDefault();
			requestApply();
		}
	}

	function handleClose() {
		resetSelection();
		scheduleLoad.cancel();
		filtersOpen = false;
		showModelPicker = false;
		dispatch('close');
	}

	function requestApply() {
		if (!selectedItem) return false;
		if (applyMode === 'replace' && targetHasMeaningfulContent) {
			showReplaceConfirmation = true;
			return false;
		}
		applySelection();
		return true;
	}

	function applySelection() {
		if (!selectedItem) return;
		dispatch('apply', { item: selectedItem, mode: applyMode });
		resetSelection();
	}
</script>

{#snippet addAs()}
	<fieldset class="flex items-center gap-2">
		<legend class="sr-only">Add as</legend>
		<span class="text-sm text-fg-subtle" aria-hidden="true">Add as</span>
		<div class="inline-grid grid-cols-3 gap-0.5 rounded border border-line bg-surface-2 p-0.5">
			{#each applyModes as choice (choice.mode)}
				<label
					class="flex h-7 cursor-pointer items-center justify-center rounded-sm px-2.5 text-sm font-medium transition-colors {applyMode === choice.mode
						? 'bg-surface-3 text-fg shadow-raised'
						: 'text-fg-muted hover:text-fg'}"
				>
					<input class="sr-only" type="radio" name="segment-apply-mode" value={choice.mode} bind:group={applyMode} />
					{choice.label}
				</label>
			{/each}
		</div>
	</fieldset>
{/snippet}

<SegmentModal
	{isOpen}
	tier="L"
	icon={kind === 'prompt' ? 'document' : 'layout-template'}
	{title}
	{context}
	primaryLabel="Apply"
	primaryDisabled={!selectedItem}
	footerStart={addAs}
	onConfirm={requestApply}
	onCancel={handleClose}
>
	<div class="flex h-full min-h-0 flex-col">
		<div class="flex flex-wrap items-center gap-2 border-b border-line px-5 py-3 max-md:px-4">
			<div class="input flex h-9 min-w-[14rem] flex-1 items-center gap-2 max-md:h-11 max-md:basis-full">
				<Icon name="search" className="h-4 w-4 flex-shrink-0 text-fg-subtle" />
				<input
					type="search"
					class="min-w-0 flex-1 bg-transparent text-base outline-none placeholder:text-fg-subtle"
					placeholder={kind === 'prompt' ? 'Search prompts…' : 'Search template names and content…'}
					aria-label={kind === 'prompt' ? 'Search prompts' : 'Search Segment Templates'}
					bind:value={searchTerm}
					data-autofocus
					on:input={handleSearchInput}
					on:keydown={handleListKeydown}
				/>
				{#if loading && loadedItems.length > 0}
					<Spinner size="sm" />
				{:else if searchHint}
					<span class="whitespace-nowrap font-mono text-sm tabular-nums text-fg-subtle">{searchHint}</span>
				{:else}
					<span class="whitespace-nowrap font-mono text-sm tabular-nums text-fg-subtle">
						{kind === 'prompt' ? total : visibleItems.length}
					</span>
				{/if}
			</div>

			{#if kind === 'prompt'}
				<div class="relative flex-shrink-0" bind:this={filtersTriggerEl}>
					<button
						type="button"
						class="input flex h-9 items-center gap-1.5 whitespace-nowrap text-sm font-medium max-md:h-11"
						aria-haspopup="dialog"
						aria-expanded={filtersOpen}
						on:click={() => (filtersOpen = !filtersOpen)}
					>
						<span>Filters</span>
						{#if filterCount > 0}
							<span class="font-mono tabular-nums text-signal">{filterCount}</span>
						{/if}
						<Icon name="chevron-down" className="h-3 w-3 text-fg-subtle" />
					</button>
					{#if filtersOpen}
						<div
							class="absolute right-0 top-[calc(100%+6px)] z-40"
							use:dismiss={{ onDismiss: () => (filtersOpen = false), ignore: [filtersTriggerEl] }}
						>
							<PromptFiltersPopover
								{filters}
								{modelLabel}
								onChange={updateFilters}
								onOpenModelPicker={() => (showModelPicker = true)}
								onClose={() => (filtersOpen = false)}
							/>
						</div>
					{/if}
				</div>

				<label class="relative flex-shrink-0">
					<span class="sr-only">Sort</span>
					<select class="input h-9 appearance-none pr-8 text-sm font-medium max-md:h-11" value={filters.sortBy} on:change={handleSortChange}>
						{#each SORT_OPTIONS as option (option.value)}
							<option value={option.value}>{option.label}</option>
						{/each}
					</select>
				</label>
			{/if}
		</div>

		{#if kind === 'prompt' && filterChips.length > 0}
			<div class="flex flex-wrap items-center gap-1.5 border-b border-line px-5 py-2 max-md:px-4">
				{#each filterChips as chip (chip.key)}
					<span class="inline-flex h-6 items-center gap-1.5 rounded border border-signal/28 bg-signal/10 px-2 font-mono text-sm text-signal">
						{chip.label}
						<button
							type="button"
							aria-label={`Remove filter ${chip.label}`}
							class="opacity-70 hover:opacity-100"
							on:click={() => removeChip(chip.key)}
						>
							<Icon name="close" className="h-3 w-3" />
						</button>
					</span>
				{/each}
				<Button variant="ghost" size="sm" onclick={clearFilters}>Clear all</Button>
				<span class="ml-auto font-mono text-sm tabular-nums text-fg-subtle">{loadedItems.length} of {total}</span>
			</div>
		{/if}

		<div class="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_22.5rem] max-md:grid-cols-1">
			<div class="min-h-0 overflow-y-auto p-2">
				{#if loading && loadedItems.length === 0}
					<div class="flex justify-center py-12"><Spinner size="lg" /></div>
				{:else if error}
					<div class="p-2"><Alert variant="danger" live="polite">{error}</Alert></div>
				{:else if visibleItems.length === 0}
					<div class="flex flex-col items-center gap-3 px-5 py-10 text-center text-base text-fg-subtle">
						{#if kind === 'prompt'}
							<span>{emptyReason}</span>
							{#if filterCount > 0}
								<Button size="sm" variant="secondary" onclick={clearFilters}>Clear filters</Button>
							{/if}
						{:else}
							<span>No Segment Templates found.</span>
						{/if}
					</div>
				{:else}
					<div
						class="space-y-0.5"
						role="listbox"
						tabindex="-1"
						aria-label={`${kind} library`}
						bind:this={listEl}
						on:keydown={handleListKeydown}
					>
						{#each visibleItems as item (item.id)}
							{@const selected = selectedId === item.id}
							<button
								type="button"
								class="grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-3 rounded px-2.5 py-2.5 text-left transition-colors {selected
									? 'bg-signal/10 shadow-[inset_2px_0_0_rgb(var(--signal))]'
									: 'hover:bg-surface-2'}"
								role="option"
								aria-selected={selected}
								data-library-row
								on:click={() => (selectedId = item.id)}
								on:dblclick={() => {
									selectedId = item.id;
									requestApply();
								}}
							>
								{#if kind === 'prompt'}
									{@const prompt = item as Prompt}
									{@const promptTitle = promptCardTitle(prompt)}
									{@const negative = prompt.usage_hint === 'negative'}
									{@const variablesSummary = summarizePromptVariables(prompt.variables)}
									{@const sourceLabel = promptSourceLabel(prompt)}
									{@const relativeLabel = promptRelativeLabel(prompt)}
									<Tooltip text={negative ? 'Negative prompt' : 'Positive prompt'}>
										<span
											class="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded border font-mono text-sm {negative
												? 'border-danger/35 text-danger'
												: 'border-line-strong bg-surface-1 text-fg-muted'}"
										>
											{negative ? '−' : '+'}
										</span>
									</Tooltip>
									<span class="flex min-w-0 flex-col gap-1">
										<span class="min-w-0 truncate text-base font-semibold {promptTitle.untitled ? 'font-medium italic text-fg-muted' : 'text-fg'}">
											{promptTitle.text}
										</span>
										<span class="line-clamp-2 text-base leading-5 text-fg-muted">
											{#each parseVariableUsageTokens(prompt.flattened_text || '') as token, index (index)}
												{#if token.type === 'variable'}<span class="font-mono text-sm text-signal">{token.raw}</span
													>{:else}{token.raw}{/if}
											{/each}
										</span>
										<span class="flex flex-wrap items-center gap-1.5">
											<Badge size="lg">{prompt.model_name || 'any model'}</Badge>
											{#if sourceLabel}<Badge size="lg">{sourceLabel}</Badge>{/if}
											{#if variablesSummary.count > 0}
												<Tooltip text={variablesSummary.broken ? 'A condition references a missing variable' : 'Variables used in this prompt'}>
													<Badge size="lg" variant={variablesSummary.broken ? 'warning' : 'signal'}>
														<Icon name="braces" className="h-3 w-3" />
														{variablesSummary.count}{variablesSummary.linked > 0 ? ` · ${variablesSummary.linked} linked` : ''}
													</Badge>
												</Tooltip>
											{/if}
											{#if prompt.nsfw}<Badge size="lg">nsfw</Badge>{/if}
											<span class="ml-auto flex items-center gap-2.5 whitespace-nowrap font-mono text-sm tabular-nums text-fg-subtle">
												<span>{promptUsageLabel(prompt)}</span>
												{#if relativeLabel}<span>{relativeLabel}</span>{/if}
											</span>
										</span>
									</span>
								{:else}
									<span class="mt-0.5 grid h-8 w-8 place-items-center rounded border border-line bg-surface-2 text-fg-subtle">
										<Icon name="layout-template" className="h-4 w-4" />
									</span>
									<span class="flex min-w-0 flex-col gap-1">
										<span class="flex min-w-0 items-center gap-2">
											<span class="truncate text-base font-medium text-fg">{itemName(item)}</span>
											{#if isPresetItem(item)}<Badge size="lg" class="flex-shrink-0">Preset</Badge>{/if}
										</span>
										<span class="line-clamp-2 text-sm leading-5 text-fg-muted">{itemPreview(item) || 'Blank starter segments'}</span>
										<span class="font-mono text-sm text-fg-subtle">{item.segments.length} segment{item.segments.length === 1 ? '' : 's'}</span>
									</span>
								{/if}
								<span
									class="mt-0.5 grid h-[18px] w-[18px] flex-shrink-0 place-items-center rounded-full border {selected
										? 'border-signal bg-signal text-accent-contrast'
										: 'border-line-hover text-transparent'}"
									aria-hidden="true"
								>
									<Icon name="check" className="h-3 w-3" />
								</span>
							</button>
						{/each}
						{#if hasMore}
							<div class="flex justify-center pt-2">
								<Button size="sm" variant="secondary" loading={loadingMore} disabled={loadingMore} onclick={() => loadPrompts(false)}>
									Show {Math.min(PAGE_SIZE, total - loadedItems.length)} more
								</Button>
							</div>
						{/if}
					</div>
				{/if}
			</div>

			<aside class="min-h-0 overflow-y-auto border-l border-line bg-surface-2/30 px-5 py-4 max-md:hidden" aria-label="Preview">
				{#if selectedItem}
					<span class="font-mono text-sm uppercase tracking-[0.04em] text-fg-subtle">Preview</span>
					<h3 class="mt-1 break-words text-md font-semibold text-fg">{itemName(selectedItem)}</h3>
					<p class="mt-0.5 text-sm text-fg-subtle">
						{#if selectedPrompt}{selectedPrompt.model_name || 'any model'} · {/if}{selectedItem.segments.length} segment{selectedItem.segments.length === 1 ? '' : 's'}{#if selectedVariables && selectedVariables.count > 0}&nbsp;· {selectedVariables.count} variable{selectedVariables.count === 1 ? '' : 's'}{/if}
					</p>
					{#if itemDescription(selectedItem)}
						<p class="mt-2 text-sm text-fg-muted">{itemDescription(selectedItem)}</p>
					{/if}
					<div class="mt-3 space-y-2">
						{#each selectedItem.segments as segment, index (index)}
							<div class="rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm leading-5 text-fg-muted">
								<span class="block font-semibold text-fg">{index + 1} · {segment.name || `Segment ${index + 1}`}</span>
								<span class="line-clamp-3 break-words">{segment.content || 'Empty'}</span>
							</div>
						{:else}
							<div class="rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm leading-5 text-fg-muted">{itemPreview(selectedItem) || 'Blank prompt'}</div>
						{/each}
					</div>
					<p class="mt-3 text-sm text-fg-subtle">Incoming segments are detached copies. Other generation settings are not changed.</p>
				{:else}
					<p class="py-10 text-center text-base text-fg-subtle">Pick one on the left to see what it adds.</p>
				{/if}
			</aside>
		</div>
	</div>
</SegmentModal>

{#if showModelPicker}
	<ModelAssignmentModal
		selectionMode="single"
		selectedModelId={filters.modelId || null}
		allowClear={true}
		title="Filter prompts by model"
		subtitle="Search the model catalog or narrow it by type, then select one model."
		onSelect={selectModelForFilter}
		onClear={() => selectModelForFilter(null)}
		onClose={() => (showModelPicker = false)}
	/>
{/if}

<SegmentModal
	isOpen={showReplaceConfirmation}
	tier="S"
	role="alertdialog"
	icon="warning"
	tone="syntax"
	title="Replace current segments?"
	context={selectedItem ? `${title} · ${itemName(selectedItem)}` : title}
	primaryLabel="Replace"
	onConfirm={() => {
		showReplaceConfirmation = false;
		applySelection();
	}}
	onCancel={() => (showReplaceConfirmation = false)}
>
	<p class="px-5 py-4 text-base text-fg-muted">
		This replaces the meaningful content in this editor. Preset, mode, form values, session, backend, seed, tags, and generation settings stay unchanged.
	</p>
</SegmentModal>
