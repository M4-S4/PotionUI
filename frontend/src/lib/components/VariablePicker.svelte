<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Icon from './Icon.svelte';
	import HighlightedText from './HighlightedText.svelte';
	import { Badge, Kbd } from '$lib/components/ui';
	import type { VariablesMap } from '$lib/utils/variableDefs';
	import {
		creatableVariableName,
		variablePickerMatcher,
		variablePickerRows,
		type VariablePickerType
	} from '$lib/utils/variablePicker';

	let {
		variables,
		selected = null,
		allowedTypes = undefined,
		disabledReasons = undefined,
		label = 'Variable',
		placeholder = 'Search variables',
		autofocus = true,
		maxListHeight = '18rem',
		class: className = '',
		onselect,
		oncreate,
		onclose
	}: {
		variables: VariablesMap;
		selected?: string | null;
		allowedTypes?: readonly VariablePickerType[];
		disabledReasons?: Record<string, string>;
		label?: string;
		placeholder?: string;
		autofocus?: boolean;
		maxListHeight?: string;
		class?: string;
		onselect?: (name: string) => void;
		oncreate?: (name: string) => void;
		onclose?: (reason: 'escape' | 'tab') => void;
	} = $props();

	const CREATE_KEY = '\u0000create';
	const uid = Math.random().toString(36).slice(2, 8);
	const listId = `variable-picker-list-${uid}`;
	const optionId = (key: string) => `variable-picker-${uid}-${key}`;

	let query = $state('');
	let activeKey = $state<string | null>(null);
	let inputRef = $state<HTMLInputElement>();
	let listRef = $state<HTMLDivElement>();

	const rows = $derived(variablePickerRows(variables, query, { allowedTypes, disabledReasons }));
	const createName = $derived(oncreate ? creatableVariableName(query, variables) : null);
	const matcher = $derived(variablePickerMatcher(query));
	const total = $derived(
		variablePickerRows(variables, '', { allowedTypes, disabledReasons }).length
	);

	const navigable = $derived([
		...rows.filter((row) => !row.disabledReason).map((row) => row.name),
		...(createName ? [CREATE_KEY] : [])
	]);
	const activeId = $derived(
		activeKey === null || !navigable.includes(activeKey)
			? undefined
			: optionId(activeKey === CREATE_KEY ? 'create' : activeKey)
	);

	$effect(() => {
		const keys = navigable;
		if (activeKey !== null && keys.includes(activeKey)) return;
		activeKey = selected && keys.includes(selected) ? selected : (keys[0] ?? null);
	});

	$effect(() => {
		if (!activeId || !listRef) return;
		listRef.querySelector<HTMLElement>(`#${activeId}`)?.scrollIntoView({ block: 'nearest' });
	});

	onMount(() => {
		if (autofocus) void tick().then(() => inputRef?.focus());
	});

	function move(delta: number) {
		if (navigable.length === 0) return;
		const current = activeKey === null ? -1 : navigable.indexOf(activeKey);
		const next = (current + delta + navigable.length) % navigable.length;
		activeKey = navigable[next];
	}

	function commit(key: string | null) {
		if (key === null) return;
		if (key === CREATE_KEY) {
			if (createName) oncreate?.(createName);
			return;
		}
		onselect?.(key);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'ArrowDown') {
			e.preventDefault();
			move(1);
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			move(-1);
		} else if (e.key === 'Home' && navigable.length > 0) {
			e.preventDefault();
			activeKey = navigable[0];
		} else if (e.key === 'End' && navigable.length > 0) {
			e.preventDefault();
			activeKey = navigable[navigable.length - 1];
		} else if (e.key === 'Enter') {
			e.preventDefault();
			e.stopPropagation();
			commit(activeKey);
		} else if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			if (query) query = '';
			else onclose?.('escape');
		} else if (e.key === 'Tab') {
			onclose?.('tab');
		}
	}

	function keepFocus(e: MouseEvent) {
		e.preventDefault();
	}
</script>

<div class="flex min-h-0 flex-col {className}" data-testid="variable-picker">
	<div class="relative flex items-center border-b border-line p-2">
		<Icon name="search" className="pointer-events-none absolute left-5 h-4 w-4 text-fg-subtle" />
		<input
			bind:this={inputRef}
			bind:value={query}
			onkeydown={handleKeydown}
			type="text"
			role="combobox"
			aria-label={label}
			aria-expanded="true"
			aria-controls={listId}
			aria-autocomplete="list"
			aria-activedescendant={activeId}
			autocomplete="off"
			spellcheck="false"
			{placeholder}
			class="input h-9 pl-9 pr-8 text-sm max-md:h-11"
		/>
		{#if query}
			<button
				type="button"
				tabindex="-1"
				aria-label="Clear search"
				onmousedown={keepFocus}
				onclick={() => {
					query = '';
					inputRef?.focus();
				}}
				class="absolute right-4 rounded p-1 text-fg-subtle hover:bg-surface-3/50 hover:text-fg"
			>
				<Icon name="close" className="h-3.5 w-3.5" />
			</button>
		{/if}
	</div>

	<div
		bind:this={listRef}
		id={listId}
		role="listbox"
		tabindex="-1"
		aria-label={label}
		onmousedown={keepFocus}
		class="min-h-0 overflow-y-auto p-1"
		style="max-height: {maxListHeight}"
	>
		{#each rows as row (row.name)}
			{@const isActive = activeKey === row.name}
			<div
				id={optionId(row.name)}
				role="option"
				tabindex="-1"
				aria-selected={isActive}
				aria-disabled={row.disabledReason ? true : undefined}
				aria-label={row.disabledReason ? `$${row.name}, ${row.disabledReason}` : `$${row.name}`}
				data-variable={row.name}
				onmouseenter={() => {
					if (!row.disabledReason) activeKey = row.name;
				}}
				onclick={() => {
					if (!row.disabledReason) commit(row.name);
				}}
				onkeydown={() => {}}
				class="flex min-h-9 w-full flex-col justify-center gap-0.5 rounded px-2.5 py-1.5 text-left transition-colors max-md:min-h-11
					{row.disabledReason ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}
					{isActive ? 'bg-signal/10' : row.disabledReason ? '' : 'hover:bg-surface-3/50'}
					{selected === row.name ? 'border-l-2 border-signal' : ''}"
			>
				<span class="flex min-w-0 items-center gap-2">
					<span class="min-w-0 truncate font-mono text-sm text-fg">$<HighlightedText text={row.name} {matcher} /></span>
					<Badge size="md" class="flex-shrink-0">{row.typeLabel}</Badge>
					{#if row.disabledReason}
						<span class="ml-auto flex-shrink-0 text-xs text-fg-subtle">{row.disabledReason}</span>
					{:else if row.mode}
						<span class="ml-auto flex-shrink-0 font-mono text-xs text-fg-muted">{row.mode}</span>
					{/if}
					{#if selected === row.name}
						<Icon name="check" className="h-4 w-4 flex-shrink-0 text-signal" />
					{/if}
				</span>
				{#if row.preview}
					<span class="truncate text-xs text-fg-subtle"><HighlightedText text={row.preview} {matcher} /></span>
				{/if}
			</div>
		{/each}

		{#if createName}
			<div
				id={optionId('create')}
				role="option"
				tabindex="-1"
				aria-selected={activeKey === CREATE_KEY}
				data-create={createName}
				onmouseenter={() => (activeKey = CREATE_KEY)}
				onclick={() => commit(CREATE_KEY)}
				onkeydown={() => {}}
				class="flex min-h-9 w-full cursor-pointer items-center gap-2 rounded px-2.5 py-1.5 text-sm max-md:min-h-11
					{activeKey === CREATE_KEY ? 'bg-signal/10' : 'hover:bg-surface-3/50'}"
			>
				<Icon name="plus" className="h-4 w-4 flex-shrink-0 text-fg-muted" />
				<span class="min-w-0 truncate text-fg">Create variable &ldquo;<span class="font-mono">${createName}</span>&rdquo;</span>
			</div>
		{/if}

		{#if rows.length === 0 && !createName}
			<div class="px-3 py-6 text-center text-sm text-fg-subtle" role="status" data-testid="variable-picker-empty">
				{#if total === 0}
					No variables yet
				{:else}
					No variables match &ldquo;{query.trim()}&rdquo;
				{/if}
			</div>
		{/if}
	</div>

	<div class="flex items-center justify-between gap-3 border-t border-line bg-surface-2/50 px-3 py-1.5 text-xs text-fg-subtle">
		<div class="hidden items-center gap-3 md:flex">
			<span class="flex items-center gap-1"><Kbd keys="↑↓" size="md" /> navigate</span>
			<span class="flex items-center gap-1"><Kbd keys="↵" size="md" /> select</span>
			<span class="flex items-center gap-1"><Kbd keys="esc" size="md" /> close</span>
		</div>
		<span class="ml-auto font-mono tabular-nums text-fg-muted">{rows.length}/{total}</span>
	</div>
</div>
