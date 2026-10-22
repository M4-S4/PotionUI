<script lang="ts">
	import { tick } from 'svelte';
	import AnchoredPanel from './AnchoredPanel.svelte';
	import HighlightedText from './HighlightedText.svelte';
	import Icon from './Icon.svelte';
	import { Kbd } from '$lib/components/ui';
	import tooltip from '$lib/actions/tooltip';
	import { compileTextMatcher, textMatches } from '$lib/utils/textMatch';

	let {
		options,
		value,
		multiple = false,
		max = Infinity,
		label,
		invalid = false,
		placeholder = multiple ? 'Pick one or more options' : 'Pick an option',
		disabledReason,
		marks = {},
		markTips = {},
		summary = '',
		class: className = '',
		onchange
	}: {
		options: string[];
		value: string[];
		multiple?: boolean;
		max?: number;
		label: string;
		invalid?: boolean;
		placeholder?: string;
		disabledReason?: (option: string) => string | null;
		marks?: Record<string, string>;
		markTips?: Record<string, string>;
		summary?: string;
		class?: string;
		onchange: (next: string[]) => void;
	} = $props();

	const uid = Math.random().toString(36).slice(2, 8);
	const listId = `option-select-list-${uid}`;
	const optionId = (position: number) => `option-select-${uid}-${position}`;

	let open = $state(false);
	let query = $state('');
	let active = $state(-1);
	let triggerRef = $state<HTMLButtonElement>();
	let inputRef = $state<HTMLInputElement>();
	let listRef = $state<HTMLDivElement>();

	const picked = $derived(value.filter((v) => v !== ''));
	const matcher = $derived(compileTextMatcher(query, 'contains').matcher);
	const rows = $derived(
		options
			.map((option) => ({ option, reason: disabledReason?.(option) ?? null }))
			.filter((row) => textMatches(matcher, row.option))
	);
	const activeId = $derived(active >= 0 && active < rows.length ? optionId(active) : undefined);
	const footerText = $derived(multiple ? `${picked.length} selected${Number.isFinite(max) ? ` · max ${max}` : ''}` : summary);

	$effect(() => {
		if (!open || !activeId || !listRef) return;
		listRef.querySelector<HTMLElement>(`#${activeId}`)?.scrollIntoView({ block: 'nearest' });
	});

	function firstEnabled(from: number, step: number): number {
		const count = rows.length;
		for (let n = 0; n < count; n++) {
			const index = (from + step * n + count * count) % count;
			if (!rows[index].reason) return index;
		}
		return -1;
	}

	async function show() {
		open = true;
		query = '';
		const current = rows.findIndex((row) => !row.reason && picked.includes(row.option));
		active = current >= 0 ? current : firstEnabled(0, 1);
		await tick();
		inputRef?.focus();
	}

	function close(refocus: boolean) {
		open = false;
		query = '';
		if (refocus) void tick().then(() => triggerRef?.focus());
	}

	function choose(position: number) {
		const row = rows[position];
		if (!row || row.reason) return;
		active = position;
		if (!multiple) {
			onchange([row.option]);
			close(true);
			return;
		}
		if (picked.includes(row.option)) onchange(picked.filter((v) => v !== row.option));
		else if (picked.length < max) onchange([...picked, row.option]);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			if (rows.length) active = firstEnabled(active + (e.key === 'ArrowDown' ? 1 : -1), e.key === 'ArrowDown' ? 1 : -1);
		} else if (e.key === 'Home' || e.key === 'End') {
			e.preventDefault();
			if (rows.length) active = e.key === 'Home' ? firstEnabled(0, 1) : firstEnabled(rows.length - 1, -1);
		} else if (e.key === 'Enter') {
			if (e.ctrlKey || e.metaKey || e.altKey) return;
			e.preventDefault();
			e.stopPropagation();
			if (!e.repeat && !e.isComposing) choose(active);
		} else if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			if (query) query = '';
			else close(true);
		} else if (e.key === 'Tab') {
			close(false);
		}
	}

	$effect(() => {
		void query;
		if (!open) return;
		if (active < 0 || active >= rows.length || rows[active]?.reason) active = firstEnabled(0, 1);
	});

	function handleTriggerKeydown(e: KeyboardEvent) {
		if (open) return;
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			void show();
		}
	}

	function keepFocus(e: MouseEvent) {
		e.preventDefault();
	}
</script>

<button
	bind:this={triggerRef}
	type="button"
	class="input flex h-auto min-h-10 w-full items-center gap-2.5 py-2 text-left text-base leading-5 max-md:min-h-11 {invalid
		? 'border-warning/60 ring-1 ring-warning/40'
		: ''} {className}"
	aria-haspopup="listbox"
	aria-expanded={open}
	aria-label={label}
	aria-invalid={invalid ? 'true' : undefined}
	data-testid="option-select"
	onclick={() => (open ? close(false) : void show())}
	onkeydown={handleTriggerKeydown}
>
	{#if picked.length === 0}
		<span class="min-w-0 flex-1 text-fg-subtle">{placeholder}</span>
	{:else if multiple}
		<span class="grid min-w-0 flex-1 gap-1.5">
			{#each picked as option (option)}
				<span class="flex min-w-0 items-center gap-2" data-option-value={option}>
					<span class="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-signal" aria-hidden="true"></span>
					<span class="min-w-0 truncate text-fg" use:tooltip={{ text: option, truncateOnly: true, wrap: true }}>{option}</span>
				</span>
			{/each}
		</span>
	{:else}
		<span
			class="line-clamp-2 min-w-0 flex-1 break-words text-fg"
			data-option-value={picked[0]}
			use:tooltip={{ text: picked[0], truncateOnly: true, wrap: true }}>{picked[0]}</span
		>
	{/if}
	<Icon name="chevron-down" className="h-3.5 w-3.5 flex-shrink-0 text-fg-subtle" />
</button>

{#if open}
	<AnchoredPanel
		anchor={triggerRef}
		label={label}
		width={560}
		heightEstimate={440}
		testId="option-picker"
		onDismiss={() => close(false)}
	>
		<div class="relative flex flex-shrink-0 items-center border-b border-line p-2">
			<Icon name="search" className="pointer-events-none absolute left-5 h-4 w-4 text-fg-subtle" />
			<input
				bind:this={inputRef}
				bind:value={query}
				onkeydown={handleKeydown}
				type="text"
				role="combobox"
				aria-label={`Search ${label}`}
				aria-expanded="true"
				aria-controls={listId}
				aria-autocomplete="list"
				aria-activedescendant={activeId}
				autocomplete="off"
				spellcheck="false"
				placeholder="Search options"
				class="input h-9 pl-9 pr-24 text-base max-md:h-11"
			/>
			<span class="pointer-events-none absolute right-5 font-mono text-sm tabular-nums text-fg-subtle" data-testid="option-count">
				{query.trim() ? `${rows.length} of ${options.length}` : options.length}
			</span>
		</div>
		<div
			bind:this={listRef}
			id={listId}
			role="listbox"
			tabindex="-1"
			aria-label={label}
			aria-multiselectable={multiple ? 'true' : undefined}
			class="min-h-0 flex-1 overflow-y-auto p-1"
			onmousedown={keepFocus}
		>
			{#each rows as row, position (row.option)}
				{@const selected = picked.includes(row.option)}
				{@const mark = marks[row.option]}
				<div
					id={optionId(position)}
					role="option"
					tabindex="-1"
					aria-selected={selected}
					aria-disabled={row.reason ? 'true' : undefined}
					data-option={row.option}
					data-active={position === active ? 'true' : undefined}
					class="grid grid-cols-[1.25rem_minmax(0,1fr)] gap-2.5 rounded px-2.5 py-2 text-base leading-5 max-md:py-2.5 {row.reason
						? 'cursor-not-allowed opacity-60'
						: 'cursor-pointer'} {position === active ? 'bg-surface-3/80 text-fg' : selected ? 'text-fg' : 'text-fg-muted'}"
					onmouseenter={() => {
						if (!row.reason) active = position;
					}}
					onclick={() => choose(position)}
					onkeydown={() => {}}
				>
					<span
						class="mt-0.5 grid h-4 w-4 place-items-center border {multiple ? 'rounded' : 'rounded-full'} {selected
							? 'border-signal bg-signal text-accent-contrast'
							: 'border-line-hover text-transparent'}"
						aria-hidden="true"
					>
						<Icon name="check" className="h-3 w-3" strokeWidth={2.5} />
					</span>
					<span class="min-w-0">
						<span class="line-clamp-3 break-words" use:tooltip={{ text: row.option, truncateOnly: true, wrap: true }}
							><HighlightedText text={row.option} {matcher} /></span
						>
						{#if row.reason || mark}
							<span class="mt-1 flex flex-wrap items-center gap-1.5">
								{#if row.reason}
									<span class="text-sm text-fg-subtle">{row.reason}</span>
								{/if}
								{#if mark}
									<span
										class="inline-flex items-center gap-1.5 rounded border px-1.5 font-mono text-sm {selected
											? 'border-warning/35 bg-warning/10 text-warning'
											: 'border-line-strong bg-surface-2 text-fg-muted'}"
										data-testid="option-used"
										use:tooltip={{ text: markTips[row.option] ?? mark, wrap: true }}
									>
										<Icon name="layers" className="h-3 w-3" />
										{mark}
									</span>
								{/if}
							</span>
						{/if}
					</span>
				</div>
			{:else}
				<div class="px-3 py-7 text-center text-base text-fg-subtle" role="status" data-testid="option-picker-empty">
					No options match &ldquo;{query.trim()}&rdquo;
				</div>
			{/each}
		</div>
		<div class="flex flex-shrink-0 items-center gap-3 border-t border-line bg-surface-2/50 px-3 py-2 text-sm text-fg-subtle">
			<div class="hidden items-center gap-3 md:flex">
				<span class="flex items-center gap-1"><Kbd keys="↑↓" /> move</span>
				<span class="flex items-center gap-1"><Kbd keys="↵" /> {multiple ? 'toggle' : 'pick'}</span>
				<span class="flex items-center gap-1"><Kbd keys="Esc" /> close</span>
			</div>
			{#if footerText}
				<span class="ml-auto font-mono tabular-nums" data-testid="option-summary">{footerText}</span>
			{/if}
		</div>
	</AnchoredPanel>
{/if}
