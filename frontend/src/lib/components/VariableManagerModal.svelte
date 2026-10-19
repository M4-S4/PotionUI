<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import BaseModal from './modals/BaseModal.svelte';
	import ConfirmFooter from './modals/ConfirmFooter.svelte';
	import { createConfirmSettlementGate, getConfirmKeyboardAction } from './modals/confirmKeyboard';
	import { Button } from '$lib/components/ui';
	import autosize from '$lib/actions/autosize';
	import Icon from './Icon.svelte';
	import Tooltip from './Tooltip.svelte';
	import VariableConditionChip from './VariableConditionChip.svelte';
	import NumberVariableEditor from './NumberVariableEditor.svelte';
	import PasteVariablesModal from './PasteVariablesModal.svelte';
	import { isValidVariableName } from '$lib/utils/promptVariables';
	import { toasts } from '$lib/stores/toast';
	import { copyText } from '$lib/utils/clipboard';
	import { serializeVariables } from '$lib/utils/variablesTransfer';
	import { applyVariableEdits } from '$lib/utils/variableEdits';
	import {
		normalizeVariableDef,
		createTextVariable,
		createChoiceVariable,
		createNumberVariable,
		clampNumberValue,
		numberRangeLabel,
		optionText,
		optionWhen,
		dependencies,
		dependents,
		resolveVariableOrder,
		eligibleOptions,
		buildVariablesForSubmit,
		isVariableLinked,
		usagesShareRoll,
		isWhenTarget,
		type ChoiceVariableMode,
		type ConditionReaders,
		type NumberVariableDef,
		type VariableDef,
		type VariableOption,
		type VariableOptionWhen,
		type VariablesMap
	} from '$lib/utils/variableDefs';

	// Manager for the tab's prompt variables. A variable has a TYPE: `text` (a
	// plain value) or `choice` (a managed list of options, edited as add/remove/edit
	// rows, never a raw `{a|b|c}` box). A choice variable shuffles (random option
	// per image) unless one option is pinned; either way it serializes to the same
	// `{a|b|c}` wire syntax a hand-typed group would (see utils/variableDefs.ts).
	//
	// Edits a local draft and commits the whole map back on every change — no
	// separate "save" step, matching the rest of the tab's live state.

	export let isOpen = false;
	export let variables: VariablesMap = {};
	export let usageCounts: Record<string, number> = {};
	export let readers: ConditionReaders = {};

	const dispatch = createEventDispatcher<{ close: void; change: VariablesMap }>();

	interface Row {
		key: string; // stable local key so bindings don't jump around as `name` is edited
		name: string;
		def: VariableDef;
	}

	const MODE_CHOICES: { mode: ChoiceVariableMode; label: string }[] = [
		{ mode: 'shuffle', label: 'Shuffle' },
		{ mode: 'pin', label: 'Pin' },
		{ mode: 'per-image', label: 'Per image' }
	];
	const USAGE_CHOICES = [
		{ linked: false, label: 'Independent' },
		{ linked: true, label: 'Linked' }
	];

	let rows: Row[] = [];
	let selectedKey: string | null = null;
	let listQuery = '';
	let mobileView: 'list' | 'detail' = 'list';
	let previousOpen = false;
	let nextKey = 0;
	let openedWith: VariablesMap = {};
	let committedEdits: VariablesMap = {};
	const settlementGate = createConfirmSettlementGate();

	$: if (isOpen !== previousOpen) {
		previousOpen = isOpen;
		if (isOpen) {
			listQuery = '';
			mobileView = 'list';
			initialize();
			settlementGate.reset();
		}
	}

	function handleCancel() {
		settlementGate.settle(() => {
			dispatch('change', applyVariableEdits(variables, committedEdits, openedWith));
			dispatch('close');
		});
	}

	function handleConfirm() {
		settlementGate.settle(() => dispatch('close'));
	}

	function handleKeydown(event: KeyboardEvent) {
		if (!isOpen) return;
		const { action, suppress } = getConfirmKeyboardAction(event);
		if (action === 'cancel') handleCancel();
		else if (action === 'confirm') handleConfirm();
		if (suppress) event.preventDefault();
	}

	function initialize() {
		rows = Object.entries(variables).map(([name, stored]) => ({
			key: `v${nextKey++}`,
			name,
			def: normalizeVariableDef(stored)
		}));
		committedEdits = JSON.parse(JSON.stringify(Object.fromEntries(rows.map((row) => [row.name, row.def]))));
		openedWith = JSON.parse(JSON.stringify(committedEdits));
		if (rows.length === 0) rows = [{ key: `v${nextKey++}`, name: '', def: createTextVariable() }];
		selectedKey = rows[0].key;
	}

	let pasteModalOpen = false;

	async function copyVariables() {
		const count = Object.keys(liveVariables).length;
		const ok = await copyText(serializeVariables(liveVariables));
		if (ok) toasts.success(`Copied ${count} variable${count === 1 ? '' : 's'}`);
		else toasts.error('Could not copy to clipboard');
	}

	function importVariables(merged: VariablesMap) {
		rows = Object.entries(merged).map(([name, stored]) => ({
			key: `v${nextKey++}`,
			name,
			def: normalizeVariableDef(stored)
		}));
		if (rows.length === 0) rows = [{ key: `v${nextKey++}`, name: '', def: createTextVariable() }];
		selectedKey = rows[0].key;
		commit();
	}

	function handlePasteConfirm(event: CustomEvent<{ variables: VariablesMap; count: number }>) {
		importVariables(event.detail.variables);
		pasteModalOpen = false;
		toasts.success(`Imported ${event.detail.count} variable${event.detail.count === 1 ? '' : 's'}`);
	}

	// Per-row name-validation error, keyed by `row.key` (the `{#each}` below is keyed
	// by `row.key`). Renaming one row can flip the error state of a DIFFERENT row
	// (the one it now collides with) whose own object reference didn't change — this
	// is a `$:` statement (not a plain function called from `{@const}`) so Svelte's
	// dependency scan sees `rows` directly and recomputes every row's error whenever
	// ANY row's name changes; a function call would only recompute the row whose own
	// reference changed, leaving the collision partner's error stale.
	$: nameErrorByKey = new Map(
		rows.map((row) => {
			if (!row.name) return [row.key, null]; // an empty draft row isn't an error yet
			if (!isValidVariableName(row.name)) {
				return [row.key, 'Letters, numbers, underscore; must not start with a number.'];
			}
			const dupes = rows.filter((r) => r.name === row.name);
			if (dupes.length > 1) return [row.key, 'Duplicate name — only the last one would be used.'];
			return [row.key, null];
		})
	);

	$: liveVariables = Object.fromEntries(
		rows.filter((r) => r.name.trim()).map((r) => [r.name.trim(), r.def])
	) as VariablesMap;

	$: selected = rows.find((r) => r.key === selectedKey) ?? null;

	$: visibleRows = (() => {
		const needle = listQuery.trim().toLowerCase();
		if (!needle) return rows;
		return rows.filter(
			(r) =>
				r.name.toLowerCase().includes(needle) ||
				(r.def.type === 'choice' && r.def.options.some((o) => optionText(o).toLowerCase().includes(needle)))
		);
	})();

	$: anyCondition = rows.some(
		(r) => r.def.type === 'choice' && r.def.options.some((o) => !!optionWhen(o))
	);
	$: anyNumber = rows.some((r) => r.def.type === 'number');
	$: showRollPreview = anyCondition || anyNumber;

	interface RollPreviewEntry {
		name: string;
		value: string;
		note: string;
		warn: boolean;
	}

	let rollPreview: RollPreviewEntry[] | null = null;
	$: if (!showRollPreview) rollPreview = null;

	function runRollPreview() {
		const { wireMap, rolls } = buildVariablesForSubmit(liveVariables);
		const resolved: Record<string, string> = {};
		const entries: RollPreviewEntry[] = [];

		for (const name of resolveVariableOrder(liveVariables)) {
			const stored = liveVariables[name];
			if (stored === undefined) continue;
			const def = normalizeVariableDef(stored);

			if (def.type === 'text') {
				resolved[name] = (def.value ?? '').trim();
				continue;
			}
			if (def.type === 'number') {
				const numberValue = wireMap[name];
				if (numberValue !== undefined && !numberValue.startsWith('{')) resolved[name] = numberValue;
				entries.push({
					name,
					value: numberValue !== undefined && numberValue.startsWith('{') ? 'per image' : (numberValue ?? ''),
					note: '',
					warn: numberValue === undefined
				});
				continue;
			}
			if (!def.options.some((o) => optionText(o).trim().length > 0)) continue;

			const value = wireMap[name];
			if (value !== undefined) resolved[name] = value;

			let note = '';
			let warn = false;

			if (def.mode === 'shuffle') {
				const conditionedTotal = def.options.filter(
					(o) => optionText(o).trim().length > 0 && !!optionWhen(o)
				).length;
				if (conditionedTotal > 0) {
					const roll = rolls[name];
					if (!roll || (roll.eligible ?? 0) === 0) {
						note = 'no option eligible and none marked Always — will expand to nothing';
						warn = true;
					} else {
						const conditionedMatched = eligibleOptions(def, resolved).filter((o) => !!optionWhen(o)).length;
						if (conditionedMatched === 0) {
							note = 'no conditioned match — fell back to Always';
							warn = true;
						} else {
							note = `${roll.eligible} of ${roll.total} eligible`;
						}
					}
				}
			}

			entries.push({ name, value: value ?? '', note, warn: warn || value === undefined });
		}

		rollPreview = entries;
	}

	function commit() {
		const next: VariablesMap = {};
		for (const row of rows) {
			const name = row.name.trim();
			if (!name || !isValidVariableName(name)) continue;
			next[name] = row.def;
		}
		const merged = applyVariableEdits(variables, committedEdits, next);
		committedEdits = JSON.parse(JSON.stringify(next));
		variables = merged;
		dispatch('change', merged);
	}

	function addRow() {
		const key = `v${nextKey++}`;
		rows = [...rows, { key, name: '', def: createTextVariable() }];
		listQuery = '';
		selectRow(key);
	}

	function selectRow(key: string) {
		selectedKey = key;
		mobileView = 'detail';
	}

	function rowMeta(row: Row): string {
		if (row.def.type === 'text') return 'text';
		if (row.def.type === 'number') {
			const numberUses = usageCounts[row.name.trim()] ?? 0;
			const lead = row.def.mode === 'pin' ? clampNumberValue(row.def) : numberRangeLabel(row.def);
			const numberParts = [lead, row.def.mode];
			if (numberUses > 0) numberParts.push(`${numberUses} use${numberUses === 1 ? '' : 's'}`);
			return numberParts.join(' · ');
		}
		const uses = usageCounts[row.name.trim()] ?? 0;
		const parts = [`${row.def.options.length} opt`, row.def.mode];
		if (uses > 0) parts.push(`${uses} use${uses === 1 ? '' : 's'}`);
		return parts.join(' · ');
	}

	function removeRow(key: string) {
		const removed = rows.find((r) => r.key === key);
		const removedAt = rows.findIndex((r) => r.key === key);
		rows = rows.filter((r) => r.key !== key);
		if (rows.length === 0) rows = [{ key: `v${nextKey++}`, name: '', def: createTextVariable() }];
		if (selectedKey === key) selectedKey = rows[Math.min(removedAt, rows.length - 1)].key;
		mobileView = 'list';

		if (removed?.name.trim()) {
			const removedName = removed.name.trim();
			let clearedCount = 0;
			rows = rows.map((r) => {
				if (r.def.type !== 'choice') return r;
				let changed = false;
				const options = r.def.options.map((o) => {
					const when = optionWhen(o);
					if (when?.var !== removedName) return o;
					changed = true;
					clearedCount += 1;
					return optionText(o);
				});
				return changed ? { ...r, def: { ...r.def, options } } : r;
			});
			if (clearedCount > 0) {
				toasts.warning(
					clearedCount === 1
						? `Removing $${removedName} cleared 1 condition that depended on it.`
						: `Removing $${removedName} cleared ${clearedCount} conditions that depended on it.`
				);
			}
		}

		commit();
	}

	function updateName(key: string, name: string) {
		rows = rows.map((r) => (r.key === key ? { ...r, name } : r));
		commit();
	}

	function setType(key: string, type: 'text' | 'choice' | 'number') {
		rows = rows.map((r) => {
			if (r.key !== key || r.def.type === type) return r;
			const def = type === 'text' ? createTextVariable() : type === 'number' ? createNumberVariable() : createChoiceVariable();
			return { ...r, def };
		});
		commit();
	}

	function updateNumberDef(key: string, def: NumberVariableDef) {
		rows = rows.map((r) => (r.key === key && r.def.type === 'number' ? { ...r, def } : r));
		commit();
	}

	function updateTextValue(key: string, value: string) {
		rows = rows.map((r) => (r.key === key && r.def.type === 'text' ? { ...r, def: { ...r.def, value } } : r));
		commit();
	}

	function updateOptionText(key: string, index: number, text: string) {
		rows = rows.map((r) => {
			if (r.key !== key || r.def.type !== 'choice') return r;
			return {
				...r,
				def: {
					...r.def,
					options: r.def.options.map((o, i) => {
						if (i !== index) return o;
						const when = optionWhen(o);
						return when ? { text, when } : text;
					})
				}
			};
		});
		commit();
	}

	function updateOptionWhen(key: string, index: number, when: VariableOptionWhen | undefined) {
		rows = rows.map((r) => {
			if (r.key !== key || r.def.type !== 'choice') return r;
			return {
				...r,
				def: {
					...r.def,
					options: r.def.options.map((o, i): VariableOption => {
						if (i !== index) return o;
						const text = optionText(o);
						return when ? { text, when } : text;
					})
				}
			};
		});
		commit();
	}

	function addOption(key: string) {
		rows = rows.map((r) =>
			r.key === key && r.def.type === 'choice' ? { ...r, def: { ...r.def, options: [...r.def.options, ''] } } : r
		);
		commit();
	}

	function removeOption(key: string, index: number) {
		rows = rows.map((r) => {
			if (r.key !== key || r.def.type !== 'choice') return r;
			const options = r.def.options.filter((_, i) => i !== index);
			// Keep the pin pointing at the same option (or clear it if that's the
			// one being removed) rather than silently repinning to whatever now
			// sits at the old index.
			let pinnedIndex = r.def.pinnedIndex;
			if (pinnedIndex !== null) {
				if (pinnedIndex === index) pinnedIndex = null;
				else if (pinnedIndex > index) pinnedIndex -= 1;
			}
			return { ...r, def: { ...r.def, options, pinnedIndex } };
		});
		commit();
	}

	function setMode(key: string, mode: ChoiceVariableMode) {
		rows = rows.map((r) => {
			if (r.key !== key || r.def.type !== 'choice') return r;
			// Switching into pin mode with nothing pinned yet defaults to the
			// first option, so the picker below always shows a real selection.
			const pinnedIndex = mode === 'pin' ? (r.def.pinnedIndex ?? 0) : r.def.pinnedIndex;
			return { ...r, def: { ...r.def, mode, pinnedIndex } };
		});
		commit();
	}

	function setLinked(key: string, linked: boolean) {
		rows = rows.map((r) => (r.key === key && r.def.type === 'choice' ? { ...r, def: { ...r.def, linked } } : r));
		commit();
	}

	function setPinnedIndex(key: string, pinnedIndex: number) {
		rows = rows.map((r) => (r.key === key && r.def.type === 'choice' ? { ...r, def: { ...r.def, pinnedIndex } } : r));
		commit();
	}

</script>

<svelte:window on:keydown|capture={handleKeydown} />

<BaseModal
	{isOpen}
	title="Prompt variables"
	sizeClass="md:w-[min(94vw,80rem)]"
	handleEscapeKey={false}
	on:close={handleCancel}
>
	<svelte:fragment slot="headerIcon"><Icon name="braces" className="h-5 w-5 text-fg-muted" /></svelte:fragment>
	<div class="space-y-3 p-4 sm:p-6">
		<p class="text-xs text-fg-subtle">
			Defined here, used in any segment as <code class="rounded bg-surface-2 px-1 py-0.5 font-mono text-xs text-fg">${'{'}name{'}'}</code>.
			A <span class="text-fg">Choice</span> variable picks a new option each time you click Generate — same idea as a
			phrasebook chip's shuffle. A <span class="text-fg">Number</span> variable rolls a value between a minimum and a maximum. Shared by every segment of this tab, for the life of the session.
		</p>

		<div class="flex min-h-0 flex-col gap-3 md:h-[62vh] md:flex-row">
			<div
				class="min-h-0 flex-shrink-0 flex-col rounded-lg border border-line bg-surface-1 md:flex md:w-[280px] {mobileView === 'list' ? 'flex' : 'hidden'}"
			>
				<div class="space-y-2 border-b border-line p-2">
					<label class="flex h-8 items-center gap-2 rounded border border-line bg-surface-2 px-2.5 text-fg-subtle focus-within:border-signal">
						<Icon name="search" className="h-3.5 w-3.5 flex-shrink-0" />
						<input
							type="text"
							class="min-w-0 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-fg-subtle"
							placeholder="Filter variables"
							bind:value={listQuery}
							aria-label="Filter variables"
						/>
					</label>
					<div class="flex items-center gap-2 px-0.5">
						<span class="rounded bg-surface-2 px-1.5 font-mono text-xs tabular-nums text-fg-muted">{rows.length}</span>
						<span class="text-xs text-fg-subtle">variables</span>
					</div>
				</div>
				<div class="min-h-0 flex-1 overflow-y-auto p-1.5 max-md:max-h-[50vh]" role="listbox" aria-label="Variables">
					{#each visibleRows as row (row.key)}
						{@const error = nameErrorByKey.get(row.key) ?? null}
						{@const used = usageCounts[row.name.trim()] ?? 0}
						<button
							type="button"
							role="option"
							aria-selected={row.key === selectedKey}
							class="flex w-full flex-col gap-0.5 rounded border-l-2 px-2.5 py-1.5 text-left transition-colors {row.key === selectedKey
								? 'border-signal bg-signal/10'
								: 'border-transparent hover:bg-surface-2'}"
							on:click={() => selectRow(row.key)}
						>
							<span class="flex min-w-0 items-center gap-1.5">
								<span class="truncate font-mono text-sm {error ? 'text-danger' : 'text-fg'}">{row.name ? `$${row.name}` : 'unnamed'}</span>
								{#if used > 0}
									<Tooltip text={`Used in the prompt · ${used}`}>
										<span class="inline-flex flex-shrink-0 text-success" role="img" aria-label={`Used in the prompt ${used} ${used === 1 ? 'time' : 'times'}`} data-testid="variable-used-tick">
											<Icon name="check" className="h-3.5 w-3.5" strokeWidth={3} />
										</span>
									</Tooltip>
								{/if}
							</span>
							<span class="flex min-w-0 items-center gap-1.5">
							<span class="min-w-0 flex-1 truncate font-mono text-xs tabular-nums text-fg-subtle">{rowMeta(row)}</span>
							<span class="flex-shrink-0 rounded bg-surface-2 px-1 font-mono text-xs text-fg-subtle" aria-label={`${row.def.type} variable`}>
								{row.def.type === 'number' ? '123' : row.def.type === 'text' ? 'Aa' : '[ ]'}
							</span>
						</span>
						</button>
					{:else}
						<p class="px-2.5 py-3 text-xs text-fg-subtle">No variables match.</p>
					{/each}
				</div>
			</div>

			<div class="min-h-0 min-w-0 flex-1 overflow-y-auto md:block {mobileView === 'detail' ? 'block' : 'hidden'}">
				{#if selected}
					{@const row = selected}
					{@const error = nameErrorByKey.get(row.key) ?? null}
					<div class="rounded-lg border border-line bg-surface-2/30 p-3">
						<div class="space-y-3">
							<div class="flex flex-wrap items-center gap-2">
								<button
									type="button"
									class="inline-flex h-8 items-center gap-1 rounded px-1.5 text-xs font-medium text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg md:hidden"
									on:click={() => (mobileView = 'list')}
								>
									<Icon name="chevron-left" className="h-3.5 w-3.5" />
									Variables
								</button>
								<input
									type="text"
									class="input w-44 font-mono text-sm {error ? 'border-danger' : ''}"
									placeholder="name"
									value={row.name}
									on:input={(e) => updateName(row.key, e.currentTarget.value)}
									aria-invalid={!!error}
									aria-label="Variable name"
								/>

								<div class="inline-flex rounded bg-surface-2 p-0.5" role="tablist" aria-label="Variable type">
									<button
										type="button"
										role="tab"
										aria-selected={row.def.type === 'text'}
										class="rounded px-2.5 py-1 text-xs font-medium transition-colors {row.def.type === 'text'
											? 'bg-surface-1 text-fg shadow-raised'
											: 'text-fg-muted hover:text-fg'}"
										on:click={() => setType(row.key, 'text')}
									>
										Text
									</button>
									<button
										type="button"
										role="tab"
										aria-selected={row.def.type === 'choice'}
										class="rounded px-2.5 py-1 text-xs font-medium transition-colors {row.def.type === 'choice'
											? 'bg-signal/15 text-signal'
											: 'text-fg-muted hover:text-fg'}"
										on:click={() => setType(row.key, 'choice')}
									>
										Choice
									</button>
									<button
										type="button"
										role="tab"
										aria-selected={row.def.type === 'number'}
										class="rounded px-2.5 py-1 text-xs font-medium transition-colors {row.def.type === 'number'
											? 'bg-signal/15 text-signal'
											: 'text-fg-muted hover:text-fg'}"
										on:click={() => setType(row.key, 'number')}
									>
										Number
									</button>
								</div>

								{#if row.def.type === 'choice'}
									{@const deps = dependencies(row.name.trim(), liveVariables)}
									{@const usedBy = dependents(row.name.trim(), liveVariables)}
									{#if deps.length > 0}
										<span
											class="inline-flex items-center gap-1 rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle"
										>
											after
											<span class="text-signal normal-case">{deps.map((d) => `$${d}`).join(', ')}</span>
										</span>
									{/if}
									{#if usedBy.length > 0}
										<span
											class="inline-flex items-center gap-1 rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle"
										>
											used by
											<span class="text-signal normal-case">{usedBy.map((d) => `$${d}`).join(', ')}</span>
										</span>
									{/if}
								{/if}

								<Tooltip text="Remove">
									<button
										type="button"
										class="ml-auto inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded text-fg-muted transition-colors hover:bg-surface-2 hover:text-danger"
										on:click={() => removeRow(row.key)}
										aria-label={`Remove variable ${row.name || ''}`.trim()}
									>
										<Icon name="trash" className="h-4 w-4" />
									</button>
								</Tooltip>
							</div>

							{#if error}
								<p class="text-xs text-danger">{error}</p>
							{/if}

							{#if row.def.type === 'text'}
								<input
									type="text"
									class="input w-full text-sm"
									placeholder="value"
									value={row.def.value}
									on:input={(e) => updateTextValue(row.key, e.currentTarget.value)}
									aria-label="Variable value"
								/>
							{:else if row.def.type === 'number'}
								<NumberVariableEditor
									name={row.name.trim()}
									def={row.def}
									variables={liveVariables}
									usageCount={usageCounts[row.name.trim()] ?? 0}
									readers={readers[row.name.trim()] ?? []}
									onChange={(next) => updateNumberDef(row.key, next)}
								/>
							{:else}
								{@const whenTarget = isWhenTarget(row.name.trim(), liveVariables)}
								{@const usageCount = usageCounts[row.name.trim()] ?? 0}
								{@const usagesLocked = whenTarget || row.def.mode !== 'shuffle'}
								<div class="grid gap-3 sm:grid-cols-2">
									<div class="space-y-1.5">
										<span class="font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle">Value</span>
										<div class="grid grid-cols-3 gap-0.5 rounded bg-surface-2 p-0.5" role="radiogroup" aria-label={`How ${row.name || 'this variable'} picks a value`}>
											{#each MODE_CHOICES as choice (choice.mode)}
												<button
													type="button"
													role="radio"
													aria-checked={row.def.mode === choice.mode}
													class="rounded px-2 py-1 text-xs font-medium transition-colors {row.def.mode === choice.mode
														? 'bg-surface-1 text-fg shadow-raised'
														: 'text-fg-muted hover:text-fg'}"
													on:click={() => setMode(row.key, choice.mode)}
												>
													{choice.label}
												</button>
											{/each}
										</div>
									</div>
									<div class="space-y-1.5">
										<span class="font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle">Usages{usageCount > 0 ? ` · ${usageCount}` : ''}</span>
										<div class="grid grid-cols-2 gap-0.5 rounded bg-surface-2 p-0.5" role="radiogroup" aria-label={`Whether each usage of ${row.name || 'this variable'} rolls separately`}>
											{#each USAGE_CHOICES as choice (choice.label)}
												<button
													type="button"
													role="radio"
													aria-checked={usagesShareRoll(row.name.trim(), liveVariables, readers) === choice.linked}
													disabled={usagesLocked}
													class="rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 {usagesShareRoll(row.name.trim(), liveVariables, readers) === choice.linked
														? 'bg-surface-1 text-fg shadow-raised'
														: 'text-fg-muted hover:text-fg'}"
													on:click={() => setLinked(row.key, choice.linked)}
												>
													{choice.label}
												</button>
											{/each}
										</div>
									</div>
								</div>
								<p class="text-xs text-fg-subtle">
									{#if whenTarget}
										Other options depend on this variable, so every usage shares one roll.
									{:else if row.def.mode === 'pin'}
										Always uses the pinned option; pick it with the radio on the left.
									{:else if row.def.mode === 'per-image'}
										Rolls again for every image, and every usage rolls on its own, so usages can't be linked in this mode. Check the result card to see what each image got.
									{:else if isVariableLinked(row.name.trim(), liveVariables, readers)}
										Every usage shares one roll per generation.
									{:else}
										Each usage rolls its own value per generation.
									{/if}
								</p>

								<div class="space-y-1.5">
									<div class="flex items-center gap-1.5 px-0.5">
										<span class="min-w-0 flex-1 font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle">Options · {row.def.options.length}</span>
										<span class="hidden w-[190px] flex-shrink-0 font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle md:block">When</span>
										<span class="hidden w-7 flex-shrink-0 md:block"></span>
									</div>
									{#each row.def.options as option, index (index)}
										<div class="flex flex-wrap items-start gap-1.5 md:flex-nowrap">
											{#if row.def.mode === 'pin'}
												<button
													type="button"
													role="radio"
													aria-checked={(row.def.pinnedIndex ?? 0) === index}
													aria-label={`Pin option ${index + 1}`}
													class="mt-2 grid h-4 w-4 flex-shrink-0 place-items-center rounded-full border {(row.def.pinnedIndex ?? 0) === index
														? 'border-signal'
														: 'border-line-hover'}"
													on:click={() => setPinnedIndex(row.key, index)}
												>
													{#if (row.def.pinnedIndex ?? 0) === index}
														<span class="h-2 w-2 rounded-full bg-signal"></span>
													{/if}
												</button>
											{:else}
												<span class="mt-1.5 w-5 flex-shrink-0 text-right font-mono text-xs tabular-nums text-fg-subtle">{index + 1}</span>
											{/if}
											<textarea
												use:autosize={optionText(option)}
												class="input min-w-0 flex-1 resize-none py-1 text-sm leading-5"
												rows="2"
												placeholder={`option ${index + 1}`}
												value={optionText(option)}
												on:input={(e) => updateOptionText(row.key, index, e.currentTarget.value)}
												aria-label={`Option ${index + 1}`}
											></textarea>
											<div class="w-full flex-shrink-0 md:w-[190px]">
												<VariableConditionChip
													ownerName={row.name.trim()}
													when={optionWhen(option)}
													mode={row.def.mode}
													variables={liveVariables}
													onChange={(when) => updateOptionWhen(row.key, index, when)}
												/>
											</div>
											<button
												type="button"
												class="inline-flex h-7 w-7 flex-shrink-0 items-center justify-center rounded text-fg-muted transition-colors hover:bg-surface-3 hover:text-danger"
												on:click={() => removeOption(row.key, index)}
												aria-label={`Remove option ${index + 1}`}
											>
												<Icon name="trash" className="h-3.5 w-3.5" />
											</button>
										</div>
									{/each}
								</div>

								<button
									type="button"
									class="inline-flex items-center gap-1 rounded px-1.5 py-1 text-xs font-medium text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg"
									on:click={() => addOption(row.key)}
								>
									<Icon name="plus" className="h-3.5 w-3.5" />
									Add option
								</button>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		</div>

		<div class="flex flex-wrap items-center gap-2">
			<Button variant="secondary" size="xs" icon="plus" onclick={addRow}>Add variable</Button>
		</div>
	</div>

	<PasteVariablesModal
		isOpen={pasteModalOpen}
		existing={liveVariables}
		on:close={() => (pasteModalOpen = false)}
		on:confirm={handlePasteConfirm}
	/>
	<svelte:fragment slot="footer">
		<ConfirmFooter confirmLabel="Done" stackOnPhone onCancel={handleCancel} onConfirm={handleConfirm}>
			{#snippet leftActions()}
				<Tooltip text="Copy variables as JSON">
					<Button
						variant="secondary"
						size="xs"
						icon="copy"
						disabled={Object.keys(liveVariables).length === 0}
						onclick={copyVariables}
					>
						Copy
					</Button>
				</Tooltip>
				<Tooltip text="Paste variables JSON">
					<Button variant="secondary" size="xs" icon="clipboard-list" onclick={() => (pasteModalOpen = true)}>
						Paste
					</Button>
				</Tooltip>
				{#if showRollPreview}
					<span class="mx-1 h-4 w-px flex-shrink-0 bg-line-strong/60 max-sm:hidden"></span>
					<Button variant="secondary" size="xs" onclick={runRollPreview}>Roll preview</Button>
					{#each rollPreview ?? [] as entry, i (entry.name)}
						{#if i > 0}<Icon name="chevron-right" className="h-3 w-3 flex-shrink-0 text-fg-subtle" />{/if}
						<span class="inline-flex min-w-0 flex-wrap items-center gap-1 font-mono text-xs">
							<span class="whitespace-nowrap text-fg-subtle">${entry.name}</span>
							<span class="break-words {entry.warn ? 'text-warning' : 'text-fg'}">{entry.value || '∅'}</span>
							{#if entry.note}
								<span class="text-xs {entry.warn ? 'text-warning' : 'text-fg-subtle'}">{entry.note}</span>
							{/if}
						</span>
					{/each}
				{/if}
			{/snippet}
		</ConfirmFooter>
	</svelte:fragment>
</BaseModal>
