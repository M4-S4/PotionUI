<script lang="ts">
	import { untrack } from 'svelte';
	import BaseModal from './modals/BaseModal.svelte';
	import ConfirmFooter from './modals/ConfirmFooter.svelte';
	import { getConfirmKeyboardAction } from './modals/confirmKeyboard';
	import NumberStepper from './NumberStepper.svelte';
	import Tooltip from './Tooltip.svelte';
	import VariableSelect from './VariableSelect.svelte';
	import { Button, IconButton } from '$lib/components/ui';
	import { MAX_BRANCHES, MAX_TESTS, MAX_VALUES, type ConditionOp } from '$lib/utils/promptConditions';
	import {
		conditionIssue,
		conditionValues,
		conditionVarKind,
		defaultTest,
		draftFromRaw,
		lastRunFor,
		operatorChoice,
		operatorsFor,
		plainConditionSummary,
		rawFromDraft,
		retargetTest,
		setTestOperator,
		type BranchDraft,
		type ConditionDraft
	} from '$lib/utils/conditionTags';
	import { normalizeVariableDef, numberDecimals, optionText, type VariableRoll, type VariablesMap } from '$lib/utils/variableDefs';

	let {
		raw,
		variables,
		rolls = {},
		checkVariables = true,
		onsave,
		onremove,
		oncancel,
		onCreateVariable
	}: {
		raw: string;
		variables: VariablesMap;
		rolls?: Record<string, VariableRoll>;
		checkVariables?: boolean;
		onsave: (raw: string) => void;
		onremove?: () => void;
		oncancel: () => void;
		onCreateVariable?: (name: string) => void;
	} = $props();

	let draft = $state<ConditionDraft>(
		untrack(() => draftFromRaw(raw)) ?? { branches: [{ kind: 'if', joiner: 'and', tests: [], body: '' }] }
	);
	let pendingCreate = $state<{ branch: number; test: number; name: string } | null>(null);

	const nextRaw = $derived(rawFromDraft(draft, variables));
	const invalid = $derived(nextRaw === null);
	const shownRaw = $derived(nextRaw ?? raw);
	const summaryText = $derived(plainConditionSummary(shownRaw));
	const issue = $derived(checkVariables ? conditionIssue(shownRaw, variables) : null);
	const lastRun = $derived(lastRunFor(shownRaw, conditionValues(variables, rolls)));
	const variableNames = $derived(Object.keys(variables));
	const missingName = $derived(
		draft.branches.flatMap((b) => b.tests.map((t) => t.var)).find((n) => !(n in variables)) ?? null
	);

	$effect(() => {
		const pending = pendingCreate;
		if (!pending || !(pending.name in variables)) return;
		pendingCreate = null;
		setVar(pending.branch, pending.test, pending.name);
	});

	function edit(mutate: (d: ConditionDraft) => void) {
		const next = JSON.parse(JSON.stringify(draft)) as ConditionDraft;
		mutate(next);
		draft = next;
	}

	function setVar(branch: number, test: number, name: string) {
		edit((d) => {
			d.branches[branch].tests[test] = retargetTest(d.branches[branch].tests[test], name, variables);
		});
	}

	function createAndTarget(branch: number, test: number, name: string) {
		pendingCreate = { branch, test, name };
		onCreateVariable?.(name);
	}

	function setOp(branch: number, test: number, op: ConditionOp) {
		edit((d) => {
			d.branches[branch].tests[test] = setTestOperator(d.branches[branch].tests[test], op, variables);
		});
	}

	function setNumber(branch: number, test: number, field: 'number' | 'low' | 'high', value: number) {
		edit((d) => {
			const t = d.branches[branch].tests[test];
			t[field] = String(value);
			if (field === 'low' && t.op !== 'in' && t.op !== 'not in') t.number = String(value);
		});
	}

	function setText(branch: number, test: number, text: string) {
		edit((d) => {
			d.branches[branch].tests[test].strings = [text];
		});
	}

	function toggleOption(branch: number, test: number, option: string) {
		edit((d) => {
			const t = d.branches[branch].tests[test];
			if (t.op === 'in' || t.op === 'not in') {
				const has = t.strings.includes(option);
				const next = has ? t.strings.filter((s) => s !== option) : [...t.strings.filter((s) => s !== ''), option];
				t.strings = next.length > 0 ? next.slice(0, MAX_VALUES) : t.strings;
			} else {
				t.strings = [option];
			}
		});
	}

	function setBody(branch: number, body: string) {
		edit((d) => {
			d.branches[branch].body = body;
		});
	}

	function setJoiner(branch: number, joiner: 'and' | 'or') {
		edit((d) => {
			d.branches[branch].joiner = joiner;
		});
	}

	function addTest(branch: number) {
		edit((d) => {
			const b = d.branches[branch];
			if (b.tests.length >= MAX_TESTS) return;
			const name = b.tests[b.tests.length - 1]?.var ?? variableNames[0] ?? 'value';
			b.tests.push(defaultTest(name, variables));
		});
	}

	function removeTest(branch: number, test: number) {
		edit((d) => {
			const b = d.branches[branch];
			if (b.tests.length > 1) b.tests.splice(test, 1);
		});
	}

	function addElseIf() {
		edit((d) => {
			if (d.branches.length >= MAX_BRANCHES) return;
			const lastTest = d.branches.flatMap((b) => b.tests).at(-1);
			const name = lastTest?.var ?? variableNames[0] ?? 'value';
			const branch: BranchDraft = { kind: 'elseif', joiner: 'and', tests: [defaultTest(name, variables)], body: '' };
			const elseAt = d.branches.findIndex((b) => b.kind === 'else');
			if (elseAt === -1) d.branches.push(branch);
			else d.branches.splice(elseAt, 0, branch);
		});
	}

	function addOtherwise() {
		edit((d) => {
			if (!d.branches.some((b) => b.kind === 'else')) d.branches.push({ kind: 'else', joiner: 'and', tests: [], body: '' });
		});
	}

	function removeBranch(index: number) {
		edit((d) => {
			if (d.branches[index]?.kind !== 'if') d.branches.splice(index, 1);
		});
	}

	function optionsOf(name: string): string[] {
		const def = normalizeVariableDef(variables[name]);
		return def.type === 'choice' ? [...new Set(def.options.map((o) => optionText(o).trim()).filter((o) => o))] : [];
	}

	function numberStep(name: string): number {
		const def = normalizeVariableDef(variables[name]);
		return def.type === 'number' ? def.step : 1;
	}

	function numberPlaces(name: string): number {
		const def = normalizeVariableDef(variables[name]);
		return def.type === 'number' ? numberDecimals(def) : 0;
	}

	function branchLabel(branch: BranchDraft): string {
		if (branch.kind === 'if') return 'If';
		return branch.kind === 'elseif' ? 'Else if' : 'Otherwise';
	}

	function save() {
		if (nextRaw === null) return;
		onsave(nextRaw);
	}

	function handleWindowKeydown(e: KeyboardEvent) {
		if (e.target instanceof Element && e.target.closest('[data-chip-layer]')) return;
		if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
			e.preventDefault();
			e.stopPropagation();
			if (!e.repeat) save();
			return;
		}
		const { action, suppress } = getConfirmKeyboardAction(e);
		if (e.key === 'Escape') e.stopPropagation();
		if (action === 'cancel') oncancel();
		else if (action === 'confirm') save();
		if (suppress) e.preventDefault();
	}

	const sectionLabel = 'font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle';
	const fieldClass = 'input h-9 min-w-0 py-1 text-sm max-md:h-11';
</script>

<svelte:window onkeydowncapture={handleWindowKeydown} />

{#snippet testRow(branch: BranchDraft, bi: number, test: BranchDraft['tests'][number], ti: number)}
	{@const kind = conditionVarKind(test.var, variables)}
	{@const ops = operatorsFor(kind)}
	<div class="space-y-2" data-testid="condition-test">
		<div class="flex items-center gap-2">
			<VariableSelect
				class="min-w-0 flex-1"
				{variables}
				value={test.var}
				missing={!(test.var in variables)}
				label="Variable"
				onselect={(name) => setVar(bi, ti, name)}
				oncreate={onCreateVariable ? (name) => createAndTarget(bi, ti, name) : undefined}
			/>
			<select
				class="{fieldClass} min-w-0 flex-1"
				value={operatorChoice(kind, test.op).op}
				aria-label="Operator"
				onchange={(e) => setOp(bi, ti, e.currentTarget.value as ConditionOp)}
			>
				{#each ops as choice (choice.op)}
					<option value={choice.op}>{choice.glyph} {choice.label}</option>
				{/each}
			</select>
			{#if branch.tests.length > 1}
				<Tooltip text="Remove test">
					<IconButton icon="close" label="Remove test" size="sm" onclick={() => removeTest(bi, ti)} />
				</Tooltip>
			{/if}
		</div>
		<div class="min-w-0" data-testid="condition-value">
			{#if test.op === 'is empty' || test.op === 'is not empty'}
				<span class="text-sm text-fg-subtle">Nothing to enter</span>
			{:else if kind === 'number' && (test.op === 'in' || test.op === 'not in')}
				<div class="flex flex-wrap items-center gap-2">
					<div class="w-36">
						<NumberStepper
							value={Number(test.low)}
							step={numberStep(test.var)}
							decimals={numberPlaces(test.var)}
							label="From"
							onChange={(v) => setNumber(bi, ti, 'low', v)}
						/>
					</div>
					<span class="text-fg-subtle">&ndash;</span>
					<div class="w-36">
						<NumberStepper
							value={Number(test.high)}
							step={numberStep(test.var)}
							decimals={numberPlaces(test.var)}
							label="To"
							onChange={(v) => setNumber(bi, ti, 'high', v)}
						/>
					</div>
				</div>
			{:else if kind === 'number' || test.op === '<' || test.op === '<=' || test.op === '>' || test.op === '>='}
				<div class="w-40">
					<NumberStepper
						value={Number(test.number)}
						step={numberStep(test.var)}
						decimals={numberPlaces(test.var)}
						label="Value"
						onChange={(v) => setNumber(bi, ti, 'number', v)}
					/>
				</div>
			{:else if kind === 'choice'}
				<div class="flex min-w-0 flex-wrap gap-1.5" role="group" aria-label="Values">
					{#each optionsOf(test.var) as option (option)}
						<button
							type="button"
							aria-pressed={test.strings.includes(option)}
							class="max-w-full whitespace-normal break-words rounded border px-2.5 py-1.5 text-left text-sm font-medium transition-colors max-md:py-2.5 {test.strings.includes(option)
								? 'border-signal/40 bg-signal/15 text-signal'
								: 'border-line bg-surface-2 text-fg-muted hover:text-fg'}"
							onclick={() => toggleOption(bi, ti, option)}
						>
							{option}
						</button>
					{/each}
				</div>
			{:else}
				<input
					type="text"
					class="{fieldClass} w-full"
					aria-label="Text"
					placeholder="text"
					maxlength="80"
					value={test.strings[0] ?? ''}
					oninput={(e) => setText(bi, ti, e.currentTarget.value)}
				/>
			{/if}
		</div>
	</div>
{/snippet}

<BaseModal
	isOpen={true}
	handleEscapeKey={false}
	labelledBy="condition-modal-title"
	sizeClass="w-full md:w-[45rem] md:max-w-[calc(100vw-2rem)]"
	on:close={oncancel}
>
	<svelte:fragment slot="headerIcon">
		<span class="rounded border border-line bg-surface-3 px-2 py-1 font-mono text-sm font-semibold text-ai-3">if</span>
	</svelte:fragment>
	<svelte:fragment slot="header">
		<div class="min-w-0" role="heading" aria-level="2">
			<strong id="condition-modal-title" class="block text-sm font-semibold text-fg">Condition</strong>
			<span class="block truncate text-sm text-fg-muted" data-testid="condition-summary">{summaryText}</span>
		</div>
	</svelte:fragment>

	<div class="space-y-3 p-5" role="group" aria-label="Edit condition" data-testid="condition-editor-body">
		{#each draft.branches as branch, bi (bi)}
			<div
				class="space-y-3 rounded-lg border border-line bg-surface-2/40 p-3"
				data-testid="condition-branch"
				data-kind={branch.kind}
			>
				<div class="flex items-center gap-2">
					<span class="{sectionLabel} flex-1">{branchLabel(branch)}</span>
					{#if branch.kind !== 'else' && branch.tests.length > 1}
						<div class="inline-flex items-center gap-1 text-xs text-fg-subtle" role="radiogroup" aria-label="Match">
							<span>Match</span>
							{#each [['and', 'all'], ['or', 'any']] as [value, text] (value)}
								<button
									type="button"
									role="radio"
									aria-checked={branch.joiner === value}
									class="rounded px-2 py-1 text-xs font-medium {branch.joiner === value
										? 'bg-surface-3 text-fg'
										: 'text-fg-muted hover:text-fg'}"
									onclick={() => setJoiner(bi, value as 'and' | 'or')}
								>
									{text}
								</button>
							{/each}
						</div>
					{/if}
					{#if branch.kind !== 'if'}
						<Tooltip text="Remove {branchLabel(branch).toLowerCase()}">
							<IconButton icon="trash" label="Remove {branchLabel(branch).toLowerCase()}" size="sm" onclick={() => removeBranch(bi)} />
						</Tooltip>
					{/if}
				</div>
				{#if branch.kind !== 'else'}
					<div class="space-y-3">
						{#each branch.tests as test, ti (ti)}
							{#if ti > 0}
								<div class={sectionLabel}>{branch.joiner}</div>
							{/if}
							{@render testRow(branch, bi, test, ti)}
						{/each}
						{#if branch.tests.length < MAX_TESTS}
							<Button variant="ghost" size="xs" icon="plus" onclick={() => addTest(bi)}>Add test</Button>
						{/if}
					</div>
				{/if}
				<textarea
					class="input min-h-[5rem] w-full resize-y py-2 text-sm leading-5"
					rows="3"
					placeholder={branch.kind === 'else' ? 'Nothing is added otherwise' : 'Text added when this is true'}
					aria-label="{branchLabel(branch)} text"
					value={branch.body}
					oninput={(e) => setBody(bi, e.currentTarget.value)}
				></textarea>
			</div>
		{/each}

		<div class="flex flex-wrap items-center gap-2">
			{#if draft.branches.length < MAX_BRANCHES}
				<Button variant="secondary" size="sm" icon="plus" onclick={addElseIf}>Else if</Button>
			{/if}
			{#if !draft.branches.some((b) => b.kind === 'else') && draft.branches.length < MAX_BRANCHES}
				<Button variant="secondary" size="sm" icon="plus" onclick={addOtherwise}>Otherwise</Button>
			{/if}
		</div>

		{#if invalid}
			<p class="text-sm text-warning" role="alert" data-testid="condition-invalid">
				Finish the highlighted values to save this change. Branch text can't contain another condition.
			</p>
		{/if}
		{#if issue}
			<p class="text-sm text-warning" role="status" data-testid="condition-issue">
				{issue}
				{#if missingName && onCreateVariable}
					<button type="button" class="ml-1 rounded px-1.5 py-0.5 font-medium text-fg underline" onclick={() => onCreateVariable?.(missingName)}>
						Create ${missingName}
					</button>
				{/if}
			</p>
		{/if}

		<div class="flex flex-wrap items-center gap-2 border-t border-line pt-3">
			<span class={sectionLabel}>Last run</span>
			<span class="min-w-0 flex-1 text-sm text-fg-muted" data-testid="condition-last-run">
				{#if lastRun && lastRun.settled}
					{#each lastRun.reads as read (read.name)}
						<span class="font-mono text-fg">${read.name} = {read.value}</span>
					{/each}
					&rarr; {lastRun.body.trim() || 'nothing'}
				{:else}
					Decided when you generate
				{/if}
			</span>
		</div>
	</div>

	<svelte:fragment slot="footer">
		<ConfirmFooter stackOnPhone confirmLabel="Save" confirmDisabled={invalid} onCancel={oncancel} onConfirm={save}>
			{#snippet leftActions()}
				{#if onremove}
					<Button variant="ghost" size="sm" icon="trash" class="text-danger hover:bg-danger/10" onclick={onremove}>
						Remove condition
					</Button>
				{/if}
			{/snippet}
		</ConfirmFooter>
	</svelte:fragment>
</BaseModal>
