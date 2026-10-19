<script lang="ts">
	import { untrack } from 'svelte';
	import type { Readable } from 'svelte/store';
	import Icon from './Icon.svelte';
	import InlinePopoverChip from './InlinePopoverChip.svelte';
	import NumberStepper from './NumberStepper.svelte';
	import Tooltip from './Tooltip.svelte';
	import { Button, IconButton } from '$lib/components/ui';
	import autosize from '$lib/actions/autosize';
	import { MAX_BRANCHES, MAX_TESTS, MAX_VALUES } from '$lib/utils/promptConditions';
	import {
		conditionIssue,
		conditionTooltip,
		conditionValues,
		conditionVarKind,
		defaultTest,
		draftFromRaw,
		isConditionRaw,
		lastRunFor,
		operatorChoice,
		operatorsFor,
		plainConditionSummary,
		rawFromDraft,
		retargetTest,
		setTestOperator,
		summarizeCondition,
		type BranchDraft,
		type ConditionDraft,
		type TestDraft
	} from '$lib/utils/conditionTags';
	import { normalizeVariableDef, numberDecimals, optionText, type VariableRoll, type VariablesMap } from '$lib/utils/variableDefs';
	import type { ConditionOp } from '$lib/utils/promptConditions';

	let {
		raw,
		context,
		disabled = false,
		variant = 'default',
		startOpen = false,
		onchange,
		onremove,
		onOpenChange,
		onCreateVariable
	}: {
		raw: string;
		context: Readable<{ variables: VariablesMap; rolls: Record<string, VariableRoll> }>;
		disabled?: boolean;
		variant?: 'default' | 'segment-composer';
		startOpen?: boolean;
		onchange?: (newRaw: string) => void;
		onremove?: () => void;
		onOpenChange?: (open: boolean) => void;
		onCreateVariable?: (name: string) => void;
	} = $props();

	let open = $state(untrack(() => startOpen));
	let current = $state(untrack(() => raw));
	let local = $state<ConditionDraft | null>(untrack(() => draftFromRaw(raw)));
	let invalid = $state(false);

	const variables = $derived($context.variables);
	const rolls = $derived($context.rolls);
	const summary = $derived(summarizeCondition(current));
	const issue = $derived(conditionIssue(current, variables));
	const tooltip = $derived(issue ?? conditionTooltip(current));
	const lastRun = $derived(lastRunFor(current, conditionValues(variables, rolls)));
	const variableNames = $derived(Object.keys(variables));
	const missingName = $derived(
		(() => {
			const names = (local?.branches ?? []).flatMap((b) => b.tests.map((t) => t.var));
			return names.find((n) => !(n in variables)) ?? null;
		})()
	);

	$effect(() => {
		onOpenChange?.(open);
	});

	function commit(next: ConditionDraft) {
		local = next;
		const rawNext = rawFromDraft(next, variables);
		invalid = rawNext === null;
		if (rawNext === null || rawNext === current) return;
		current = rawNext;
		onchange?.(rawNext);
	}

	function edit(mutate: (draft: ConditionDraft) => void) {
		if (!local) return;
		const next = JSON.parse(JSON.stringify(local)) as ConditionDraft;
		mutate(next);
		commit(next);
	}

	function setVar(branch: number, test: number, name: string) {
		edit((d) => {
			d.branches[branch].tests[test] = retargetTest(d.branches[branch].tests[test], name, variables);
		});
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
		return def.type === 'choice' ? def.options.map((o) => optionText(o).trim()).filter((o) => o) : [];
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

	function handleRemove() {
		onremove?.();
		open = false;
	}

	function handleKeydown(event: KeyboardEvent) {
		if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
			event.preventDefault();
			open = false;
		}
	}

	const sectionLabel = 'font-mono text-xs uppercase tracking-[0.04em] text-fg-subtle';
	const selectClass = 'input h-8 min-w-0 py-1 text-sm max-md:h-11';
</script>

{#snippet testRow(branch: BranchDraft, bi: number, test: TestDraft, ti: number)}
	{@const kind = conditionVarKind(test.var, variables)}
	{@const ops = operatorsFor(kind)}
	<div class="flex flex-wrap items-start gap-1.5" data-testid="condition-test">
		<select
			class="{selectClass} w-[8.5rem]"
			value={test.var}
			aria-label="Variable"
			onchange={(e) => setVar(bi, ti, e.currentTarget.value)}
		>
			{#if !(test.var in variables)}
				<option value={test.var}>${test.var} (missing)</option>
			{/if}
			{#each variableNames as name (name)}
				<option value={name}>${name}</option>
			{/each}
		</select>
		<select
			class="{selectClass} w-[8.5rem]"
			value={operatorChoice(kind, test.op).op}
			aria-label="Operator"
			onchange={(e) => setOp(bi, ti, e.currentTarget.value as ConditionOp)}
		>
			{#each ops as choice (choice.op)}
				<option value={choice.op}>{choice.glyph} {choice.label}</option>
			{/each}
		</select>
		{#if test.op === 'is empty' || test.op === 'is not empty'}
			<span class="self-center text-xs text-fg-subtle">nothing to enter</span>
		{:else if kind === 'number' && (test.op === 'in' || test.op === 'not in')}
			<div class="flex items-center gap-1">
				<div class="w-[7.5rem]">
					<NumberStepper
						value={Number(test.low)}
						step={numberStep(test.var)}
						decimals={numberPlaces(test.var)}
						label="From"
						onChange={(v) => setNumber(bi, ti, 'low', v)}
					/>
				</div>
				<span class="text-fg-subtle">&ndash;</span>
				<div class="w-[7.5rem]">
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
			<div class="w-[9rem]">
				<NumberStepper
					value={Number(test.number)}
					step={numberStep(test.var)}
					decimals={numberPlaces(test.var)}
					label="Value"
					onChange={(v) => setNumber(bi, ti, 'number', v)}
				/>
			</div>
		{:else if kind === 'choice'}
			<div class="flex min-w-0 flex-1 flex-wrap gap-1" role="group" aria-label="Values">
				{#each optionsOf(test.var) as option (option)}
					<button
						type="button"
						aria-pressed={test.strings.includes(option)}
						class="rounded border px-2 py-1 text-xs font-medium transition-colors max-md:py-2.5 {test.strings.includes(option)
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
				class="{selectClass} min-w-[8rem] flex-1"
				aria-label="Text"
				placeholder="text"
				maxlength="80"
				value={test.strings[0] ?? ''}
				oninput={(e) => setText(bi, ti, e.currentTarget.value)}
			/>
		{/if}
		{#if branch.tests.length > 1}
			<Tooltip text="Remove test">
				<IconButton icon="close" label="Remove test" size="sm" onclick={() => removeTest(bi, ti)} />
			</Tooltip>
		{/if}
	</div>
{/snippet}

<InlinePopoverChip
	{variant}
	kind="condition"
	tone={issue ? 'warning' : 'signal'}
	density="tight"
	{disabled}
	onremove={onremove}
	bind:open
	canOpen={!!local}
	popoverWidth={520}
	class="condition-chip{issue ? ' broken' : ''}"
	removeTitle="Remove condition"
	popoverLabel="Edit condition"
	headerMark="if"
	headerTitle="Condition"
	headerSubtitle={plainConditionSummary(current)}
>
	{#snippet label()}
		<span class="font-mono text-sm {issue ? 'text-warning' : 'text-ai-3'}">if</span>
		{#if summary}
			<Tooltip text={tooltip} wrapperClass="min-w-0 inline-flex items-center gap-1">
			<span class="max-w-[22rem] truncate font-mono text-sm text-fg">{summary.head}</span>
			<span class="text-fg-subtle">&rarr;</span>
			<span class="max-w-[10rem] truncate text-sm text-fg">{summary.then || 'nothing'}</span>
			{#if summary.otherwise !== null}
				<span class="max-w-[9rem] truncate text-sm text-fg-subtle">&middot; else {summary.otherwise || 'nothing'}</span>
			{/if}
			{#if summary.extra > 0}
				<span class="rounded bg-surface-2 px-1 font-mono text-xs tabular-nums text-fg-muted">+{summary.extra}</span>
			{/if}
			</Tooltip>
		{:else}
			<span class="max-w-[22rem] truncate font-mono text-sm text-warning">{current}</span>
		{/if}
		<Icon name="chevron-down" className="w-3 h-3 text-fg-subtle flex-shrink-0" />
	{/snippet}

	{#snippet composerLabel()}
		<span class="chip-mark">if</span>
		{#if summary}
			<Tooltip text={tooltip} wrapperClass="min-w-0 inline-flex items-center">
			<span class="chip-context">{summary.head}</span>
			<span class="chip-label" data-testid="condition-chip-summary">&rarr; {summary.then || 'nothing'}{summary.otherwise !== null ? ` · else ${summary.otherwise || 'nothing'}` : ''}</span>
			{#if summary.extra > 0}
				<span class="chip-state">+{summary.extra}</span>
			{/if}
			</Tooltip>
		{:else}
			<span class="chip-label">{current}</span>
		{/if}
	{/snippet}

	{#snippet popover()}
		{#if local}
			<div class="space-y-2" role="presentation" onkeydown={handleKeydown}>
				{#each local.branches as branch, bi (bi)}
					<div
						class="space-y-2 rounded-lg border border-line bg-surface-2/40 p-2.5"
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
							<div class="space-y-1.5">
								{#each branch.tests as test, ti (ti)}
									{#if ti > 0}
										<div class="{sectionLabel}">{branch.joiner}</div>
									{/if}
									{@render testRow(branch, bi, test, ti)}
								{/each}
								{#if branch.tests.length < MAX_TESTS}
									<Button variant="ghost" size="xs" icon="plus" onclick={() => addTest(bi)}>Add test</Button>
								{/if}
							</div>
						{/if}
						<textarea
							use:autosize={branch.body}
							class="input max-h-28 min-h-[3.25rem] w-full resize-none py-1.5 text-sm leading-5"
							rows="2"
							placeholder={branch.kind === 'else' ? 'Nothing is added otherwise' : 'text added when this is true'}
							aria-label="{branchLabel(branch)} text"
							value={branch.body}
							oninput={(e) => setBody(bi, e.currentTarget.value)}
						></textarea>
					</div>
				{/each}

				<div class="flex flex-wrap items-center gap-2">
					{#if local.branches.length < MAX_BRANCHES}
						<Button variant="secondary" size="xs" icon="plus" onclick={addElseIf}>Else if</Button>
					{/if}
					{#if !local.branches.some((b) => b.kind === 'else') && local.branches.length < MAX_BRANCHES}
						<Button variant="secondary" size="xs" icon="plus" onclick={addOtherwise}>Otherwise</Button>
					{/if}
				</div>

				{#if invalid}
					<p class="text-xs text-warning" role="alert" data-testid="condition-invalid">
						Finish the highlighted values to apply this change. Branch text can't contain another condition.
					</p>
				{/if}
				{#if issue}
					<p class="text-xs text-warning" role="status" data-testid="condition-issue">
						{issue}
						{#if missingName && onCreateVariable}
							<button type="button" class="ml-1 rounded px-1.5 py-0.5 font-medium text-fg underline" onclick={() => onCreateVariable?.(missingName)}>
								Create ${missingName}
							</button>
						{/if}
					</p>
				{/if}

				<div class="flex flex-wrap items-center gap-2 border-t border-line pt-2">
					<span class="{sectionLabel}">Last run</span>
					<span class="min-w-0 flex-1 text-xs text-fg-muted" data-testid="condition-last-run">
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
				<div class="flex items-center gap-2">
					<Button variant="ghost" size="xs" icon="trash" onclick={handleRemove}>Remove condition</Button>
					<span class="ml-auto"></span>
					<Button variant="primary" size="xs" onclick={() => (open = false)}>Done</Button>
				</div>
			</div>
		{/if}
	{/snippet}
</InlinePopoverChip>
