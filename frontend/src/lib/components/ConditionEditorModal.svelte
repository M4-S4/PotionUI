<script lang="ts">
	import { untrack } from 'svelte';
	import SegmentModal from './modals/SegmentModal.svelte';
	import NumberStepper from './NumberStepper.svelte';
	import OptionSelect from './OptionSelect.svelte';
	import Tooltip from './Tooltip.svelte';
	import VariableSelect from './VariableSelect.svelte';
	import Icon from './Icon.svelte';
	import { Button, IconButton } from '$lib/components/ui';
	import { MAX_BRANCHES, MAX_STRING_CHARS, MAX_TESTS, MAX_VALUES, type ConditionOp } from '$lib/utils/promptConditions';
	import {
		branchName,
		branchProblemText,
		catchingBranch,
		checkDraft,
		choiceOptions,
		conditionIssue,
		conditionValues,
		conditionVarKind,
		draftFromRaw,
		draftProblemText,
		fitsCondition,
		lastRunFor,
		nestedToElseIf,
		nextBranchTest,
		operatorChoice,
		operatorsFor,
		optionUsage,
		plainConditionSummary,
		retargetTest,
		setTestOperator,
		type BranchDraft,
		type ConditionDraft,
		type DraftProblem
	} from '$lib/utils/conditionTags';
	import { normalizeVariableDef, numberDecimals, type VariableRoll, type VariablesMap } from '$lib/utils/variableDefs';

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

	const OP_WORDS: Partial<Record<ConditionOp, string>> = { '!=': 'is not', 'not in': 'is none of', '=': 'is', in: 'is one of' };

	let draft = $state<ConditionDraft>(
		untrack(() => draftFromRaw(raw)) ?? { branches: [{ kind: 'if', joiner: 'and', tests: [], body: '' }] }
	);
	const initial = untrack(() => JSON.stringify(draft));
	let pendingCreate = $state<{ branch: number; test: number; name: string } | null>(null);

	const checked = $derived(checkDraft(draft, variables));
	const nextRaw = $derived(checked.raw);
	const invalid = $derived(nextRaw === null);
	const dirty = $derived(JSON.stringify(draft) !== initial);
	const canUseElseIf = $derived(nestedToElseIf(draft) !== null);
	const shownRaw = $derived(nextRaw ?? raw);
	const issue = $derived(checkVariables ? conditionIssue(shownRaw, variables) : null);
	const lastRun = $derived(lastRunFor(shownRaw, conditionValues(variables, rolls)));
	const variableNames = $derived(Object.keys(variables));
	const missingName = $derived(
		draft.branches.flatMap((b) => b.tests.map((t) => t.var)).find((n) => !(n in variables)) ?? null
	);
	const switchName = $derived.by(() => {
		const tests = draft.branches.filter((b) => b.kind !== 'else').flatMap((b) => b.tests);
		const first = tests[0]?.var;
		return first && tests.every((t) => t.var === first) && conditionVarKind(first, variables) === 'choice' ? first : null;
	});
	const contextText = $derived(
		switchName
			? `Switch on $${switchName} · ${draft.branches.length} branch${draft.branches.length === 1 ? '' : 'es'}`
			: plainConditionSummary(shownRaw)
	);
	const looseProblems = $derived(checked.problems.filter((p) => p.kind === 'branches' || p.kind === 'grammar'));

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

	function setValues(branch: number, test: number, values: string[]) {
		edit((d) => {
			const t = d.branches[branch].tests[test];
			if (t.op === 'in' || t.op === 'not in') {
				if (values.length > 0) t.strings = values.slice(0, MAX_VALUES);
			} else {
				t.strings = values.slice(0, 1);
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
			b.tests.push(nextBranchTest(d, name, variables));
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
			const branch: BranchDraft = { kind: 'elseif', joiner: 'and', tests: [nextBranchTest(d, name, variables)], body: '' };
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

	function useElseIf() {
		edit((d) => {
			const next = nestedToElseIf(d);
			if (next) d.branches = next.branches;
		});
	}

	function removeBranch(index: number) {
		edit((d) => {
			if (d.branches[index]?.kind !== 'if') d.branches.splice(index, 1);
		});
	}

	function problemsOf(branch: number): DraftProblem[] {
		return checked.problems.filter((p) => 'branch' in p && p.branch === branch);
	}

	function testProblem(branch: number, test: number): DraftProblem | undefined {
		return checked.problems.find((p) => p.kind === 'value' && p.branch === branch && p.test === test);
	}

	function usageMarks(branch: number, name: string): { marks: Record<string, string>; tips: Record<string, string> } {
		const marks: Record<string, string> = {};
		const tips: Record<string, string> = {};
		for (const [option, uses] of optionUsage(draft, branch, name)) {
			const names = uses.map((use) => {
				const word = OP_WORDS[use.op];
				const label = branchName(draft, use.branch);
				return use.op === '=' || use.op === 'in' ? label : `${label} · ${word}`;
			});
			marks[option] = `Used in ${names.join(', ')}`;
			tips[option] = `Used in this condition already: ${uses.map((use) => `${branchName(draft, use.branch)} (${OP_WORDS[use.op]})`).join(' and ')} tests this option.`;
		}
		return { marks, tips };
	}

	function neverRuns(branch: number, test: BranchDraft['tests'][number]): { by: number; value: string } | null {
		if (test.op !== '=' && test.op !== 'in') return null;
		if (conditionVarKind(test.var, variables) !== 'choice') return null;
		for (const value of test.strings) {
			if (!value) continue;
			const by = catchingBranch(draft, branch, test.var, value, variables);
			if (by !== null) return { by, value };
		}
		return null;
	}

	function usedCount(branch: number, name: string): number {
		return optionUsage(draft, branch, name).size;
	}

	function numberStep(name: string): number {
		const def = normalizeVariableDef(variables[name]);
		return def.type === 'number' ? def.step : 1;
	}

	function numberPlaces(name: string): number {
		const def = normalizeVariableDef(variables[name]);
		return def.type === 'number' ? numberDecimals(def) : 0;
	}

	function save() {
		if (nextRaw === null) return false;
		onsave(nextRaw);
	}

	function jumpToProblem() {
		const first = document.querySelector<HTMLElement>('[data-testid="condition-editor-body"] [data-testid="condition-problem"]');
		first?.scrollIntoView({ block: 'center' });
		first?.closest<HTMLElement>('[data-testid="condition-branch"]')?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
	}

	const tooLongHint = `Longer than ${MAX_STRING_CHARS} characters, so it can't be used in a condition`;
	const eyebrow = 'font-mono text-sm uppercase tracking-[0.04em] text-fg-subtle';
	const fieldClass = 'input h-9 min-w-0 py-1 text-base max-md:h-11';
	const problemClass = 'flex items-start gap-1.5 text-sm text-warning';
</script>

{#snippet problemLine(problem: DraftProblem, text: string)}
	<p class={problemClass} data-testid="condition-problem" data-kind={problem.kind}>
		<Icon name="warning" className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
		<span class="min-w-0">{text}</span>
		{#if problem.kind === 'nested' && problem.branch === draft.branches.length - 1 && canUseElseIf}
			<Button variant="secondary" size="sm" class="ml-1 flex-shrink-0" onclick={useElseIf}>Use Else if</Button>
		{/if}
	</p>
{/snippet}

{#snippet testRow(branch: BranchDraft, bi: number, test: BranchDraft['tests'][number], ti: number)}
	{@const kind = conditionVarKind(test.var, variables)}
	{@const ops = operatorsFor(kind)}
	{@const problem = testProblem(bi, ti)}
	{@const flagged = problem !== undefined}
	{@const shadow = neverRuns(bi, test)}
	<div class="space-y-2" data-testid="condition-test" data-invalid={flagged ? 'true' : undefined}>
		<div class="flex flex-wrap items-center gap-2">
			<VariableSelect
				class="w-60 max-md:flex-1"
				{variables}
				value={test.var}
				missing={!(test.var in variables)}
				label="Variable"
				onselect={(name) => setVar(bi, ti, name)}
				oncreate={onCreateVariable ? (name) => createAndTarget(bi, ti, name) : undefined}
			/>
			<select
				class="{fieldClass} w-44 max-md:flex-1"
				value={operatorChoice(kind, test.op).op}
				aria-label="Operator"
				onchange={(e) => setOp(bi, ti, e.currentTarget.value as ConditionOp)}
			>
				{#each ops as choice (choice.op)}
					<option value={choice.op}>{choice.glyph} {choice.label}</option>
				{/each}
			</select>
			{#if branch.tests.length > 1}
				<Tooltip text="Remove test" wrapperClass="ml-auto inline-flex items-center">
					<IconButton icon="close" label="Remove test" size="sm" onclick={() => removeTest(bi, ti)} />
				</Tooltip>
			{/if}
		</div>
		<div class="min-w-0" data-testid="condition-value">
			{#if test.op === 'is empty' || test.op === 'is not empty'}
				<span class="text-base text-fg-subtle">Nothing to enter</span>
			{:else if kind === 'number' && (test.op === 'in' || test.op === 'not in')}
				<div class="flex flex-wrap items-center gap-2">
					<div class="w-36 {flagged ? 'rounded ring-1 ring-warning/60' : ''}">
						<NumberStepper
							value={Number(test.low)}
							step={numberStep(test.var)}
							decimals={numberPlaces(test.var)}
							commitOnInput
							label="From"
							onChange={(v) => setNumber(bi, ti, 'low', v)}
						/>
					</div>
					<span class="text-fg-subtle">&ndash;</span>
					<div class="w-36 {flagged ? 'rounded ring-1 ring-warning/60' : ''}">
						<NumberStepper
							value={Number(test.high)}
							step={numberStep(test.var)}
							decimals={numberPlaces(test.var)}
							commitOnInput
							label="To"
							onChange={(v) => setNumber(bi, ti, 'high', v)}
						/>
					</div>
				</div>
			{:else if kind === 'number' || test.op === '<' || test.op === '<=' || test.op === '>' || test.op === '>='}
				<div class="w-40 {flagged ? 'rounded ring-1 ring-warning/60' : ''}">
					<NumberStepper
						value={Number(test.number)}
						step={numberStep(test.var)}
						decimals={numberPlaces(test.var)}
						commitOnInput
						label="Value"
						onChange={(v) => setNumber(bi, ti, 'number', v)}
					/>
				</div>
			{:else if kind === 'choice'}
				{@const usage = usageMarks(bi, test.var)}
				{@const multiple = test.op === 'in' || test.op === 'not in'}
				{@const used = usedCount(bi, test.var)}
				<OptionSelect
					options={choiceOptions(test.var, variables)}
					value={test.strings.filter((s) => s !== '' && fitsCondition(s))}
					{multiple}
					max={MAX_VALUES}
					label={multiple ? `Values for $${test.var}` : `Value for $${test.var}`}
					invalid={flagged}
					disabledReason={(option) => (fitsCondition(option) ? null : tooLongHint)}
					marks={usage.marks}
					markTips={usage.tips}
					summary={used > 0 ? `${used} used in other branches` : ''}
					onchange={(values) => setValues(bi, ti, values)}
				/>
			{:else}
				<input
					type="text"
					class="{fieldClass} w-full {flagged ? 'border-warning/60' : ''}"
					aria-label="Text"
					aria-invalid={flagged ? 'true' : undefined}
					placeholder="text"
					maxlength={MAX_STRING_CHARS}
					value={test.strings[0] ?? ''}
					oninput={(e) => setText(bi, ti, e.currentTarget.value)}
				/>
			{/if}
		</div>
		{#if problem}
			{@render problemLine(problem, branchProblemText(problem, draft, variables))}
		{/if}
		{#if shadow}
			<p class="flex items-start gap-1.5 text-sm text-fg-muted" data-testid="condition-never-runs">
				<Icon name="warning" className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-warning" />
				<span class="min-w-0">
					{branchName(draft, shadow.by)} already catches &ldquo;<span class="inline-block max-w-[18rem] truncate align-bottom">{shadow.value}</span>&rdquo;, so {branchName(draft, bi)} never runs for it.
				</span>
			</p>
		{/if}
	</div>
{/snippet}

{#snippet contextLine()}
	<span data-testid="condition-summary">{contextText}</span>
{/snippet}

{#snippet footerStart()}
	{#if onremove}
		<Button variant="ghost" size="sm" icon="trash" class="!text-danger hover:bg-danger/10" onclick={onremove}>
			Remove condition
		</Button>
	{/if}
	{#if invalid}
		<span class="h-5 w-px bg-line-strong" aria-hidden="true"></span>
		<Button variant="ghost" size="sm" icon="warning" class="!text-warning hover:bg-warning/10" onclick={jumpToProblem}>
			<span data-testid="condition-invalid">{checked.problems.length} problem{checked.problems.length === 1 ? '' : 's'}</span>
		</Button>
	{/if}
{/snippet}

<SegmentModal
	tier="L"
	title="Condition"
	titleId="condition-modal-title"
	contextContent={contextLine}
	mark="if"
	tone="condition"
	primaryLabel="Save"
	primaryDisabled={invalid}
	submit="ctrl"
	{dirty}
	discardMessage="Your edits to this condition are lost."
	footerStart={footerStart}
	onConfirm={save}
	onCancel={oncancel}
>
	<div class="space-y-2.5 px-5 py-4 max-md:px-4" role="group" aria-label="Edit condition" data-testid="condition-editor-body">
		<p class="sr-only" aria-live="polite">
			{invalid ? checked.problems.map((p) => draftProblemText(p, draft, variables)).join(' ') : ''}
		</p>
		{#each looseProblems as problem, pi (pi)}
			{@render problemLine(problem, draftProblemText(problem, draft, variables))}
		{/each}

		{#each draft.branches as branch, bi (bi)}
			{@const name = branchName(draft, bi)}
			{@const own = problemsOf(bi)}
			{@const nested = own.find((p) => p.kind === 'nested')}
			<article
				class="grid grid-cols-[8.5rem_minmax(0,1fr)] rounded-lg border bg-surface-2/35 max-md:grid-cols-1 {own.length > 0
					? 'border-warning/45'
					: 'border-line'}"
				data-testid="condition-branch"
				data-kind={branch.kind}
				data-invalid={own.length > 0 ? 'true' : undefined}
				aria-label={name}
			>
				<div class="flex flex-col gap-2 border-r border-line py-2.5 pl-3.5 pr-2 max-md:flex-row max-md:items-center max-md:border-b max-md:border-r-0">
					<div class="flex min-h-7 items-center justify-between gap-1 max-md:flex-1 max-md:justify-start max-md:gap-2">
						<span class="font-mono text-sm font-semibold uppercase tracking-[0.05em] {branch.kind === 'else' ? 'text-fg-muted' : 'text-fg'}">
							{name}
						</span>
						{#if branch.kind !== 'if'}
							<Tooltip text="Remove {name.toLowerCase()}" wrapperClass="inline-flex items-center max-md:ml-auto">
								<IconButton icon="trash" label="Remove {name.toLowerCase()}" size="sm" class="hover:text-danger" onclick={() => removeBranch(bi)} />
							</Tooltip>
						{/if}
					</div>
					{#if branch.kind !== 'else' && branch.tests.length > 1}
						<div class="inline-grid grid-cols-2 gap-0.5 self-start rounded border border-line bg-surface-2 p-0.5" role="radiogroup" aria-label="Match">
							{#each [['and', 'All'], ['or', 'Any']] as [value, text] (value)}
								<button
									type="button"
									role="radio"
									aria-checked={branch.joiner === value}
									class="h-7 rounded-sm px-2.5 text-sm font-medium transition-colors {branch.joiner === value
										? 'bg-surface-3 text-fg shadow-raised'
										: 'text-fg-muted hover:text-fg'}"
									onclick={() => setJoiner(bi, value as 'and' | 'or')}
								>
									{text}
								</button>
							{/each}
						</div>
					{/if}
				</div>
				<div class="min-w-0 space-y-2.5 px-3.5 py-3">
					{#if branch.kind !== 'else'}
						{#each branch.tests as test, ti (ti)}
							{#if ti > 0}
								<div class={eyebrow}>{branch.joiner}</div>
							{/if}
							{@render testRow(branch, bi, test, ti)}
						{/each}
						{#each own.filter((p) => p.kind === 'tests' || p.kind === 'head') as problem, pi (pi)}
							{@render problemLine(problem, branchProblemText(problem, draft, variables))}
						{/each}
						{#if branch.tests.length < MAX_TESTS}
							<Button variant="ghost" size="sm" icon="plus" onclick={() => addTest(bi)}>Add test</Button>
						{/if}
					{/if}
					<label class="block">
						<span class="mb-1.5 block {eyebrow}">{branch.kind === 'else' ? 'Otherwise add' : 'Then add'}</span>
						<textarea
							class="input min-h-[3.5rem] w-full resize-y py-2 text-base leading-5 {nested ? 'border-warning/60' : ''}"
							rows="2"
							placeholder={branch.kind === 'else' ? 'Nothing is added otherwise' : 'Text added when this is true'}
							aria-label="{name} text"
							aria-invalid={nested ? 'true' : undefined}
							data-autofocus={bi === 0 ? '' : undefined}
							value={branch.body}
							oninput={(e) => setBody(bi, e.currentTarget.value)}
						></textarea>
					</label>
					{#if nested}
						{@render problemLine(nested, branchProblemText(nested, draft, variables))}
					{/if}
				</div>
			</article>
		{/each}

		<div class="flex flex-wrap items-center gap-2 pt-1">
			{#if draft.branches.length < MAX_BRANCHES}
				<Button variant="secondary" size="sm" icon="plus" onclick={addElseIf}>Else if</Button>
			{/if}
			{#if !draft.branches.some((b) => b.kind === 'else') && draft.branches.length < MAX_BRANCHES}
				<Button variant="secondary" size="sm" icon="plus" onclick={addOtherwise}>Otherwise</Button>
			{/if}
		</div>

		{#if issue}
			<p class="{problemClass} items-center" role="status" data-testid="condition-issue">
				<Icon name="warning" className="h-3.5 w-3.5 flex-shrink-0" />
				<span class="min-w-0">{issue}</span>
				{#if missingName && onCreateVariable}
					<Button variant="secondary" size="sm" class="flex-shrink-0" onclick={() => onCreateVariable?.(missingName)}>
						Create ${missingName}
					</Button>
				{/if}
			</p>
		{/if}

		<div class="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 rounded-lg border border-dashed border-line-strong px-3 py-2.5 text-sm text-fg-muted">
			<span class="{eyebrow} flex-shrink-0">Last run</span>
			<span class="min-w-0 flex-1 break-words" data-testid="condition-last-run">
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
</SegmentModal>
