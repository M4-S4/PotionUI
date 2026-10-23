<script lang="ts">
	import Tooltip from './Tooltip.svelte';
	import NumberStepper from './NumberStepper.svelte';
	import { Button } from '$lib/components/ui';
	import {
		MAX_NUMBER_GRID,
		clampNumberValue,
		decimalPlaces,
		numberGridIssue,
		numberDecimals,
		numberGridSize,
		numberGridValue,
		numberGridValues,
		rollNumberValue,
		seededRandom,
		usagesShareRoll,
		type ChoiceVariableMode,
		type ConditionReader,
		type NumberTrackRange,
		type NumberVariableDef,
		type VariablesMap
	} from '$lib/utils/variableDefs';

	let {
		name,
		def,
		variables,
		usageCount = 0,
		readers = [],
		ranges = [],
		onChange
	}: {
		name: string;
		def: NumberVariableDef;
		variables: VariablesMap;
		usageCount?: number;
		readers?: ConditionReader[];
		ranges?: NumberTrackRange[];
		onChange: (def: NumberVariableDef) => void;
	} = $props();

	const MODE_CHOICES: { mode: ChoiceVariableMode; label: string }[] = [
		{ mode: 'shuffle', label: 'Shuffle' },
		{ mode: 'pin', label: 'Pin' },
		{ mode: 'per-image', label: 'Per image' }
	];
	const USAGE_CHOICES = [
		{ linked: false, label: 'Independent' },
		{ linked: true, label: 'Linked' }
	];

	let previewSeed = $state(1);

	const decimals = $derived(numberDecimals(def));
	const isDecimal = $derived(decimals > 0);
	const trackRanges = $derived(ranges.length > 0 ? ranges : readers.flatMap((reader) => reader.splits ?? []));
	const issue = $derived(numberGridIssue(def));
	const valid = $derived(issue === null);
	const size = $derived(numberGridSize(def));
	const tooLarge = $derived(issue === 'too-large');
	const conditionRead = $derived(readers.length > 0);
	const usagesLocked = $derived(def.mode !== 'shuffle' || conditionRead);
	const sharesRoll = $derived(usagesShareRoll(name, variables, { [name]: readers }));
	const stepUnit = $derived(isDecimal ? 10 ** -decimalPlaces(def.step) : 1);
	const pinned = $derived(clampNumberValue(def));
	const label = $derived(name || 'this variable');
	const usageRef = $derived('${' + name + '}');

	const previewValues = $derived.by(() => {
		if (!valid || tooLarge) return [] as string[];
		if (def.mode === 'pin') return [pinned];
		const random = seededRandom(previewSeed * 7919 + size);
		const count = def.mode === 'per-image' ? 4 : 1;
		return Array.from({ length: count }, () => rollNumberValue(def, random)?.value ?? '');
	});

	const gridPreview = $derived.by(() => {
		if (!valid || tooLarge || size === 0) return '';
		const values = numberGridValues(def, size);
		if (values.length <= 4) return values.join(', ');
		return `${values.slice(0, 3).join(', ')} … ${values[values.length - 1]}`;
	});

	function position(value: number | string): number {
		const span = def.max - def.min;
		if (!(span > 0)) return 50;
		return Math.min(Math.max(((Number(value) - def.min) / span) * 100, 0), 100);
	}

	function patch(next: Partial<NumberVariableDef>) {
		const merged = { ...def, ...next };
		if (merged.max < merged.min) {
			if ('min' in next) merged.max = merged.min;
			else merged.min = merged.max;
		}
		merged.value = Number(clampNumberValue(merged));
		onChange(merged);
	}

	function setMode(mode: ChoiceVariableMode) {
		patch({ mode });
	}

	function setLinked(linked: boolean) {
		patch({ linked });
	}

	function setStep(step: number) {
		if (step > 0) patch({ step });
	}

	function setNumbers(decimal: boolean) {
		if (decimal === isDecimal) return;
		if (decimal) {
			patch({ step: 0.1 });
			return;
		}
		patch({
			step: Math.max(1, Math.round(def.step)),
			min: Math.round(def.min),
			max: Math.max(Math.round(def.max), Math.round(def.min)),
			value: Math.round(def.value)
		});
	}

	function reroll() {
		previewSeed += 1;
	}

	const sectionLabel = 'font-mono text-sm uppercase tracking-[0.04em] text-fg-subtle';
	const ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8];
</script>

<div class="space-y-3" data-testid="number-editor">
	<div class="grid gap-3 sm:grid-cols-2">
		<div class="space-y-1.5">
			<span class={sectionLabel}>Value</span>
			<div class="grid grid-cols-3 gap-0.5 rounded bg-surface-2 p-0.5" role="radiogroup" aria-label={`How ${label} picks a value`}>
				{#each MODE_CHOICES as choice (choice.mode)}
					<button
						type="button"
						role="radio"
						aria-checked={def.mode === choice.mode}
						class="rounded px-2 py-1 text-sm font-medium transition-colors max-md:py-2.5 {def.mode === choice.mode
							? 'bg-surface-1 text-fg shadow-raised'
							: 'text-fg-muted hover:text-fg'}"
						onclick={() => setMode(choice.mode)}
					>
						{choice.label}
					</button>
				{/each}
			</div>
			<p class="text-sm text-fg-subtle" data-testid="number-mode-note">
				{#if def.mode === 'pin'}
					The same number on every image. Min and max still limit what you can type, so a pinned value is always inside the range.
				{:else if def.mode === 'per-image'}
					Rolls again for every image, with that image's seed. Reuse restores the same rolls.
				{:else}
					Rolled when you click Generate and shared by the whole batch, exactly like a Choice variable in Shuffle.
				{/if}
			</p>
		</div>
		<div class="space-y-1.5">
			<span class={sectionLabel}>Usages{usageCount > 0 ? ` · ${usageCount}` : ''}</span>
			<div class="grid grid-cols-2 gap-0.5 rounded bg-surface-2 p-0.5" role="radiogroup" aria-label={`Whether each usage of ${label} rolls separately`}>
				{#each USAGE_CHOICES as choice (choice.label)}
					<button
						type="button"
						role="radio"
						aria-checked={sharesRoll === choice.linked}
						disabled={usagesLocked}
						class="rounded px-2 py-1 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 max-md:py-2.5 {sharesRoll === choice.linked
							? 'bg-surface-1 text-fg shadow-raised'
							: 'text-fg-muted hover:text-fg'}"
						onclick={() => setLinked(choice.linked)}
					>
						{choice.label}
					</button>
				{/each}
			</div>
			<p class="text-sm text-fg-subtle" data-testid="number-usage-note">
				{#if conditionRead}
					Linked because a condition reads it. Every <code class="rounded bg-surface-2 px-1 font-mono text-fg">{usageRef}</code> and every condition in an image share one value.
				{:else if def.mode === 'pin'}
					Pin applies to every usage.
				{:else if def.mode === 'per-image'}
					Each usage rolls on its own while no condition reads this variable.
				{:else if sharesRoll}
					Every usage shares one roll per generation.
				{:else}
					Each usage rolls its own value per generation.
				{/if}
			</p>
		</div>
	</div>

	<div class="space-y-1.5">
		<span class={sectionLabel}>Range</span>
		<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
			<div class="space-y-1">
				<span class={sectionLabel}>Min</span>
				<NumberStepper value={def.min} step={def.step} {decimals} label="Minimum" onChange={(min) => patch({ min })} />
			</div>
			<div class="space-y-1">
				<span class={sectionLabel}>Max</span>
				<NumberStepper value={def.max} step={def.step} {decimals} label="Maximum" onChange={(max) => patch({ max })} />
			</div>
			<div class="col-span-2 space-y-1 sm:col-span-1">
				<span class={sectionLabel}>Step</span>
				<NumberStepper
					value={def.step}
					step={stepUnit}
					min={stepUnit}
					decimals={decimalPlaces(def.step)}
					label="Step"
					onChange={setStep}
				/>
			</div>
			<div class="col-span-2 space-y-1 sm:col-span-1">
				<span class={sectionLabel}>Numbers</span>
				<div class="grid h-8 grid-cols-2 gap-0.5 rounded bg-surface-2 p-0.5 max-md:h-11" role="radiogroup" aria-label="Whole or decimal numbers">
					<button
						type="button"
						role="radio"
						aria-checked={!isDecimal}
						class="min-w-0 truncate rounded px-2 text-sm font-medium transition-colors {!isDecimal
							? 'bg-surface-1 text-fg shadow-raised'
							: 'text-fg-muted hover:text-fg'}"
						onclick={() => setNumbers(false)}
					>
						Whole
					</button>
					<button
						type="button"
						role="radio"
						aria-checked={isDecimal}
						class="min-w-0 truncate rounded px-2 text-sm font-medium transition-colors {isDecimal
							? 'bg-surface-1 text-fg shadow-raised'
							: 'text-fg-muted hover:text-fg'}"
						onclick={() => setNumbers(true)}
					>
						Decimal{#if isDecimal}<span class="max-sm:hidden"> &middot; {decimals} {decimals === 1 ? 'place' : 'places'}</span>{/if}
					</button>
				</div>
			</div>
		</div>
		{#if def.mode === 'pin'}
			<div class="grid max-w-xs gap-1 pt-1">
				<span class={sectionLabel}>Value</span>
				<NumberStepper
					value={Number(pinned)}
					step={def.step}
					{decimals}
					min={def.min}
					max={def.max}
					label="Pinned value"
					onChange={(value) => patch({ value })}
				/>
			</div>
		{/if}
		{#if issue === 'too-large'}
			<p class="text-sm text-warning" role="alert" data-testid="number-grid-warning">
				This range has {size} values and the limit is {MAX_NUMBER_GRID}. Raise the step to roll it.
			</p>
		{:else if issue}
			<p class="text-sm text-danger" role="alert" data-testid="number-range-error">
				{#if issue === 'step'}
					Step must be above 0, with at most 6 decimal places.
				{:else if issue === 'precision'}
					Min can't have more decimal places than the step, and the step is limited to 6 places.
				{:else if issue === 'value'}
					Enter a number below 1 trillion with at most 12 digits.
				{:else}
					Max must be at or above min, and both must stay below 1 trillion with at most 12 digits.
				{/if}
			</p>
		{/if}
	</div>

	{#if valid && !tooLarge}
		<div class="space-y-1.5" data-testid="number-preview">
			<div class="flex items-center gap-1.5">
				<span class="{sectionLabel} min-w-0 flex-1">
					{#if trackRanges.length > 0}Where conditions split it &middot; {/if}{def.mode === 'pin' ? 'Pinned value' : 'Dots are the roll preview'}
				</span>
				{#if def.mode !== 'pin'}
					<Tooltip text="Roll the preview again">
						<Button variant="ghost" size="sm" icon="refresh" onclick={reroll}>Roll again</Button>
					</Tooltip>
				{/if}
			</div>
			<div class="relative mx-3 h-10" data-testid="number-track">
				<div class="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 rounded bg-line-strong"></div>
				{#each ticks as tick (tick)}
					<div class="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-line-strong" style:left="{(tick / (ticks.length - 1)) * 100}%"></div>
				{/each}
				{#each trackRanges as range, ri (ri)}
					<div
						class="absolute top-1/2 h-1 -translate-y-1/2 rounded bg-signal/30"
						style:left="{position(range.from)}%"
						style:width="{Math.max(position(range.to) - position(range.from), 0)}%"
					></div>
				{/each}
				{#each previewValues as value, index (index)}
					<div
						class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
						style:left="{position(value)}%"
						data-testid="number-dot"
						data-value={value}
					>
						<span class="absolute bottom-full left-1/2 mb-0.5 -translate-x-1/2 font-mono text-sm tabular-nums text-fg-muted">{value}</span>
						<span class="grid h-5 w-5 place-items-center rounded-full border border-signal bg-signal/20 font-mono text-sm tabular-nums text-signal">
							{#if def.mode === 'per-image'}{index + 1}{/if}
						</span>
					</div>
				{/each}
			</div>
			<div class="mx-3 flex justify-between font-mono text-sm tabular-nums text-fg-subtle">
				<span>{numberGridValue(def, 0)}</span>
				<span>{numberGridValue(def, size - 1)}</span>
			</div>
			{#if trackRanges.length > 0}
				<div class="mx-3 flex justify-between font-mono text-sm uppercase tracking-[0.04em] text-signal">
					{#each trackRanges as range, ri (ri)}
						<span>{range.label}</span>
					{/each}
				</div>
			{/if}
			<p class="text-sm text-fg-subtle" data-testid="number-preview-note">
				{#if def.mode === 'pin'}
					Every image gets <span class="font-mono tabular-nums text-fg">{pinned}</span>.
				{:else if def.mode === 'per-image'}
					{size} possible values, each equally likely. {previewValues.map((v, i) => `Image ${i + 1} got ${v}`).join(', ')}.
				{:else}
					Next roll <span class="rounded bg-surface-2 px-1.5 font-mono tabular-nums text-fg">{previewValues[0]}</span> from {size} {size === 1 ? 'value' : 'values'}: {gridPreview}
				{/if}
			</p>
		</div>
	{/if}

	<div class="space-y-1.5">
		<span class={sectionLabel}>Used by</span>
		{#if usageCount === 0 && readers.length === 0}
			<p class="text-sm text-fg-subtle">Nothing uses this variable yet.</p>
		{/if}
		<div class="space-y-1.5">
			{#if usageCount > 0}
				<div class="flex flex-wrap items-center gap-2 rounded border border-line bg-surface-2/50 px-2.5 py-2 text-sm text-fg-muted" data-testid="number-used-by-text">
					<span class="inline-flex items-center gap-1 rounded border border-line-strong bg-surface-2 px-1.5 font-mono text-sm text-fg">
						<span class="text-fg-subtle">$</span>{name}
					</span>
					<span>used in text</span>
					<span class="font-mono text-sm tabular-nums text-fg-subtle">{usageCount} {usageCount === 1 ? 'use' : 'uses'}</span>
				</div>
			{/if}
			{#each readers as reader, index (index)}
				<div class="flex flex-wrap items-center gap-2 rounded border border-line bg-surface-2/50 px-2.5 py-2 text-sm text-fg-muted" data-testid="number-used-by-condition">
					<span class="inline-flex items-center gap-1 rounded border border-line-strong bg-surface-2 px-1.5 font-mono text-sm text-fg">
						<span class="text-signal">if</span>{reader.label}
					</span>
					<span>condition reads it</span>
					{#if reader.segment}
						<span class="font-mono text-sm text-fg-subtle">{reader.segment}</span>
					{/if}
				</div>
			{/each}
		</div>
	</div>
</div>
