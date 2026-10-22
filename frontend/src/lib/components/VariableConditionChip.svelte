<script lang="ts">
	import AnchoredPanel from './AnchoredPanel.svelte';
	import Icon from './Icon.svelte';
	import Tooltip from './Tooltip.svelte';
	import VariablePicker from './VariablePicker.svelte';
	import { Button, IconButton } from '$lib/components/ui';
	import {
		dependents,
		normalizeVariableDef,
		optionText,
		whenIsBroken,
		type ChoiceVariableMode,
		type VariableOptionWhen,
		type VariablesMap
	} from '$lib/utils/variableDefs';

	let {
		ownerName,
		optionNumber = 1,
		when,
		mode,
		variables,
		onChange
	}: {
		ownerName: string;
		optionNumber?: number;
		when: VariableOptionWhen | undefined;
		mode: ChoiceVariableMode;
		variables: VariablesMap;
		onChange: (when: VariableOptionWhen | undefined) => void;
	} = $props();

	let open = $state(false);
	let pickingVar = $state(false);
	let chipRef = $state<HTMLSpanElement>();

	let struckThrough = $derived(mode === 'pin' || mode === 'per-image');
	let broken = $derived(when ? whenIsBroken(when, variables) : false);

	let pickableEntries = $derived(
		Object.keys(variables).map((name) => {
			const def = normalizeVariableDef(variables[name]);
			let reason: string | null = null;
			if (name === ownerName) reason = 'this variable';
			else if (def.type !== 'choice') reason = def.type === 'number' ? 'number variable' : 'text variable';
			else if (dependents(ownerName, variables, true).includes(name)) reason = `depends on $${ownerName}`;
			return { name, reason };
		})
	);

	let disabledReasons = $derived(
		Object.fromEntries(pickableEntries.filter((entry) => entry.reason).map((entry) => [entry.name, entry.reason as string]))
	);

	let referencedDef = $derived(when ? normalizeVariableDef(variables[when.var]) : null);
	let referencedOptions = $derived(
		referencedDef && referencedDef.type === 'choice'
			? [...new Set(referencedDef.options.map((o) => optionText(o).trim()).filter((t) => t.length > 0))]
			: []
	);

	function closePopover() {
		open = false;
		pickingVar = false;
	}

	function toggleOpen() {
		if (struckThrough) return;
		pickingVar = !when;
		open = !open;
	}

	function clear(e: MouseEvent) {
		e.stopPropagation();
		onChange(undefined);
		open = false;
		pickingVar = false;
	}

	function pickVariable(name: string) {
		onChange({ var: name, values: [] });
		pickingVar = false;
	}

	function toggleValue(value: string) {
		if (!when) return;
		const values = when.values.includes(value) ? when.values.filter((v) => v !== value) : [...when.values, value];
		onChange(values.length > 0 ? { var: when.var, values } : undefined);
	}

	const chipBase = 'inline-flex h-7 min-w-0 max-w-full items-center rounded border px-2.5 font-mono text-sm';
</script>

<span bind:this={chipRef} class="relative inline-flex min-w-0 max-w-full">
	{#if struckThrough && when}
		<Tooltip
			wrapperClass="inline-flex min-w-0 max-w-full"
			text={mode === 'pin'
				? 'Pinning wins regardless of this condition.'
				: 'Per-image ignores conditions — every option is used.'}
		>
			<span class="{chipBase} border-line-strong bg-surface-2 text-fg-disabled line-through">
				<span class="truncate">${when.var} = {when.values.join(', ')}</span>
			</span>
		</Tooltip>
	{:else if !when}
		<button
			type="button"
			class="{chipBase} border-line-strong bg-surface-2 uppercase tracking-[0.04em] text-fg-subtle transition-colors hover:border-line-hover hover:text-fg-muted"
			aria-label={`Set condition for this option of ${ownerName}`}
			onclick={toggleOpen}
		>
			Always
		</button>
	{:else}
		<Tooltip wrapperClass="inline-flex min-w-0 max-w-full" text={`$${when.var} = ${when.values.join(', ')}`}>
			<span class="{chipBase} {broken ? 'border-warning/35 bg-warning/10 text-warning' : 'border-signal/28 bg-signal/10 text-signal'}">
				<button
					type="button"
					class="min-w-0 truncate"
					aria-label={`Edit condition for this option of ${ownerName}`}
					onclick={toggleOpen}
				>
					${when.var} ∈ {when.values.join(', ')}
				</button>
				<button
					type="button"
					class="ml-2 flex-shrink-0 opacity-70 transition-opacity hover:opacity-100"
					aria-label={`Clear condition on ${ownerName}`}
					onclick={clear}
				>
					<Icon name="close" className="h-3.5 w-3.5" />
				</button>
			</span>
		</Tooltip>
	{/if}
</span>

{#if open}
	<AnchoredPanel anchor={chipRef} label={`Condition for ${ownerName}`} heightEstimate={320} escape onDismiss={closePopover}>
		<header class="flex min-h-[52px] flex-shrink-0 items-center gap-3 border-b border-line bg-surface-2/50 py-2 pl-3.5 pr-2">
			<span class="grid h-7 w-7 flex-shrink-0 place-items-center rounded border border-line bg-surface-2 font-mono text-base font-semibold leading-none text-ai-3" aria-hidden="true">if</span>
			<div class="min-w-0 flex-1">
				<strong class="block truncate text-md font-semibold text-fg">Only when</strong>
				<span class="block truncate text-sm text-fg-muted">Option {optionNumber} of <span class="font-mono text-fg">${ownerName}</span></span>
			</div>
			<IconButton icon="close" label="Close" size="sm" onclick={closePopover} />
		</header>
		<div class="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5">
			<div class="flex flex-wrap items-center gap-2 text-base text-fg-muted">
				<span>Only when</span>
				<button
					type="button"
					class="input flex h-9 w-auto items-center gap-2 py-1 font-mono text-base text-fg max-md:h-11"
					onclick={() => (pickingVar = !pickingVar)}
				>
					{when ? `$${when.var}` : 'choose…'}
					<Icon name="chevron-down" className="h-3.5 w-3.5 text-fg-subtle" />
				</button>
				<span>is one of</span>
			</div>
			{#if pickingVar}
				<div class="-mx-3.5 border-y border-line">
					<VariablePicker
						{variables}
						selected={when?.var ?? null}
						{disabledReasons}
						label="Condition variable"
						maxListHeight="14rem"
						onselect={pickVariable}
						onclose={() => {
							if (when) pickingVar = false;
							else closePopover();
						}}
					/>
				</div>
			{:else if when}
				<div class="space-y-0.5 rounded-lg border border-line p-1" role="group" aria-label={`Values of $${when.var}`}>
					{#each referencedOptions as value (value)}
						{@const on = when.values.includes(value)}
						<button
							type="button"
							class="grid w-full grid-cols-[1.25rem_minmax(0,1fr)] items-start gap-2.5 rounded px-2.5 py-2 text-left text-base leading-5 transition-colors hover:bg-surface-2 {on ? 'text-fg' : 'text-fg-muted'}"
							aria-label={`Toggle ${value}`}
							aria-pressed={on}
							onclick={() => toggleValue(value)}
						>
							<span
								class="mt-0.5 grid h-4 w-4 place-items-center rounded border {on
									? 'border-signal bg-signal text-accent-contrast'
									: 'border-line-hover text-transparent'}"
								aria-hidden="true"
							>
								<Icon name="check" className="h-3 w-3" strokeWidth={2.5} />
							</span>
							<span class="min-w-0 break-words">{value}</span>
						</button>
					{:else}
						<p class="px-2.5 py-2 text-sm text-fg-subtle">${when.var} has no options yet.</p>
					{/each}
				</div>
				<p class="text-sm text-fg-subtle">Values come from ${when.var}'s options; nothing to type.</p>
			{/if}
		</div>
		<footer class="flex min-h-[52px] flex-shrink-0 items-center gap-2 border-t border-line py-2.5 pl-3.5 pr-2.5">
			{#if when}
				<Button variant="ghost" size="sm" onclick={clear}>Set to Always</Button>
			{/if}
			<Button variant="primary" size="sm" class="ml-auto" onclick={closePopover}>Done</Button>
		</footer>
	</AnchoredPanel>
{/if}
