<script lang="ts">
	import { tick } from 'svelte';
	import AnchoredPanel from './AnchoredPanel.svelte';
	import Icon from './Icon.svelte';
	import VariablePicker from './VariablePicker.svelte';
	import { Badge } from '$lib/components/ui';
	import { normalizeVariableDef, type VariablesMap } from '$lib/utils/variableDefs';
	import type { VariablePickerType } from '$lib/utils/variablePicker';

	let {
		variables,
		value,
		missing = false,
		allowedTypes = undefined,
		disabledReasons = undefined,
		label = 'Variable',
		disabled = false,
		class: className = '',
		onselect,
		oncreate
	}: {
		variables: VariablesMap;
		value: string;
		missing?: boolean;
		allowedTypes?: readonly VariablePickerType[];
		disabledReasons?: Record<string, string>;
		label?: string;
		disabled?: boolean;
		class?: string;
		onselect?: (name: string) => void;
		oncreate?: (name: string) => void;
	} = $props();

	const TYPE_LABELS = { text: 'Text', choice: 'Choice', number: 'Number' } as const;

	let open = $state(false);
	let triggerRef = $state<HTMLButtonElement>();

	const typeLabel = $derived(missing || !(value in variables) ? '' : TYPE_LABELS[normalizeVariableDef(variables[value]).type]);

	function close(refocus: boolean) {
		open = false;
		if (refocus) void tick().then(() => triggerRef?.focus());
	}

	function toggle() {
		if (disabled) return;
		open = !open;
	}

	function handleTriggerKeydown(e: KeyboardEvent) {
		if (disabled || open) return;
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			open = true;
		}
	}
</script>

<button
	bind:this={triggerRef}
	type="button"
	{disabled}
	class="input flex h-9 min-w-0 items-center gap-2 py-1 text-left text-base max-md:h-11 {className}"
	aria-label={label}
	aria-haspopup="listbox"
	aria-expanded={open}
	data-testid="variable-select"
	onclick={toggle}
	onkeydown={handleTriggerKeydown}
>
	<span class="min-w-0 flex-1 truncate font-mono {missing ? 'text-warning' : 'text-fg'}">${value}</span>
	{#if missing}<span class="flex-shrink-0 text-sm text-warning">missing</span>{:else if typeLabel}<Badge size="lg" class="flex-shrink-0 max-md:hidden">{typeLabel}</Badge>{/if}
	<Icon name="chevron-down" className="h-3 w-3 flex-shrink-0 text-fg-subtle" />
</button>

{#if open}
	<AnchoredPanel anchor={triggerRef} label={`Choose ${label.toLowerCase()}`} onDismiss={() => (open = false)}>
		<VariablePicker
			{variables}
			selected={missing ? null : value}
			{allowedTypes}
			{disabledReasons}
			{label}
			onselect={(name) => {
				onselect?.(name);
				close(true);
			}}
			oncreate={oncreate
				? (name) => {
						oncreate?.(name);
						close(true);
					}
				: undefined}
			onclose={(reason) => close(reason === 'escape')}
		/>
	</AnchoredPanel>
{/if}
