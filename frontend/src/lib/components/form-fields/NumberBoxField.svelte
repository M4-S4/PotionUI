<script lang="ts">
	import FieldShell from './FieldShell.svelte';
	import { numberBoxIssue, parseNumberBox } from './numberBox';

	export let name: string | null;
	export let config: any = {};
	export let value: any;
	export let onChange: (fieldName: string, value: any) => void;

	$: label = config.title || name || '';
	$: description = config.description || '';
	$: tooltip = config.tooltip || config.configuration?.tooltip;
	$: disabled = config.disabled ?? false;
	$: unit = typeof config.unit === 'string' ? config.unit : '';
	$: integer = config.integer === true;
	$: min = config.minimum ?? config.min;
	$: max = config.maximum ?? config.max;
	$: placeholder = config.placeholder ?? 'Auto';

	let text = '';
	let focused = false;
	$: if (!focused) text = typeof value === 'number' && Number.isFinite(value) ? String(value) : '';
	$: issue = numberBoxIssue(typeof value === 'number' ? value : null, text, min, max);

	function handleInput(event: Event) {
		text = (event.currentTarget as HTMLInputElement).value;
		if (!name) return;
		const parsed = parseNumberBox(text, integer);
		if (parsed === undefined) return;
		onChange(name, parsed);
	}

	function handleBlur() {
		focused = false;
		const parsed = parseNumberBox(text, integer);
		text = typeof parsed === 'number' ? String(parsed) : '';
	}
</script>

<FieldShell {name} {label} {description} {tooltip} labelFor={name ? `${name}-input` : undefined}>
	<div class="relative">
		<input
			id={name ? `${name}-input` : undefined}
			type="text"
			inputmode={integer ? 'numeric' : 'decimal'}
			value={text}
			{placeholder}
			{disabled}
			aria-invalid={issue !== null}
			class="input h-8 w-full px-2.5 py-0 font-mono text-sm tabular-nums {unit ? 'pr-9' : ''}"
			on:focus={() => (focused = true)}
			on:input={handleInput}
			on:blur={handleBlur}
		/>
		{#if unit}
			<span
				class="pointer-events-none absolute inset-y-0 right-2.5 flex items-center font-mono text-sm text-fg-subtle"
			>
				{unit}
			</span>
		{/if}
	</div>
	{#if issue}
		<p class="mt-1 font-mono text-sm text-danger" role="alert">{issue}</p>
	{/if}
</FieldShell>
