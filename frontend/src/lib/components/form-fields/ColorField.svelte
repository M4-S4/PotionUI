<script lang="ts">
	import SegmentedControl from '$lib/components/ui/SegmentedControl.svelte';
	import FieldShell from './FieldShell.svelte';
	import { normalizeHex, TRANSPARENT } from './colorValue';

	export let name: string | null;
	export let config: any = {};
	export let value: any;
	export let onChange: (fieldName: string, value: any) => void;

	const MODES = [
		{ id: 'transparent', label: 'Transparent' },
		{ id: 'color', label: 'Colour' }
	];

	$: label = config.title || name || '';
	$: description = config.description || '';
	$: tooltip = config.tooltip || config.configuration?.tooltip;
	$: disabled = config.disabled ?? false;
	$: allowTransparent = config.allow_transparent === true;
	$: transparent = allowTransparent && value === TRANSPARENT;
	$: held = normalizeHex(value);

	let remembered: string | null = null;
	$: if (held) remembered = held;
	$: shown = held ?? remembered ?? normalizeHex(config.default) ?? '#000000';

	let hexText = '';
	let editing = false;
	$: if (!editing) hexText = shown;
	$: invalid = editing && hexText.length > 0 && normalizeHex(hexText) === null;

	function commit(next: string) {
		if (name) onChange(name, next);
	}

	function pickMode(id: string) {
		if (id === 'transparent') {
			commit(TRANSPARENT);
		} else {
			commit(shown);
		}
	}

	function commitHex() {
		const parsed = normalizeHex(hexText);
		editing = false;
		if (parsed) commit(parsed);
		hexText = parsed ?? shown;
	}

	function handleHexKeydown(event: KeyboardEvent) {
		if (event.key !== 'Enter') return;
		event.preventDefault();
		commitHex();
	}

	function handleSwatch(event: Event) {
		const picked = normalizeHex((event.currentTarget as HTMLInputElement).value);
		if (picked) commit(picked);
	}
</script>

<FieldShell {name} {label} {description} {tooltip} labelId={name ? `${name}-label` : undefined}>
	<div class="flex flex-col gap-2">
		{#if allowTransparent}
			<div>
				<SegmentedControl
					variant="toggle"
					ariaLabel="{label} mode"
					items={MODES}
					selected={transparent ? 'transparent' : 'color'}
					onSelect={pickMode}
				/>
			</div>
		{/if}
		{#if !transparent}
			<div class="flex items-center gap-2">
				<span
					class="relative block h-8 w-8 shrink-0 overflow-hidden rounded border border-line-strong"
					style="background-color: {shown};"
				>
					<input
						type="color"
						value={shown.toLowerCase()}
						{disabled}
						aria-label="{label} swatch"
						class="absolute inset-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
						on:input={handleSwatch}
					/>
				</span>
				<input
					type="text"
					bind:value={hexText}
					maxlength="7"
					spellcheck="false"
					{disabled}
					aria-label="{label} hex"
					aria-invalid={invalid}
					class="input h-8 w-28 px-2 py-0 font-mono text-sm tabular-nums"
					on:focus={() => (editing = true)}
					on:blur={commitHex}
					on:keydown={handleHexKeydown}
				/>
			</div>
		{/if}
	</div>
</FieldShell>
