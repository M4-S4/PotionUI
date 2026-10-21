<script lang="ts">
	import FieldShell from './FieldShell.svelte';
	import { Button, IconButton, Input } from '$lib/components/ui';
	import { addRow, normalizeRows, patchRow, removeRow, type KeyValueRow } from './keyValueRows';

	let {
		name,
		config = {},
		value,
		onChange
	}: {
		name: string | null;
		config?: any;
		value: unknown;
		onChange: (fieldName: string, value: KeyValueRow[]) => void;
	} = $props();

	let rows = $derived(normalizeRows(value));
	let label = $derived(config.title || name || '');
	let description = $derived(config.description || '');
	let tooltip = $derived(config.tooltip);
	let secretToggle = $derived(Boolean(config.secret_toggle));
	let disabled = $derived(config.disabled ?? false);

	function emit(next: KeyValueRow[]) {
		if (name) onChange(name, next);
	}
</script>

<FieldShell {name} {label} {description} {tooltip}>
	<div class="space-y-2" data-key-value-list={name}>
		{#each rows as row, index (index)}
			<div class="flex items-center gap-2" data-key-value-row>
				<Input
					class="min-w-0 flex-1 font-mono text-xs"
					type="text"
					placeholder="Name"
					aria-label={`${label} name ${index + 1}`}
					value={row.name}
					{disabled}
					oninput={(event: Event) =>
						emit(patchRow(rows, index, { name: (event.currentTarget as HTMLInputElement).value }))}
				/>
				<Input
					class="min-w-0 flex-[2] font-mono text-xs"
					type={row.secret ? 'password' : 'text'}
					autocomplete="off"
					placeholder="Value"
					aria-label={`${label} value ${index + 1}`}
					value={row.value}
					{disabled}
					onfocus={(event: FocusEvent) => {
						if (row.secret && row.value === '***') (event.currentTarget as HTMLInputElement).select();
					}}
					oninput={(event: Event) =>
						emit(patchRow(rows, index, { value: (event.currentTarget as HTMLInputElement).value }))}
				/>
				{#if secretToggle}
					<IconButton
						icon="shield"
						size="sm"
						label={row.secret ? 'Stored as a secret' : 'Store as a secret'}
						active={Boolean(row.secret)}
						ariaPressed={Boolean(row.secret)}
						{disabled}
						onclick={() => emit(patchRow(rows, index, { secret: !row.secret }))}
					/>
				{/if}
				<IconButton
					icon="trash"
					size="sm"
					label={`Remove ${label} row ${index + 1}`}
					{disabled}
					onclick={() => emit(removeRow(rows, index))}
				/>
			</div>
		{/each}
		<Button size="xs" variant="secondary" icon="plus" {disabled} onclick={() => emit(addRow(rows))}>
			Add row
		</Button>
	</div>
</FieldShell>
