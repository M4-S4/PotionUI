<script lang="ts">
	import SegmentedField from '$lib/components/form-fields/SegmentedField.svelte';
	import SelectField from '$lib/components/form-fields/SelectField.svelte';
	import SliderField from '$lib/components/form-fields/SliderField.svelte';
	import NumberBoxField from '$lib/components/form-fields/NumberBoxField.svelte';
	import CheckboxField from '$lib/components/form-fields/CheckboxField.svelte';
	import ColorField from '$lib/components/form-fields/ColorField.svelte';
	import AnchorField from '$lib/components/form-fields/AnchorField.svelte';
	import TextInput from '$lib/components/form-fields/TextInput.svelte';
	import { controlFor, toFieldConfig, visibleParams, type ControlKind } from './params';
	import type { ToolParam } from './types';

	let {
		params,
		values,
		onChange
	}: {
		params: ToolParam[];
		values: Record<string, unknown>;
		onChange: (name: string, value: unknown) => void;
	} = $props();

	const COMPONENTS: Record<ControlKind, any> = {
		segmented: SegmentedField,
		dropdown: SelectField,
		slider: SliderField,
		number: NumberBoxField,
		switch: CheckboxField,
		color: ColorField,
		anchor: AnchorField,
		text: TextInput
	};

	type Row = { key: string; params: ToolParam[]; paired: boolean };

	function rowsOf(list: ToolParam[]): Row[] {
		const rows: Row[] = [];
		for (const param of list) {
			const last = rows[rows.length - 1];
			if (controlFor(param) === 'number' && last?.paired) {
				last.params.push(param);
				last.key += `|${param.name}`;
			} else {
				rows.push({ key: param.name, params: [param], paired: controlFor(param) === 'number' });
			}
		}
		return rows;
	}

	let rows = $derived(rowsOf(visibleParams(params, values)));
</script>

{#each rows as row (row.key)}
	<div class={row.params.length > 1 ? 'grid grid-cols-2 gap-3' : ''}>
		{#each row.params as param (param.name)}
			{@const control = controlFor(param)}
			{@const Control = COMPONENTS[control]}
			<div data-param={param.name} data-control={control} class="min-w-0">
				<Control
					name={param.name}
					config={toFieldConfig(param)}
					value={values[param.name]}
					onChange={(name: string, next: unknown) => onChange(name, next)}
				/>
			</div>
		{/each}
	</div>
{/each}
