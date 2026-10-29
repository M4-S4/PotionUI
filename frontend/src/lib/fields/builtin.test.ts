import { describe, it, expect, vi } from 'vitest';

vi.mock('$lib/plugin-api/componentResolver', () => ({
	resolvePluginComponent: vi.fn(),
	getPluginRevision: () => 'rev-1'
}));

import { registerBuiltinFieldComponents } from './builtin';
import { resolveFieldComponent } from './registry';
import SelectField from '$lib/components/form-fields/SelectField.svelte';
import TagsField from '$lib/components/form-fields/TagsField.svelte';
import KeyValueField from '$lib/components/form-fields/KeyValueField.svelte';
import SegmentedField from '$lib/components/form-fields/SegmentedField.svelte';
import AnchorField from '$lib/components/form-fields/AnchorField.svelte';
import ColorField from '$lib/components/form-fields/ColorField.svelte';

describe('fields/builtin', () => {
	it('registers sampler and schedule onto the same component select uses', async () => {
		registerBuiltinFieldComponents();

		const [select, sampler, schedule] = await Promise.all([
			resolveFieldComponent('select'),
			resolveFieldComponent('sampler'),
			resolveFieldComponent('schedule')
		]);

		expect(sampler).toBe(SelectField);
		expect(schedule).toBe(SelectField);
		expect(sampler).toBe(select);
		expect(schedule).toBe(select);
	});

	it('registers tags onto TagsField', async () => {
		registerBuiltinFieldComponents();

		const tags = await resolveFieldComponent('tags');

		expect(tags).toBe(TagsField);
	});

	it('registers key_value_list onto KeyValueField', async () => {
		registerBuiltinFieldComponents();

		const keyValueList = await resolveFieldComponent('key_value_list');

		expect(keyValueList).toBe(KeyValueField);
	});

	it('registers segmented, anchor and color so presets can use the controls the media tool dialog draws', async () => {
		registerBuiltinFieldComponents();

		const [segmented, anchor, color] = await Promise.all([
			resolveFieldComponent('segmented'),
			resolveFieldComponent('anchor'),
			resolveFieldComponent('color')
		]);

		expect(segmented).toBe(SegmentedField);
		expect(anchor).toBe(AnchorField);
		expect(color).toBe(ColorField);
	});
});
