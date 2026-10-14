import { describe, it, expect } from 'vitest';
import { pluginSettingFormValues } from './pluginSettingValues';
import type { PluginSettingSchema } from '$lib/stores/plugins';

const flag = (overrides: Partial<PluginSettingSchema> = {}): PluginSettingSchema => ({
	name: 'hide_poi',
	type: 'boolean',
	label: 'Hide POI',
	...overrides
});

describe('pluginSettingFormValues', () => {
	it('reads a boolean stored as "False" as off', () => {
		expect(pluginSettingFormValues([flag()], { hide_poi: 'False' })).toEqual({ hide_poi: false });
	});

	it.each(['True', 'true', '1', 'yes', 'ON', true])('reads a boolean stored as %j as on', (stored) => {
		expect(pluginSettingFormValues([flag()], { hide_poi: stored })).toEqual({ hide_poi: true });
	});

	it.each(['false', '0', '', 'no', false])('reads a boolean stored as %j as off', (stored) => {
		expect(pluginSettingFormValues([flag()], { hide_poi: stored })).toEqual({ hide_poi: false });
	});

	it('gives an unset boolean its default, coerced to a boolean', () => {
		expect(pluginSettingFormValues([flag({ default: true })], {})).toEqual({ hide_poi: true });
		expect(pluginSettingFormValues([flag({ default: 'false' })], {})).toEqual({ hide_poi: false });
	});

	it('leaves an unset boolean without a default unset', () => {
		expect(pluginSettingFormValues([flag()], {})).toEqual({});
	});

	it('turns a numeric number setting into a number', () => {
		const schema: PluginSettingSchema[] = [
			{ name: 'limit', type: 'number', label: 'Limit' },
			{ name: 'scale', type: 'number', label: 'Scale', default: '2.5' }
		];
		expect(pluginSettingFormValues(schema, { limit: '10' })).toEqual({ limit: 10, scale: 2.5 });
	});

	it('keeps a number setting that is not numeric as stored', () => {
		const schema: PluginSettingSchema[] = [{ name: 'limit', type: 'number', label: 'Limit' }];
		expect(pluginSettingFormValues(schema, { limit: 'abc' })).toEqual({ limit: 'abc' });
		expect(pluginSettingFormValues(schema, { limit: '' })).toEqual({ limit: '' });
	});

	it('leaves string and select settings untouched', () => {
		const schema: PluginSettingSchema[] = [
			{ name: 'api_base', type: 'string', label: 'API base' },
			{ name: 'code', type: 'string', label: 'Code' },
			{
				name: 'mode',
				type: 'select',
				label: 'Mode',
				options: [
					{ value: 'true', label: 'Yes' },
					{ value: 'false', label: 'No' }
				]
			}
		];
		const stored = { api_base: 'https://civitai.com', code: '10', mode: 'false' };
		expect(pluginSettingFormValues(schema, stored)).toEqual(stored);
	});

	it('keeps stored keys the schema does not declare', () => {
		expect(pluginSettingFormValues([flag()], { hide_poi: 'true', legacy_key: 'False' })).toEqual({
			hide_poi: true,
			legacy_key: 'False'
		});
	});

	it('never gives an info entry a value', () => {
		const schema: PluginSettingSchema[] = [{ name: 'notice', type: 'info', label: 'Notice', default: 'x' }];
		expect(pluginSettingFormValues(schema, {})).toEqual({});
		expect(pluginSettingFormValues(schema, { notice: 'x' })).toEqual({});
	});

	it('reads back what the form saved, so nothing is dirty after the refetch', () => {
		const schema: PluginSettingSchema[] = [
			flag(),
			flag({ name: 'download_media', default: true }),
			{ name: 'limit', type: 'number', label: 'Limit' }
		];
		const form = pluginSettingFormValues(schema, { hide_poi: 'True', limit: '10' });
		form.hide_poi = false;
		const stored = Object.fromEntries(
			Object.entries(form).map(([k, v]) => [k, typeof v === 'boolean' ? JSON.stringify(v) : String(v)])
		);
		const reread = pluginSettingFormValues(schema, stored);
		expect(reread).toEqual({ hide_poi: false, download_media: true, limit: 10 });
		expect(JSON.stringify(reread)).toBe(JSON.stringify(form));
	});
});
