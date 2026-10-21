// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { settle } from './helpers/settle';
import { mountComponent } from './helpers/mount';

vi.mock('$lib/services/api/index', () => ({
	api: { getPresetFormSchema: vi.fn() }
}));

const { api } = await import('$lib/services/api/index');
const { registerFieldComponent } = await import('$lib/fields/registry');
const { clearSchemaCache } = await import('$lib/form/schemaCache');
const { default: TextInput } = await import('$lib/components/form-fields/TextInput.svelte');
const { default: RowField } = await import('$lib/components/form-fields/RowField.svelte');
const { default: DynamicForm } = await import('$lib/components/DynamicForm.svelte');

registerFieldComponent('string', { component: TextInput });
registerFieldComponent('row', { component: RowField });

const SCHEMA = {
	properties: {
		media_inputs: {
			type: 'row',
			children: [
				{ name: 'prompt', type: 'string', title: 'Prompt', default: '' },
				{ name: 'media_inputs', type: 'mixed-media-stub', title: 'References', merge_from: ['old_clips', 'old_tracks'] }
			]
		}
	}
};

const image = { path: 'uploads/a.png', type: 'image' };
const video = { path: 'uploads/v.mp4', type: 'video' };
const audio = { path: 'uploads/t.wav', type: 'audio' };

beforeEach(() => {
	clearSchemaCache();
	vi.mocked(api.getPresetFormSchema).mockResolvedValue({
		success: true,
		data: { preset_id: 'p', form_schema: SCHEMA }
	} as never);
});

function mount(initialData: Record<string, unknown>) {
	const log: Array<Record<string, unknown>> = [];
	const form = mountComponent(DynamicForm, {
		presetId: `native-${Math.random()}`,
		mode: 'refs',
		initialData,
		onFormDataChange: (data: Record<string, unknown>) => log.push(data)
	});
	return { form, log };
}

describe('DynamicForm merge_from', () => {
	it('folds saved old keys into the field on first load and drops them', async () => {
		const { log } = mount({ media_inputs: [image], old_clips: [video], old_tracks: [audio] });
		await settle(10);
		const published = log[log.length - 1];
		expect(published.media_inputs).toEqual([image, video, audio]);
		expect('old_clips' in published).toBe(false);
		expect('old_tracks' in published).toBe(false);
	});

	it('folds old keys from a later session load', async () => {
		const { form, log } = mount({ media_inputs: [image] });
		await settle(10);
		expect(log[log.length - 1].media_inputs).toEqual([image]);
		form.rerender({ initialData: { media_inputs: [], old_tracks: [audio] } });
		await settle(10);
		const published = log[log.length - 1];
		expect(published.media_inputs).toEqual([audio]);
		expect('old_tracks' in published).toBe(false);
	});
});
