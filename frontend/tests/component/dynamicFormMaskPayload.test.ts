// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getPresetFormSchema: vi.fn(),
		listGenerationMedia: vi.fn().mockResolvedValue({ success: false }),
		getUploadInfo: vi.fn().mockResolvedValue({ success: false }),
		getHistoryTools: vi.fn().mockResolvedValue({ success: true, data: [] }),
		listUploads: vi.fn().mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } }),
		listLibraryItems: vi.fn().mockResolvedValue({ success: true, data: { items: [], total: 0 } }),
		getTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } })
	}
}));
vi.mock('$lib/utils/storage', () => ({
	storage: { get: vi.fn().mockReturnValue(null), set: vi.fn(), remove: vi.fn() }
}));
vi.mock('$lib/components/form-fields/mediaLoaderProbe', () => ({ probeMediaFile: vi.fn().mockResolvedValue({}) }));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MaskResultEditors.svelte')).default
}));

const { api } = await import('$lib/services/api/index');
const { registerFieldComponent } = await import('$lib/fields/registry');
const { clearSchemaCache } = await import('$lib/form/schemaCache');
const { default: MediaLoaderField } = await import('$lib/components/form-fields/MediaLoaderField.svelte');
const { default: DynamicForm } = await import('$lib/components/DynamicForm.svelte');
const { createClassComponent } = await import('svelte/legacy');

const { default: GroupField } = await import('$lib/components/form-fields/GroupField.svelte');
registerFieldComponent('image', { component: MediaLoaderField });
registerFieldComponent('group', { component: GroupField });

const SCHEMA = {
	properties: {
		main: {
			type: 'group',
			children: [{ name: 'source_image', type: 'image', title: 'Image', configuration: { allow_inpaint: true } }]
		}
	}
};

const UPLOADED = {
	path: '/abs/storage/uploads/cat.png',
	relative_path: 'uploads/cat.png',
	url: '/api/media/uploads/cat.png',
	name: 'cat.png',
	type: 'image',
	metadata: { width: 512, height: 512, size: 100 }
};

let destroy: (() => void) | undefined;

beforeEach(() => {
	clearSchemaCache();
	document.body.innerHTML = '';
	vi.mocked(api.getPresetFormSchema).mockResolvedValue({
		success: true,
		data: { preset_id: 'p', form_schema: SCHEMA }
	} as never);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
});

describe('A painted mask reaches the published form data', () => {
	for (const echo of [false, true]) {
		it(`keeps source_image_inpaint_mask (parent echoes initialData: ${echo})`, async () => {
			const target = document.createElement('div');
			document.body.appendChild(target);
			const log: Array<Record<string, unknown>> = [];
			let form: { $set: (props: Record<string, unknown>) => void; $destroy: () => void };
			form = createClassComponent({
				component: DynamicForm as never,
				target,
				props: {
					presetId: `native-${Math.random()}`,
					mode: 'edit',
					initialData: { source_image: UPLOADED },
					onFormDataChange: (data: Record<string, unknown>) => {
						log.push(data);
						if (echo) queueMicrotask(() => form.$set({ initialData: data }));
					}
				}
			});
			destroy = () => {
				form.$destroy();
				target.remove();
			};
			await settle(10);
			expect(log[log.length - 1].source_image).toBe(UPLOADED.path);

			target.querySelector<HTMLButtonElement>('[data-tools-trigger]')!.click();
			await settle(5);
			document.body.querySelector<HTMLButtonElement>('[data-tools-menu] [data-tool="mask"]')!.click();
			await settle(10);

			expect(log[log.length - 1].source_image_inpaint_mask).toBe('/abs/storage/uploads/mask-1.png');
			await settle(20);
			expect(log[log.length - 1].source_image_inpaint_mask).toBe('/abs/storage/uploads/mask-1.png');
		});
	}
});
