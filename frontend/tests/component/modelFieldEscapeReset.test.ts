// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount, unmount } from 'svelte';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getModels: vi.fn(),
		getPresetModels: vi.fn(),
		getModelById: vi.fn(),
		getTags: vi.fn(),
		getModelDownloadStatus: vi.fn(),
		startModelDownload: vi.fn(),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn(),
		getToken: vi.fn(() => null),
		getBaseURL: vi.fn(() => 'http://localhost')
	}
}));

const { api } = await import('$lib/services/api/index');
const { default: ModelField } = await import('$lib/components/form-fields/ModelField.svelte');

const flush = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

const dropdown = () => document.body.querySelector('.fixed.z-overlay');
const tagInput = () => document.body.querySelector('input[placeholder="Tag name..."]');

function pressEscape() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
}

describe('ModelField Escape', () => {
	let instance: ReturnType<typeof mount> | null = null;

	beforeEach(async () => {
		vi.mocked(api.getModels).mockResolvedValue({
			success: true,
			data: { models: [], total: 0, availability_indexed: true }
		} as never);
		vi.mocked(api.getTags).mockResolvedValue({ success: true, data: { tags: [] } } as never);
		instance = mount(ModelField as never, {
			target: document.body,
			props: {
				name: 'model',
				config: { title: 'Model', configuration: { model_type: 'checkpoint' } },
				value: { modelPath: '', tagFilters: [] },
				onChange: vi.fn()
			}
		});
		await flush(30);
	});

	afterEach(() => {
		if (instance) unmount(instance);
		instance = null;
		document.body.innerHTML = '';
	});

	it('closes the dropdown and resets the open tag input on Escape', async () => {
		const addTag = Array.from(document.querySelectorAll('button')).find((b) => b.textContent?.includes('+ Tag'))!;
		addTag.click();
		await flush(5);
		expect(tagInput()).not.toBeNull();

		document.querySelector<HTMLInputElement>('input#model')!.focus();
		await flush(30);
		expect(dropdown()).not.toBeNull();

		pressEscape();
		await flush(5);

		expect(dropdown()).toBeNull();
		expect(tagInput()).toBeNull();
	});

	it('leaves the tag input alone when the dropdown is closed', async () => {
		const addTag = Array.from(document.querySelectorAll('button')).find((b) => b.textContent?.includes('+ Tag'))!;
		addTag.click();
		await flush(5);

		pressEscape();
		await flush(5);

		expect(tagInput()).not.toBeNull();
	});
});
