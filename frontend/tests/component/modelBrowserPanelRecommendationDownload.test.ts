import { describe, it, expect, vi, afterEach } from 'vitest';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getModels: vi.fn(),
		getPresetModels: vi.fn(),
		getTags: vi.fn(),
		getModelDownloadStatus: vi.fn(),
		startModelDownload: vi.fn(),
		getUnindexedModelsCount: vi.fn(),
		listPresets: vi.fn(),
		getPresetSlotVariants: vi.fn()
	}
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: {
		subscribe: (fn: (v: unknown) => void) => (fn({ user: { account_type: 'ADMIN' } }), () => {})
	}
}));

vi.mock('$lib/utils/logger', () => ({
	logger: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
	getErrorMessage: (error: unknown) => String(error)
}));

const { api } = await import('$lib/services/api/index');
const { default: ModelBrowserPanel } = await import('../../src/lib/components/form-fields/ModelBrowserPanel.svelte');
const { createClassComponent } = await import('svelte/legacy');

const BIREFNET = {
	name: 'BiRefNet-general.safetensors',
	installed: false,
	link: 'https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors',
	size: '444 MB',
	sha256: '9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154'
};

const PROVIDER_BACKED = {
	name: 'Great Checkpoint.safetensors',
	installed: false,
	provider: 'civitai',
	ref: '{"model_id": "12345", "version_id": "67890"}'
};

let destroy: (() => void) | undefined;

function mountPanel(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: ModelBrowserPanel as never,
		target,
		props: { onSelect: vi.fn(), presetId: 'pixal3d', ...props }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	return target;
}

function stubApi() {
	vi.mocked(api.getPresetModels).mockResolvedValue({ success: true, data: { models: [], total: 0 } } as never);
	vi.mocked(api.getUnindexedModelsCount).mockResolvedValue({ success: true, data: { by_type: {} } } as never);
	vi.mocked(api.getPresetSlotVariants).mockResolvedValue({ gpu: {} as never, slots: [] });
	vi.mocked(api.startModelDownload).mockResolvedValue({ success: true, data: { download_id: 'd1' } } as never);
}

async function clickDownload(target: HTMLElement, name: string) {
	await settle();
	const row = Array.from(target.querySelectorAll('[data-recommended-download]')).find((el) =>
		el.textContent?.includes(name)
	);
	expect(row).toBeTruthy();
	(row!.querySelector('button') as HTMLButtonElement).click();
	await settle();
}

afterEach(() => {
	destroy?.();
	destroy = undefined;
	vi.clearAllMocks();
});

describe('ModelBrowserPanel recommendation download', () => {
	it('posts the name, link and checksum with the field model type for a link recommendation', async () => {
		stubApi();
		const target = mountPanel({ modelType: 'detection_segm', recommendations: [BIREFNET] });

		await clickDownload(target, BIREFNET.name);

		expect(api.startModelDownload).toHaveBeenCalledTimes(1);
		expect(api.startModelDownload).toHaveBeenCalledWith({
			name: 'BiRefNet-general.safetensors',
			model_type: 'detection_segm',
			link: BIREFNET.link,
			sha256: BIREFNET.sha256
		});
	});

	it('posts the name, provider and ref with the field model type for a provider-backed recommendation', async () => {
		stubApi();
		const target = mountPanel({ modelType: 'checkpoint', recommendations: [PROVIDER_BACKED] });

		await clickDownload(target, PROVIDER_BACKED.name);

		expect(api.startModelDownload).toHaveBeenCalledWith({
			name: 'Great Checkpoint.safetensors',
			model_type: 'checkpoint',
			provider: 'civitai',
			ref: PROVIDER_BACKED.ref
		});
	});
});
