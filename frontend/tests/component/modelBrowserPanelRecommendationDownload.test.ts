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
	provider: 'huggingface',
	ref: '{"repo": "ZhengPeng7/BiRefNet", "file": "model.safetensors", "revision": "main"}',
	size: '444 MB'
};

const UPSCALER = {
	name: 'RealESRGAN_x4plus.pth',
	installed: false,
	link: 'https://example.com/RealESRGAN_x4plus.pth',
	sha256: 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
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
	it('posts the recommendation name and the field model type for a provider-backed recommendation', async () => {
		stubApi();
		const target = mountPanel({ modelType: 'detection_segm', recommendations: [BIREFNET] });

		await clickDownload(target, BIREFNET.name);

		expect(api.startModelDownload).toHaveBeenCalledTimes(1);
		expect(api.startModelDownload).toHaveBeenCalledWith({
			name: 'BiRefNet-general.safetensors',
			model_type: 'detection_segm',
			provider: 'huggingface',
			ref: BIREFNET.ref
		});
	});

	it('posts the recommendation name, link and checksum for a direct-link recommendation', async () => {
		stubApi();
		const target = mountPanel({ modelType: 'upscaler', recommendations: [UPSCALER] });

		await clickDownload(target, UPSCALER.name);

		expect(api.startModelDownload).toHaveBeenCalledWith({
			name: 'RealESRGAN_x4plus.pth',
			model_type: 'upscaler',
			link: UPSCALER.link,
			sha256: UPSCALER.sha256
		});
	});
});
