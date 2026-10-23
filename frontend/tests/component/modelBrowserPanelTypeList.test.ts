// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getModels: vi.fn(),
		getPresetModels: vi.fn(),
		getTags: vi.fn(),
		getModelDownloadStatus: vi.fn(),
		startModelDownload: vi.fn()
	}
}));

vi.mock('$lib/stores/auth', () => ({
	authStore: { subscribe: (fn: (v: unknown) => void) => (fn({ user: null }), () => {}) }
}));

const { api } = await import('$lib/services/api/index');
const { default: ModelBrowserPanel } = await import(
	'../../src/lib/components/form-fields/ModelBrowserPanel.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

const RESPONSE = {
	success: true,
	data: {
		models: [{ id: 'm1', filename: 'flux-dev.safetensors', model_type: 'diffusion_model' }],
		total: 1,
		indexed: true
	}
};

function mountPanel(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: ModelBrowserPanel as never,
		target,
		props: { modelType: 'diffusion_model', presetId: 'preset-1', onSelect: vi.fn(), ...props }
	});
	return component as unknown as { $set: (props: Record<string, unknown>) => void; $destroy: () => void };
}

function flush(ms = 0) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

let mounted: ReturnType<typeof mountPanel> | undefined;

afterEach(() => {
	mounted?.$destroy();
	mounted = undefined;
	vi.clearAllMocks();
});

describe('ModelBrowserPanel model type list', () => {
	it('sends the joined type list and the unquantized filter on the preset request', async () => {
		vi.mocked(api.getPresetModels).mockResolvedValue(RESPONSE as never);
		mounted = mountPanel({
			modelTypes: ['diffusion_model', 'unet', 'text_encoder'],
			unquantizedOnly: true
		});
		await flush();

		expect(api.getPresetModels).toHaveBeenCalledTimes(1);
		const [presetId, modelType, , opts] = vi.mocked(api.getPresetModels).mock.calls[0];
		expect(presetId).toBe('preset-1');
		expect(modelType).toBe('diffusion_model');
		expect(opts).toMatchObject({ modelTypes: 'diffusion_model,unet,text_encoder', unquantizedOnly: true });
	});

	it('keeps the single-type request free of the new parameters', async () => {
		vi.mocked(api.getPresetModels).mockResolvedValue(RESPONSE as never);
		mounted = mountPanel({ modelType: 'lora' });
		await flush();

		const opts = vi.mocked(api.getPresetModels).mock.calls[0][3];
		expect(opts?.modelTypes).toBeUndefined();
		expect(opts?.unquantizedOnly).toBeUndefined();
	});

	it('refetches when the type list or the filter changes', async () => {
		vi.mocked(api.getPresetModels).mockResolvedValue(RESPONSE as never);
		mounted = mountPanel({ modelTypes: ['diffusion_model', 'unet'] });
		await flush();
		expect(api.getPresetModels).toHaveBeenCalledTimes(1);

		mounted.$set({ modelTypes: ['diffusion_model', 'unet', 'text_encoder'] });
		await flush();
		expect(api.getPresetModels).toHaveBeenCalledTimes(2);
		expect(vi.mocked(api.getPresetModels).mock.calls[1][3]).toMatchObject({
			modelTypes: 'diffusion_model,unet,text_encoder'
		});

		mounted.$set({ unquantizedOnly: true });
		await flush();
		expect(api.getPresetModels).toHaveBeenCalledTimes(3);
		expect(vi.mocked(api.getPresetModels).mock.calls[2][3]).toMatchObject({ unquantizedOnly: true });
	});

	it('does not refetch when nothing in the scope changed', async () => {
		vi.mocked(api.getPresetModels).mockResolvedValue(RESPONSE as never);
		mounted = mountPanel({ modelTypes: ['diffusion_model', 'unet'] });
		await flush();
		mounted.$set({ modelTypes: ['diffusion_model', 'unet'] });
		await flush();
		expect(api.getPresetModels).toHaveBeenCalledTimes(1);
	});
});
