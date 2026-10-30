import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';

export const GIB = 1024 ** 3;

export function recipe(overrides: Partial<SetupRecipe> = {}): SetupRecipe {
	return {
		id: 'r',
		name: 'Recipe',
		summary: 'summary',
		description: '',
		engine: 'native',
		category: 'image',
		artifact_count: 1,
		total_download_bytes: 10 * GIB,
		presets: [],
		last_completed_at: null,
		blurb: 'A blurb',
		min_vram_gb: null,
		recommended_vram_gb: null,
		sources: ['huggingface.co'],
		first_run: true,
		...overrides
	};
}

export const ZIMAGE = recipe({
	id: 'zimage-starter',
	name: 'Z-Image Starter',
	min_vram_gb: 12,
	recommended_vram_gb: 16,
	total_download_bytes: 19.27 * GIB
});
export const SDXL = recipe({
	id: 'sdxl-starter',
	name: 'SDXL Starter',
	min_vram_gb: 8,
	total_download_bytes: 6.62 * GIB
});
export const QWEN = recipe({
	id: 'qwen-image-starter',
	name: 'Qwen-Image Starter',
	recommended_vram_gb: 16,
	total_download_bytes: 28 * GIB
});

export function machine(vram: number | null, overrides: Partial<SetupMachine> = {}): SetupMachine {
	return {
		gpu:
			vram === null
				? null
				: { name: 'NVIDIA GeForce RTX', vram_bytes: vram * GIB, compute_capability: '8.9', supported: true },
		ram_total_bytes: 64 * GIB,
		models_disk: { path: '/data/models', free_bytes: 412 * GIB, total_bytes: 900 * GIB },
		...overrides
	};
}
