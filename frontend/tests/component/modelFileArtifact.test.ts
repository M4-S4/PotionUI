import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync } from 'svelte';

const hoisted = vi.hoisted(() => ({ goto: vi.fn() }));
vi.mock('$app/navigation', () => ({ goto: hoisted.goto }));

const { default: ModelFileArtifact } = await import(
	'$lib/components/generation/artifacts/ModelFileArtifact.svelte'
);
const { default: LiveModelFileArtifacts } = await import(
	'$lib/components/generation/LiveModelFileArtifacts.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

const GB = 1024 ** 3;

const MODEL_DATA = {
	index: 0,
	kind: 'model',
	model_type: 'diffusion_model',
	file_name: 'flux-fp8.safetensors',
	size_bytes: 11.9 * GB,
	source_size_bytes: 23.8 * GB,
	format: 'fp8_scaled',
	model_id: 'm-9',
	indexed: true,
	index_warning: null,
	summary: {
		layers_quantized: 412,
		layers_kept: 38,
		energy_kept: 0.987,
		worst_layers: [{ key: 'blocks.3.attn.qkv', rel_l2: 0.031, cosine: 0.9994 }],
		sources: [{ name: 'style.safetensors', sha256: 'ab'.repeat(32), strength: 0.8 }],
		licence_note: 'Derived from a non-commercial model'
	}
};

const LORA_DATA = {
	index: 0,
	kind: 'lora',
	model_type: 'lora',
	file_name: 'merged.safetensors',
	size_bytes: 0.2 * GB,
	source_size_bytes: null,
	format: 'lora',
	model_id: 'l-1',
	indexed: true,
	summary: { rank: 32, energy_kept: 0.93 }
};

function mount(component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({ component: component as never, target, props });
	flushSync();
	return {
		target,
		destroy: () => {
			instance.$destroy();
			target.remove();
		}
	};
}

function button(root: ParentNode, text: string): HTMLButtonElement | undefined {
	return Array.from(root.querySelectorAll('button')).find((b) =>
		(b.textContent || '').trim().startsWith(text)
	) as HTMLButtonElement | undefined;
}

describe('ModelFileArtifact', () => {
	let cleanup: (() => void) | null = null;

	afterEach(() => {
		cleanup?.();
		cleanup = null;
		hoisted.goto.mockReset();
	});

	it('renders a model card with name, type, format, sizes, quality and sources', () => {
		const mounted = mount(ModelFileArtifact, { artifact: { artifact_data: MODEL_DATA } });
		cleanup = mounted.destroy;
		const text = mounted.target.textContent ?? '';

		expect(mounted.target.querySelector('[data-model-file-name]')?.textContent).toBe(
			'flux-fp8.safetensors'
		);
		expect(text).toContain('diffusion model');
		expect(text).toContain('fp8 scaled');
		expect(mounted.target.querySelector('[data-model-file-sizes]')?.textContent).toBe(
			'23.8 → 11.9 GB'
		);
		expect(text).toContain('-50%');
		expect(mounted.target.querySelector('[data-model-file-quality]')?.textContent).toContain('412');
		expect(mounted.target.querySelector('[data-model-file-energy]')?.textContent).toBe('98.7%');
		expect(mounted.target.querySelector('[data-model-file-sources]')?.textContent).toContain(
			'style.safetensors'
		);
		expect(mounted.target.querySelector('[data-model-file-sources]')?.textContent).toContain('0.8x');
		expect(mounted.target.querySelector('[data-model-file-licence]')?.textContent).toBe(
			'Derived from a non-commercial model'
		);
		expect(mounted.target.querySelector('[data-model-file-rank]')).toBeNull();
	});

	it('shows the format badge only when it differs from the type', () => {
		const lora = mount(ModelFileArtifact, { artifact: { artifact_data: LORA_DATA } });
		cleanup = lora.destroy;
		expect(lora.target.textContent?.match(/lora/gi)?.length).toBe(1);
		lora.destroy();
		const model = mount(ModelFileArtifact, { artifact: { artifact_data: MODEL_DATA } });
		cleanup = model.destroy;
		expect(model.target.textContent).toContain('fp8 scaled');
	});

	it('keeps the extension in its own non-shrinking span', () => {
		const mounted = mount(ModelFileArtifact, { artifact: { artifact_data: MODEL_DATA } });
		cleanup = mounted.destroy;
		const parts = mounted.target.querySelectorAll('[data-model-file-name] > span');
		expect(parts[0].textContent).toBe('flux-fp8');
		expect(parts[1].textContent).toBe('.safetensors');
	});

	it('keeps worst layers collapsed until asked', () => {
		const mounted = mount(ModelFileArtifact, { artifact: { artifact_data: MODEL_DATA } });
		cleanup = mounted.destroy;

		expect(mounted.target.querySelector('[data-model-file-worst]')).toBeNull();
		const toggle = button(mounted.target, 'Worst layers')!;
		expect(toggle.getAttribute('aria-expanded')).toBe('false');

		toggle.click();
		flushSync();

		expect(toggle.getAttribute('aria-expanded')).toBe('true');
		const list = mounted.target.querySelector('[data-model-file-worst]');
		expect(list?.textContent).toContain('blocks.3.attn.qkv');
		expect(list?.textContent).toContain('0.0310');
		expect(list?.textContent).toContain('0.9994');
	});

	it('opens the indexed model in Models', () => {
		const mounted = mount(ModelFileArtifact, { artifact: { artifact_data: MODEL_DATA } });
		cleanup = mounted.destroy;

		button(mounted.target, 'Open in Models')!.click();

		expect(hoisted.goto).toHaveBeenCalledWith('/models/m-9');
	});

	it('shows rank and energy kept for a LoRA and just its own size', () => {
		const mounted = mount(ModelFileArtifact, { artifact: { artifact_data: LORA_DATA } });
		cleanup = mounted.destroy;

		expect(mounted.target.querySelector('[data-model-file-card]')?.getAttribute('data-kind')).toBe(
			'lora'
		);
		expect(mounted.target.querySelector('[data-model-file-rank]')?.textContent).toBe('32');
		expect(mounted.target.querySelector('[data-model-file-energy]')?.textContent).toBe('93.0%');
		expect(mounted.target.querySelector('[data-model-file-sizes]')?.textContent).toBe('204.8 MB');
	});

	it('degrades cleanly when every optional field is missing', () => {
		const mounted = mount(ModelFileArtifact, {
			artifact: { artifact_data: { kind: 'model', file_name: 'bare.safetensors' } }
		});
		cleanup = mounted.destroy;

		expect(mounted.target.querySelector('[data-model-file-name]')?.textContent).toBe(
			'bare.safetensors'
		);
		expect(mounted.target.querySelector('[data-model-file-sizes]')).toBeNull();
		expect(mounted.target.querySelector('[data-model-file-quality]')).toBeNull();
		expect(mounted.target.querySelector('[data-model-file-sources]')).toBeNull();
		expect(button(mounted.target, 'Worst layers')).toBeUndefined();
		expect(button(mounted.target, 'Open in Models')).toBeUndefined();
	});

	it('says so when the file could not be added to the library', () => {
		const mounted = mount(ModelFileArtifact, {
			artifact: {
				artifact_data: { ...MODEL_DATA, model_id: null, indexed: false, index_warning: 'Root offline' }
			}
		});
		cleanup = mounted.destroy;

		expect(button(mounted.target, 'Open in Models')).toBeUndefined();
		expect(mounted.target.querySelector('[data-model-file-index-warning]')?.textContent).toContain(
			'Root offline'
		);
	});
});

describe('LiveModelFileArtifacts', () => {
	let cleanup: (() => void) | null = null;
	afterEach(() => {
		cleanup?.();
		cleanup = null;
	});

	function tabWith(artifacts: unknown[]) {
		return { id: 't1', generation: { artifacts } } as never;
	}

	it('renders one card per model_file artifact and ignores other types', () => {
		const mounted = mount(LiveModelFileArtifacts, {
			tab: tabWith([
				{ artifact_type: 'model_file', artifact_data: MODEL_DATA },
				{ artifact_type: 'model_file', artifact_data: LORA_DATA },
				{ artifact_type: 'seed', artifact_data: { seed: 1 } }
			])
		});
		cleanup = mounted.destroy;

		expect(mounted.target.querySelectorAll('[data-model-file-card]').length).toBe(2);
	});

	it('renders nothing without a model_file artifact', () => {
		const mounted = mount(LiveModelFileArtifacts, {
			tab: tabWith([{ artifact_type: 'seed', artifact_data: { seed: 1 } }])
		});
		cleanup = mounted.destroy;

		expect(mounted.target.querySelector('[data-live-model-file-artifacts]')).toBeNull();
	});
});
