// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LLMConfigFormData } from '../../src/routes/admin/components/LLMConfigForm.svelte';
import type { LLMContextWindowDiscovery } from '$lib/types/llm';

vi.mock('$lib/services/api/index', () => ({
	api: {
		listNativeCheckpoints: vi.fn().mockResolvedValue({ success: true, data: [] }),
		getLLMProviderTypes: vi.fn().mockResolvedValue({ success: true, data: { types: [] } }),
		discoverLLMContextWindow: vi.fn()
	}
}));

const api = await import('$lib/services/api/index');
const { default: LLMConfigForm } = await import(
	'../../src/routes/admin/components/LLMConfigForm.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

function makeDraft(overrides: Partial<LLMConfigFormData> = {}): LLMConfigFormData {
	return {
		name: 'Test config',
		type: 'ollama',
		model: 'llama3',
		api_key: '',
		base_url: 'http://localhost:11434',
		enabled: true,
		supports_vision: false,
		disable_system_prompt: false,
		memory_reflection: true,
		system_message: 'You are helpful.',
		temperature: 0.7,
		max_tokens: 1000,
		timeout: 30,
		provider_options: {},
		...overrides
	};
}

function discovery(overrides: Partial<LLMContextWindowDiscovery> = {}): LLMContextWindowDiscovery {
	return {
		supported: true,
		detected: 32768,
		detected_kind: 'runtime',
		effective: 32768,
		discovered_at: null,
		error: null,
		...overrides
	};
}

function mockDiscover(data: LLMContextWindowDiscovery) {
	vi.mocked(api.api.discoverLLMContextWindow).mockResolvedValue({ success: true, data } as never);
}

function mount(draft: LLMConfigFormData) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LLMConfigForm as never,
		target,
		props: { draft, mode: 'create', idPrefix: 'test' }
	});
	return {
		target,
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

async function settle() {
	await vi.advanceTimersByTimeAsync(1000);
	for (let i = 0; i < 8; i++) await Promise.resolve();
}

function readout(target: HTMLElement): string {
	return (target.querySelector('[data-testid="context-window-readout"]')?.textContent ?? '').replace(/\s+/g, ' ').trim();
}

let mounted: ReturnType<typeof mount> | undefined;

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe('LLMConfigForm — context window', () => {
	it.each(['openai', 'anthropic', 'ollama', 'native', 'local'])('shows the field for %s', async (type) => {
		mockDiscover(discovery());
		mounted = mount(makeDraft({ type }));
		await settle();
		expect(mounted.target.querySelector('#test-context-window')).toBeTruthy();
	});

	it('no longer renders the Ollama Context Size field', async () => {
		mockDiscover(discovery());
		mounted = mount(makeDraft());
		await settle();
		expect(mounted.target.querySelector('#test-ollama-num_ctx')).toBeNull();
		expect(mounted.target.textContent).not.toContain('Context Size');
	});

	it('shows the GPU-memory note for Ollama only', async () => {
		mockDiscover(discovery());
		mounted = mount(makeDraft());
		await settle();
		expect(mounted.target.textContent).toContain('uses more GPU memory, shared with image generation');
		mounted.destroy();
		mounted = mount(makeDraft({ type: 'openai' }));
		await settle();
		expect(mounted.target.textContent).not.toContain('shared with image generation');
	});

	it('typing writes context_window and drops a stale num_ctx; clearing deletes the key', async () => {
		mockDiscover(discovery());
		const draft = makeDraft({ provider_options: { num_ctx: 4096 } });
		mounted = mount(draft);
		await settle();
		const input = mounted.target.querySelector('#test-context-window') as HTMLInputElement;
		input.value = '16384';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		await settle();
		expect(draft.provider_options.context_window).toBe(16384);
		expect('num_ctx' in draft.provider_options).toBe(false);

		input.value = '';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		await settle();
		expect('context_window' in draft.provider_options).toBe(false);
	});

	it('detects once on open with the typed key and model', async () => {
		mockDiscover(discovery());
		mounted = mount(makeDraft({ api_key: 'sk-typed' }));
		await settle();
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledTimes(1);
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledWith({
			type: 'ollama',
			base_url: 'http://localhost:11434',
			api_key: 'sk-typed',
			config_id: null,
			model: 'llama3'
		});
	});

	it('does not loop: re-renders and typing in the field do not re-request', async () => {
		mockDiscover(discovery());
		const draft = makeDraft();
		mounted = mount(draft);
		await settle();
		const input = mounted.target.querySelector('#test-context-window') as HTMLInputElement;
		for (const v of ['1', '12', '123']) {
			input.value = v;
			input.dispatchEvent(new Event('input', { bubbles: true }));
			await settle();
		}
		await settle();
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledTimes(1);
	});

	it('the Detect button requests again', async () => {
		mockDiscover(discovery());
		mounted = mount(makeDraft());
		await settle();
		const button = Array.from(mounted.target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Detect');
		button!.click();
		await settle();
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledTimes(2);
	});

	it('readout: runtime-detected window with a blank field', async () => {
		mockDiscover(discovery());
		mounted = mount(makeDraft());
		await settle();
		expect(readout(mounted.target)).toBe(`Using ${(32768).toLocaleString()} tokens, detected from the server`);
	});

	it('readout: a set field shows only the detected hint', async () => {
		mockDiscover(discovery({ detected: 131072, effective: 4096 }));
		mounted = mount(makeDraft({ provider_options: { context_window: 4096 } }));
		await settle();
		expect(readout(mounted.target)).toBe(`Detected ${(131072).toLocaleString()} tokens`);
	});

	it('readout: a failed request reports the fallback too', async () => {
		vi.mocked(api.api.discoverLLMContextWindow).mockRejectedValue(new Error('network'));
		mounted = mount(makeDraft());
		await settle();
		expect(readout(mounted.target)).toContain('until a window is set');
	});
});

describe('LLMContextWindow — request guard', () => {
	async function mountField(props: Record<string, unknown>) {
		const { default: LLMContextWindow } = await import(
			'../../src/routes/admin/components/llm/LLMContextWindow.svelte'
		);
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: LLMContextWindow as never,
			target,
			props: { id: 'cw', type: 'ollama', model: 'llama3', baseUrl: 'http://x', value: undefined, onchange: () => {}, ...props }
		});
		mounted = {
			target,
			destroy: () => {
				component.$destroy();
				target.remove();
			}
		};
		return component as unknown as { $set: (props: Record<string, unknown>) => void };
	}

	it('does not re-request when only the value changes', async () => {
		mockDiscover(discovery());
		const component = await mountField({});
		await settle();
		for (const value of [1, 12, 123, undefined]) {
			component.$set({ value });
			await settle();
		}
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledTimes(1);
	});

	it('requests once more, debounced, when the model changes', async () => {
		mockDiscover(discovery());
		const component = await mountField({});
		await settle();
		component.$set({ model: 'l' });
		component.$set({ model: 'll' });
		component.$set({ model: 'qwen3' });
		await settle();
		expect(api.api.discoverLLMContextWindow).toHaveBeenCalledTimes(2);
		expect(vi.mocked(api.api.discoverLLMContextWindow).mock.calls[1][0].model).toBe('qwen3');
	});

	it('does not request for a blank model', async () => {
		mockDiscover(discovery());
		await mountField({ model: '  ' });
		await settle();
		expect(api.api.discoverLLMContextWindow).not.toHaveBeenCalled();
	});
});
