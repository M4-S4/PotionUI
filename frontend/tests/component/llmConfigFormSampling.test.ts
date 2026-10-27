import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LLMConfigFormData } from '../../src/routes/admin/components/LLMConfigForm.svelte';
import type { LLMProfile } from '$lib/types/llm';
import { settle } from './helpers/settle';

const mockGetLLMProfile = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getLLMProviderTypes: vi.fn(async () => ({ success: true, data: { types: [] } })),
		listNativeCheckpoints: vi.fn(async () => ({ success: true, data: [] }))
	}
}));

vi.mock('$lib/services/admin-api', () => ({
	getLLMProfile: (...args: unknown[]) => mockGetLLMProfile(...args)
}));

const { default: LLMConfigForm } = await import('../../src/routes/admin/components/LLMConfigForm.svelte');
const { createClassComponent } = await import('svelte/legacy');

function profile(recommended: unknown): LLMProfile {
	return {
		config_id: 'c1',
		adapter: 'ollama-chat',
		model: 'qwen3',
		discovered_at: null,
		discovery_error: null,
		fields: recommended === undefined ? {} : {
			recommended_sampling: { value: recommended, source: 'seed', seed: recommended, discovered: null, override: null }
		},
		verified: null
	};
}

function makeDraft(overrides: Partial<LLMConfigFormData> = {}): LLMConfigFormData {
	return {
		name: 'Test config',
		type: 'ollama',
		model: 'qwen3',
		api_key: '',
		base_url: '',
		enabled: true,
		supports_vision: false,
		disable_system_prompt: false,
		memory_reflection: true,
		show_thinking: false,
		system_message: 'You are helpful.',
		temperature: null,
		max_tokens: null,
		timeout: 30,
		provider_options: {},
		...overrides
	};
}

let destroy: (() => void) | undefined;

async function mount(draft: LLMConfigFormData, configId: string | null = 'c1') {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LLMConfigForm as never,
		target,
		props: { draft, mode: 'edit', layout: 'panel', idPrefix: 't', configId, apiKeySet: true }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	return {
		temperature: target.querySelector('#t-temperature') as HTMLInputElement,
		maxTokens: target.querySelector('#t-max-tokens') as HTMLInputElement
	};
}

async function type(input: HTMLInputElement, value: string) {
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	await settle();
}

beforeEach(() => {
	mockGetLLMProfile.mockResolvedValue({ success: true, data: profile({ temperature: 0.6, top_p: 0.95 }) });
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('LLMConfigForm — sampling left to the model default', () => {
	it('shows an empty temperature as the model default with the recommended value from the profile', async () => {
		const { temperature } = await mount(makeDraft());

		expect(mockGetLLMProfile).toHaveBeenCalledWith('c1');
		expect(temperature.value).toBe('');
		expect(temperature.placeholder).toBe('Model default · 0.6');
	});

	it('shows a plain model default for max tokens', async () => {
		const { maxTokens } = await mount(makeDraft());

		expect(maxTokens.value).toBe('');
		expect(maxTokens.placeholder).toBe('Model default');
	});

	it('shows a plain model default when the profile recommends nothing or cannot be read', async () => {
		mockGetLLMProfile.mockResolvedValueOnce({ success: true, data: profile(undefined) });
		expect((await mount(makeDraft())).temperature.placeholder).toBe('Model default');
		destroy?.();

		mockGetLLMProfile.mockRejectedValueOnce(new Error('offline'));
		expect((await mount(makeDraft())).temperature.placeholder).toBe('Model default');
		destroy?.();

		mockGetLLMProfile.mockResolvedValueOnce({ success: false });
		expect((await mount(makeDraft())).temperature.placeholder).toBe('Model default');
	});

	it('does not ask for a profile while the configuration is not saved yet', async () => {
		const { temperature } = await mount(makeDraft(), null);

		expect(mockGetLLMProfile).not.toHaveBeenCalled();
		expect(temperature.placeholder).toBe('Model default');
	});

	it('shows the stored values, zero included', async () => {
		const { temperature, maxTokens } = await mount(makeDraft({ temperature: 0, max_tokens: 2048 }));

		expect(temperature.value).toBe('0');
		expect(maxTokens.value).toBe('2048');
	});

	it('writes a typed number to the draft and null once the box is emptied', async () => {
		const draft = makeDraft({ temperature: 0.3, max_tokens: 2048 });
		const { temperature, maxTokens } = await mount(draft);

		await type(temperature, '0.8');
		await type(maxTokens, '4096');
		expect([draft.temperature, draft.max_tokens]).toEqual([0.8, 4096]);

		await type(temperature, '');
		await type(maxTokens, '');
		expect([draft.temperature, draft.max_tokens]).toEqual([null, null]);
	});

	it('keeps a typed zero as zero, not as the model default', async () => {
		const draft = makeDraft({ temperature: 0.3 });
		const { temperature } = await mount(draft);

		await type(temperature, '0');

		expect(draft.temperature).toBe(0);
	});

	it('does not clamp max tokens to the old 4096 ceiling', async () => {
		const { maxTokens } = await mount(makeDraft());

		expect(maxTokens.hasAttribute('max')).toBe(false);
	});
});
