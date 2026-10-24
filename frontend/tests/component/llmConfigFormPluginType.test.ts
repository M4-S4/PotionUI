import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LLMConfigFormData } from '../../src/routes/admin/components/LLMConfigForm.svelte';
import { settle } from './helpers/settle';

const DEEPSEEK_URL = 'https://api.deepseek.com';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getLLMProviderTypes: vi.fn(async () => ({
			success: true,
			data: {
				types: [
					{ type: 'openai', label: null, default_base_url: null, supports_model_listing: true },
					{ type: 'deepseek', label: 'DeepSeek', default_base_url: DEEPSEEK_URL, supports_model_listing: true }
				]
			}
		})),
		listNativeCheckpoints: vi.fn()
	}
}));

const { default: LLMConfigForm } = await import('../../src/routes/admin/components/LLMConfigForm.svelte');
const { createClassComponent } = await import('svelte/legacy');

function makeDraft(overrides: Partial<LLMConfigFormData> = {}): LLMConfigFormData {
	return {
		name: 'Test config',
		type: 'openai',
		model: '',
		api_key: '',
		base_url: '',
		enabled: true,
		supports_vision: false,
		disable_system_prompt: false,
		memory_reflection: true,
		show_thinking: false,
		system_message: 'You are helpful.',
		temperature: 0.7,
		max_tokens: 1000,
		timeout: 30,
		provider_options: {},
		...overrides
	};
}

let destroy: (() => void) | undefined;

async function mount(draft: LLMConfigFormData) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LLMConfigForm as never,
		target,
		props: { draft, mode: 'create', idPrefix: 'test' }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	return target.querySelector<HTMLSelectElement>('#test-type')!;
}

function choose(select: HTMLSelectElement, value: string) {
	select.value = value;
	select.dispatchEvent(new Event('change', { bubbles: true }));
}

afterEach(() => {
	destroy?.();
	destroy = undefined;
});

describe('LLMConfigForm — a type contributed by a plugin', () => {
	it('lists the contributed type under its label after the built-in ones', async () => {
		const select = await mount(makeDraft());

		const options = [...select.options].map((option) => [option.value, option.textContent?.trim()]);

		expect(options.at(-1)).toEqual(['deepseek', 'DeepSeek']);
		expect(options.filter(([value]) => value === 'openai')).toHaveLength(1);
	});

	it('fills an empty base URL with the default of the type the admin picks', async () => {
		const draft = makeDraft();
		const select = await mount(draft);

		choose(select, 'deepseek');
		await settle();

		expect(draft.type).toBe('deepseek');
		expect(draft.base_url).toBe(DEEPSEEK_URL);
	});

	it('keeps a base URL the admin already typed', async () => {
		const draft = makeDraft({ base_url: 'https://proxy.example.org' });
		const select = await mount(draft);

		choose(select, 'deepseek');
		await settle();

		expect(draft.base_url).toBe('https://proxy.example.org');
	});
});
