import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LLMConfigFormData } from '../../src/routes/admin/components/LLMConfigForm.svelte';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getLLMProviderTypes: vi.fn(async () => ({ success: true, data: { types: [] } })),
		listNativeCheckpoints: vi.fn(async () => ({ success: true, data: [] }))
	}
}));

const { default: LLMConfigForm } = await import('../../src/routes/admin/components/LLMConfigForm.svelte');
const { createClassComponent } = await import('svelte/legacy');

function draft(type: string): LLMConfigFormData {
	return {
		name: 'Test config',
		type,
		model: 'm',
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
		provider_options: {}
	};
}

let destroy: (() => void) | undefined;

async function sections(type: string): Promise<string[]> {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LLMConfigForm as never,
		target,
		props: { draft: draft(type), mode: 'create', layout: 'panel', idPrefix: 'test' }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	return [...target.querySelectorAll('h3')].map((heading) => heading.textContent?.trim() ?? '');
}

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
});

describe('LLMConfigForm — provider option sections by type', () => {
	it.each(['openai', 'openai-compatible', 'openrouter'])('shows the OpenAI options for %s', async (type) => {
		expect(await sections(type)).toContain('OpenAI Options');
	});

	it.each(['anthropic', 'gemini', 'deepseek'])('shows no provider-specific options for %s', async (type) => {
		const headings = await sections(type);

		expect(headings).not.toContain('OpenAI Options');
		expect(headings).not.toContain('Ollama Options');
		expect(headings).not.toContain('Native Options');
	});

	it('shows the Ollama options for ollama and the native options for native', async () => {
		expect(await sections('ollama')).toContain('Ollama Options');
		destroy?.();
		expect(await sections('native')).toContain('Native Options');
	});
});

describe('LLMConfigForm — settings that live in the capability profile', () => {
	async function text(type: string): Promise<string> {
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: LLMConfigForm as never,
			target,
			props: { draft: draft(type), mode: 'edit', layout: 'panel', idPrefix: 'test', configId: 'c1', apiKeySet: true }
		});
		destroy = () => {
			component.$destroy();
			target.remove();
		};
		await settle();
		return target.textContent ?? '';
	}

	it.each(['ollama', 'native', 'anthropic', 'openai-compatible'])('has no context window field for %s', async (type) => {
		const content = await text(type);

		expect(content).not.toContain('Context window');
		expect(document.querySelector('#test-context-window')).toBeNull();
	});

	it('has no force-prompt-tools switch among the Ollama options', async () => {
		const content = await text('ollama');

		expect(content).toContain('Ollama Options');
		expect(content).not.toContain('Force Prompt Tools');
		expect(document.querySelector('#test-ollama-force_prompt_tools')).toBeNull();
	});

	it('keeps the Ollama thinking option', async () => {
		expect(await text('ollama')).toContain('Thinking Mode');
	});
});
