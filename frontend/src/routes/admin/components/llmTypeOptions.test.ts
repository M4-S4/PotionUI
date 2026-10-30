import { describe, expect, it } from 'vitest';
import type { LLMProviderType } from '$lib/types/llm';
import { baseUrlForType, llmTypeOptions, usesOpenAIOptions } from './llmTypeOptions';

const DEEPSEEK_URL = 'https://api.deepseek.com';

const provider = (
	type: string,
	label: string | null = null,
	default_base_url: string | null = null,
	adapter = 'openai-chat',
	source = 'core'
): LLMProviderType => ({
	type,
	label,
	adapter,
	default_base_url,
	supports_model_listing: true,
	source
});

const DEEPSEEK = provider('deepseek', 'DeepSeek', DEEPSEEK_URL);

describe('llmTypeOptions', () => {
	const CORE = [
		provider('anthropic', 'Anthropic', 'https://api.anthropic.com', 'anthropic-messages'),
		provider('openai', 'OpenAI', null, 'openai-responses'),
		provider('openai-compatible', 'OpenAI-compatible server'),
		provider('gemini', 'Gemini', null, 'gemini-interactions'),
		provider('openrouter', 'OpenRouter', null, 'openrouter'),
		provider('ollama', 'Ollama', null, 'ollama-chat'),
		provider('native', 'Native', null, 'native')
	];

	it('lists exactly the types the server returns, in its order, under their labels', () => {
		expect(llmTypeOptions(CORE, 'openai')).toEqual([
			{ value: 'anthropic', label: 'Anthropic' },
			{ value: 'openai', label: 'OpenAI' },
			{ value: 'openai-compatible', label: 'OpenAI-compatible server' },
			{ value: 'gemini', label: 'Gemini' },
			{ value: 'openrouter', label: 'OpenRouter' },
			{ value: 'ollama', label: 'Ollama' },
			{ value: 'native', label: 'Native' }
		]);
	});

	it('lists a type a plugin contributes after the core ones, under its label', () => {
		const options = llmTypeOptions([...CORE, DEEPSEEK], 'openai');

		expect(options.at(-1)).toEqual({ value: 'deepseek', label: 'DeepSeek' });
		expect(options).toHaveLength(CORE.length + 1);
	});

	it('falls back to the type name when the contributed type has no label', () => {
		expect(llmTypeOptions([provider('example')], 'example').at(-1)).toEqual({ value: 'example', label: 'example' });
	});

	it('never lists a type twice', () => {
		const values = llmTypeOptions([provider('openai', 'Other OpenAI'), DEEPSEEK, DEEPSEEK], 'deepseek').map((o) => o.value);

		expect(values).toEqual(['openai', 'deepseek']);
	});

	it('keeps a saved type the server no longer offers', () => {
		expect(llmTypeOptions([], 'deepseek')).toEqual([{ value: 'deepseek', label: 'deepseek' }]);
		expect(llmTypeOptions(CORE, 'local').at(-1)).toEqual({ value: 'local', label: 'local' });
	});
});

describe('usesOpenAIOptions', () => {
	it('is true for every type on the chat-completions style wire', () => {
		expect(['openai', 'openai-compatible', 'openrouter'].map(usesOpenAIOptions)).toEqual([true, true, true]);
	});

	it('is false for the other types', () => {
		expect(['anthropic', 'gemini', 'ollama', 'native', 'deepseek'].map(usesOpenAIOptions)).toEqual([
			false,
			false,
			false,
			false,
			false
		]);
	});
});

describe('baseUrlForType', () => {
	it('fills an empty base URL with the default of the chosen type', () => {
		expect(baseUrlForType('deepseek', '  ', [DEEPSEEK])).toBe(DEEPSEEK_URL);
	});

	it('leaves a base URL the admin already typed', () => {
		expect(baseUrlForType('deepseek', 'https://proxy.example.org', [DEEPSEEK])).toBe('https://proxy.example.org');
	});

	it('leaves the field alone for a type without a default', () => {
		expect(baseUrlForType('openai', '', [provider('openai'), DEEPSEEK])).toBe('');
	});

	it('clears the default of the previous type when switching to a type without one', () => {
		expect(baseUrlForType('openai', DEEPSEEK_URL, [provider('openai'), DEEPSEEK])).toBe('');
	});

	it('swaps the default of the previous type for the default of the next one', () => {
		const other = provider('other', 'Other', 'https://other.example.org/v1');

		expect(baseUrlForType('other', DEEPSEEK_URL, [DEEPSEEK, other])).toBe('https://other.example.org/v1');
	});
});
