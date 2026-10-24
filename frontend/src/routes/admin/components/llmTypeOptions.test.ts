import { describe, expect, it } from 'vitest';
import type { LLMProviderType } from '$lib/types/llm';
import { BUILTIN_LLM_TYPES, baseUrlForType, llmTypeOptions } from './llmTypeOptions';

const DEEPSEEK_URL = 'https://api.deepseek.com';

const provider = (type: string, label: string | null = null, default_base_url: string | null = null): LLMProviderType => ({
	type,
	label,
	default_base_url,
	supports_model_listing: true
});

const DEEPSEEK = provider('deepseek', 'DeepSeek', DEEPSEEK_URL);

describe('llmTypeOptions', () => {
	it('keeps the built-in types in their order with their labels', () => {
		expect(llmTypeOptions([provider('openai'), provider('ollama')], 'openai')).toEqual(BUILTIN_LLM_TYPES);
	});

	it('appends a type a plugin contributes, under its label', () => {
		expect(llmTypeOptions([provider('openai'), DEEPSEEK], 'openai').at(-1)).toEqual({ value: 'deepseek', label: 'DeepSeek' });
	});

	it('falls back to the type name when the contributed type has no label', () => {
		expect(llmTypeOptions([provider('example')], 'openai').at(-1)).toEqual({ value: 'example', label: 'example' });
	});

	it('never lists a type twice', () => {
		const values = llmTypeOptions([provider('openai', 'Other OpenAI'), DEEPSEEK, DEEPSEEK], 'deepseek').map((o) => o.value);

		expect(values).toEqual([...BUILTIN_LLM_TYPES.map((o) => o.value), 'deepseek']);
	});

	it('keeps a saved type whose plugin is no longer enabled', () => {
		expect(llmTypeOptions([], 'deepseek').at(-1)).toEqual({ value: 'deepseek', label: 'deepseek' });
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
});
