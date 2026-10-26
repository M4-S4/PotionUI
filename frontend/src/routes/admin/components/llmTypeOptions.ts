import type { LLMProviderType } from '$lib/types/llm';

export interface LLMTypeOption {
	value: string;
	label: string;
}

export const BUILTIN_LLM_TYPES: LLMTypeOption[] = [
	{ value: 'openai', label: 'OpenAI' },
	{ value: 'anthropic', label: 'Anthropic' },
	{ value: 'ollama', label: 'Ollama' },
	{ value: 'native', label: 'Native' },
	{ value: 'local', label: 'Local' }
];

export function llmTypeOptions(providerTypes: LLMProviderType[], current: string): LLMTypeOption[] {
	const options = [...BUILTIN_LLM_TYPES];
	const listed = (value: string) => options.some((option) => option.value === value);
	for (const provider of providerTypes) {
		if (!listed(provider.type)) options.push({ value: provider.type, label: provider.label ?? provider.type });
	}
	if (current && !listed(current)) options.push({ value: current, label: current });
	return options;
}

export function baseUrlForType(type: string, baseUrl: string, providerTypes: LLMProviderType[]): string {
	const current = baseUrl.trim();
	const isDefault = providerTypes.some((provider) => provider.default_base_url === current);
	if (current && !isDefault) return baseUrl;
	return providerTypes.find((provider) => provider.type === type)?.default_base_url ?? '';
}
