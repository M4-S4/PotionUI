import type { LLMProviderType } from '$lib/types/llm';

export interface LLMTypeOption {
	value: string;
	label: string;
}

export function llmTypeOptions(providerTypes: LLMProviderType[], current: string): LLMTypeOption[] {
	const options: LLMTypeOption[] = [];
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

const OPENAI_WIRE_TYPES = ['openai', 'openai-compatible', 'openrouter'];

export function usesOpenAIOptions(type: string): boolean {
	return OPENAI_WIRE_TYPES.includes(type);
}
