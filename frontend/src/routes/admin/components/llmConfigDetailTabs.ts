export type LLMConfigDetailTab = 'configuration' | 'profile' | 'toolset' | 'access';

export function llmConfigDetailTabHasFooter(tab: LLMConfigDetailTab): boolean {
	return tab === 'configuration';
}
