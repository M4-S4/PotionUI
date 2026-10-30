import { describe, it, expect } from 'vitest';
import { llmConfigDetailTabHasFooter } from './llmConfigDetailTabs';

describe('llmConfigDetailTabHasFooter', () => {
	it('shows the footer on the editable Configuration tab', () => {
		expect(llmConfigDetailTabHasFooter('configuration')).toBe(true);
	});

	it('hides the footer on the read-only Access tab', () => {
		expect(llmConfigDetailTabHasFooter('access')).toBe(false);
	});

	it('hides the shared footer on the Profile tab, which renders its own', () => {
		expect(llmConfigDetailTabHasFooter('profile')).toBe(false);
	});

	it('hides the footer on the Toolset tab, which saves per-toggle', () => {
		expect(llmConfigDetailTabHasFooter('toolset')).toBe(false);
	});
});
