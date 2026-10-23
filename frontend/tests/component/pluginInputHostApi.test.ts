import { describe, it, expect } from 'vitest';
import { initHostApi } from '$lib/plugin-api/host';
import { isEditableTarget, isTypingInField } from '$lib/utils/editableTarget';

describe('window.__potionui.input plugin contract', () => {
	it('exposes the host shared typing guard, not a copy', () => {
		initHostApi();
		const host = (window as any).__potionui;
		expect(host.input.isTypingInField).toBe(isTypingInField);
		expect(host.input.isEditableTarget).toBe(isEditableTarget);
	});
});
