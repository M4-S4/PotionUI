import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const getReadiness = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: { getReadiness: (...args: unknown[]) => getReadiness(...args) }
}));

const { refreshResumeSetup, showResumeSetup } = await import('./resumeSetup');

beforeEach(() => {
	getReadiness.mockReset();
	showResumeSetup.set(false);
});

describe('refreshResumeSetup', () => {
	it('shows the row while readiness is not fully green', async () => {
		getReadiness.mockResolvedValue({ overall: 'not_ready', checks: [] });
		await refreshResumeSetup();
		expect(get(showResumeSetup)).toBe(true);
		getReadiness.mockResolvedValue({ overall: 'degraded', checks: [] });
		await refreshResumeSetup();
		expect(get(showResumeSetup)).toBe(true);
	});

	it('hides the row once ready', async () => {
		showResumeSetup.set(true);
		getReadiness.mockResolvedValue({ overall: 'ready', checks: [] });
		await refreshResumeSetup();
		expect(get(showResumeSetup)).toBe(false);
	});

	it('fails soft to hidden when the check fails', async () => {
		showResumeSetup.set(true);
		getReadiness.mockImplementation(async () => {
			throw new Error('down');
		});
		await refreshResumeSetup();
		expect(get(showResumeSetup)).toBe(false);
	});
});
