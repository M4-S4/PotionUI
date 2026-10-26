import { writable } from 'svelte/store';
import { api } from '$lib/services/api/index';

export const showResumeSetup = writable(false);

export async function refreshResumeSetup(): Promise<void> {
	try {
		const report = await api.getReadiness();
		showResumeSetup.set(report.overall !== 'ready');
	} catch {
		showResumeSetup.set(false);
	}
}
