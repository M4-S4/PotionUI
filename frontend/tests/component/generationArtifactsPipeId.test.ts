import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync } from 'svelte';

const hoisted = vi.hoisted(() => ({ auth: null as null | import('svelte/store').Writable<unknown> }));

vi.mock('$lib/stores/auth', async () => {
	const { writable } = await import('svelte/store');
	const store = writable({ user: { account_type: 'USER' } });
	hoisted.auth = store;
	return { authStore: { subscribe: store.subscribe } };
});

vi.mock('$lib/services/api/index', () => ({
	api: {
		getGenerationRunReport: vi.fn().mockResolvedValue({
			success: true,
			data: {
				run_report: {
					prompt_template: null,
					artifacts: [{ artifact_type: 'seed', pipe_id: 7, artifact_data: { seed: 1 } }]
				}
			}
		})
	}
}));

const { default: GenerationArtifacts } = await import('../../src/lib/components/generation/artifacts/GenerationArtifacts.svelte');
const { mount, unmount } = await import('svelte');

let target: HTMLElement | undefined;
let instance: Record<string, unknown> | undefined;

async function render(accountType: string) {
	hoisted.auth!.set({ user: { account_type: accountType } });
	target = document.createElement('div');
	document.body.appendChild(target);
	instance = mount(GenerationArtifacts, { target, props: { generationId: 'g1' } });
	for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
	flushSync();
	target.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click();
	flushSync();
}

afterEach(() => {
	if (instance) unmount(instance);
	target?.remove();
	instance = undefined;
	target = undefined;
});

describe('artifact pipe id', () => {
	it('is hidden from non-admins', async () => {
		await render('USER');
		expect(target!.textContent).toContain('seed');
		expect(target!.textContent).not.toMatch(/\b7\b/);
	});

	it('is shown to admins', async () => {
		await render('ADMIN');
		expect(target!.textContent).toMatch(/\b7\b/);
	});
});
