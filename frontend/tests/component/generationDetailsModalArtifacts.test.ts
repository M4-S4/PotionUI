// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';
import { writable } from 'svelte/store';

const MODEL_PATH = '/srv/potion/models/diffusion_models/secret_dit_fp16.safetensors';

const hoisted = vi.hoisted(() => ({
	accountType: 'USER',
	clientGet: null as null | ((url: string) => Promise<unknown>),
	runReportCalls: 0
}));

function runReport() {
	return {
		schema_version: 2,
		status_history: [],
		pipe_timers: {},
		artifacts: [
			{
				at: '2026-01-01T00:00:00Z',
				pipe_id: 'model_loader',
				artifact_type: 'models',
				artifact_data: { models: [{ name: MODEL_PATH, type: 'checkpoint' }] }
			},
			{
				at: '2026-01-01T00:00:00Z',
				pipe_id: 'generator',
				artifact_type: 'diff_text',
				artifact_data: { name: 'Positive Prompt', diff: 'a cat in the rain' }
			}
		],
		plugin_outputs: {},
		prompt_template: null,
		artifacts_omitted: 0,
		plugin_outputs_omitted: 0,
		artifacts_bytes: 0,
		stored_bytes: 0
	};
}

vi.mock('$lib/stores/auth', () => ({
	authStore: Object.assign(
		writable({ user: { id: 'u1', username: 'someone', account_type: hoisted.accountType } }),
		{ logout: vi.fn() }
	)
}));

vi.mock('$lib/services/api/index', () => {
	const get = vi.fn((url: string) => {
		if (url.includes('/run-report')) {
			hoisted.runReportCalls += 1;
			return Promise.resolve({ data: { success: true, data: { run_report: runReport() } } });
		}
		return Promise.reject(new Error('not mocked'));
	});
	return {
		api: {
			getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
			getClient: vi.fn(() => ({
				get,
				post: vi.fn().mockRejectedValue(new Error('not mocked')),
				put: vi.fn().mockRejectedValue(new Error('not mocked'))
			})),
			getGenerationParams: vi
				.fn()
				.mockResolvedValue({ success: true, data: { parameters: {}, models: [] } }),
			getGenerationById: vi.fn().mockResolvedValue({ success: false, error: 'not mocked' }),
			getGenerationRunReport: vi.fn(() => {
				hoisted.runReportCalls += 1;
				return Promise.resolve({ success: true, data: { run_report: runReport() } });
			}),
			getTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } }),
			getBaseURL: vi.fn(() => ''),
			getToken: vi.fn(() => null),
			setOnAuthExpired: vi.fn(),
			setOnTokenRefreshed: vi.fn()
		}
	};
});

const { default: GenerationDetailsModal } = await import(
	'$lib/components/modals/GenerationDetailsModal.svelte'
);
const { authStore } = await import('$lib/stores/auth');
const { createClassComponent } = await import('svelte/legacy');

function generation(id: string) {
	return {
		id,
		form_data: {},
		status: 'completed' as const,
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files: [],
		segments: [],
		rating: 0,
		is_favorite: false
	};
}

let component: ReturnType<typeof createClassComponent> | null = null;
let target: HTMLDivElement | null = null;

async function openDetails(accountType: 'USER' | 'ADMIN') {
	(authStore as unknown as ReturnType<typeof writable>).set({
		user: { id: 'u1', username: 'someone', account_type: accountType }
	});
	hoisted.runReportCalls = 0;
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: GenerationDetailsModal as never,
		target,
		props: { generation: generation('gen-1') as never, isOpen: true }
	});
	flushSync();
	for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
	flushSync();
}

function artifactsSection(): Element | undefined {
	return Array.from(document.body.querySelectorAll('button, h3')).find(
		(element) => element.textContent?.trim().startsWith('Artifacts')
	);
}

afterEach(() => {
	component?.$destroy();
	component = null;
	target?.remove();
	target = null;
});

describe('GenerationDetailsModal artifacts', () => {
	it.each(['USER', 'ADMIN'] as const)('shows no Artifacts section to a %s', async (accountType) => {
		await openDetails(accountType);

		expect(document.body.textContent).toContain('Parameters');
		expect(artifactsSection()).toBeUndefined();
		expect(document.body.textContent).not.toContain(MODEL_PATH);
		expect(document.body.textContent).not.toContain('a cat in the rain');
		expect(hoisted.runReportCalls).toBe(0);
	});
});
