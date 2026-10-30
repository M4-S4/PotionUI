import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync } from 'svelte';
import type { GenerationState } from '$lib/types/tabs';

const authState = vi.hoisted(() => ({ current: null as null | import('svelte/store').Writable<unknown> }));

vi.mock('$lib/stores/auth', async () => {
	const actual = await vi.importActual<typeof import('$lib/stores/auth')>('$lib/stores/auth');
	const { writable } = await import('svelte/store');
	const store = writable({ user: { account_type: 'USER' } });
	authState.current = store;
	return { ...actual, authStore: Object.assign({}, actual.authStore, { subscribe: store.subscribe }) };
});

const { default: GenerationPanel } = await import('../../src/lib/components/GenerationPanel.svelte');
const { createClassComponent } = await import('svelte/legacy');

const preset = { id: 'flux2', name: 'FLUX.2', cover: null, category: 'image' };

function running(extra: Record<string, unknown> = {}): GenerationState {
	return {
		isGenerating: true,
		currentGeneration: { id: 'run-1', status: 'running' },
		currentProgress: {
			step: 'Loading',
			progress: 0.3,
			current_step: 'Loading checkpoint <<MODEL:flux-dev.safetensors>> <<PROGRESS:9/30>>',
			message: '<<PIPE:Acme loader:bolt>>',
			pipe_name: 'checkpoint_loader/flux',
			pipe_id: 2,
			...extra
		},
		pipeTimers: {},
		startedAt: Date.now(),
		totalTime: null,
		lastDurationMs: null,
		batchImages: [],
		batchVideos: [],
		batchAudios: [],
		artifacts: [],
		workbenchIndex: 0,
		workbenchTotal: 0,
		queue: [],
		submittedPromptTemplate: null
	} as GenerationState;
}

let target: HTMLElement | undefined;
let component: { $destroy: () => void } | undefined;

function mount(accountType: string, extra: Record<string, unknown> = {}) {
	authState.current!.set({ user: { account_type: accountType } });
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: GenerationPanel as never,
		target,
		props: { generation: running(extra), isGenerating: true, canGenerate: true, tabId: '', preset, modeLabel: 'Text to image' }
	});
	flushSync();
}

const detail = () => target!.querySelector('.deck-detail-text')!.textContent!.replace(/\s+/g, ' ').trim();
const adminMark = () => target!.querySelector('[data-admin-stage-detail]');

afterEach(() => {
	component?.$destroy();
	target?.remove();
	component = undefined;
	target = undefined;
});

describe('generation panel stage text', () => {
	it('shows plain stage text and no technical detail to non-admins', () => {
		mount('USER');
		expect(detail()).toBe('30% · Loading model · Step 9 / 30');
		expect(target!.textContent).not.toMatch(/<<|PIPE:|flux-dev|checkpoint_loader|Acme/);
		expect(adminMark()).toBeNull();
	});

	it('shows the same stage text to admins, with the technical detail behind a marker', () => {
		mount('ADMIN');
		expect(detail()).toContain('30% · Loading model');
		expect(detail()).toContain('Step 9 / 30');
		const mark = adminMark();
		expect(mark).not.toBeNull();
		expect(mark!.querySelector('[data-admin-only-mark]')!.getAttribute('aria-label')).toContain('Pipe: ');
	});

	it('shows the batch label next to the percent only for multi-item runs', () => {
		mount('USER', { prompt_index: 1, prompt_count: 3, item_progress: 0.5 });
		expect(target!.querySelector('[data-batch-progress]')!.textContent).toBe('Prompt 2 of 3');
		component!.$destroy();
		target!.remove();
		mount('USER', { item_index: 0, item_count: 1 });
		expect(target!.querySelector('[data-batch-progress]')).toBeNull();
	});
});
