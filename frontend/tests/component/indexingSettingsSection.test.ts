import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { AWAITING, GIB, MIB, RUNNING, status } from './helpers/indexingFixtures';

vi.mock('$lib/services/adminWebsocket', () => ({
	adminWebSocket: {
		onIndexingStatus: () => () => {},
		onConnected: () => () => {},
		isConnected: () => true,
		connectAsync: async () => {}
	}
}));

vi.mock('$lib/services/api', () => ({
	api: {
		getIndexingStatus: vi.fn(),
		setIndexingSettings: vi.fn(),
		pauseIndexing: vi.fn(),
		resumeIndexing: vi.fn(),
		queueIndexingDigests: vi.fn()
	}
}));

const { api } = await import('$lib/services/api');
const { indexingStatusStore } = await import('$lib/models-location/indexingStatus.svelte');
const { default: IndexingSettingsSection } = await import('$lib/models-location/IndexingSettingsSection.svelte');
const { toasts } = await import('$lib/stores/toast');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const SAVED = { success: true, data: { state: 'done' } };

function seed(next: ReturnType<typeof status> | null) {
	indexingStatusStore.status = next;
	indexingStatusStore.pickRequested = false;
	indexingStatusStore.busy = false;
	indexingStatusStore.actionError = null;
}

function mountSection() {
	return mountComponent(IndexingSettingsSection, {});
}

const text = () => document.body.textContent?.replace(/\s+/g, ' ') ?? '';
const button = (label: string) =>
	Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim() === label);
const bar = () => document.querySelector<HTMLElement>('[role="progressbar"]');
const alerts = () => Array.from(document.querySelectorAll('[role="status"]')).map((el) => el.textContent?.replace(/\s+/g, ' ').trim());

beforeEach(() => {
	vi.mocked(api.getIndexingStatus).mockResolvedValue({ success: true, data: status() });
	vi.mocked(api.setIndexingSettings).mockResolvedValue(SAVED as never);
	vi.mocked(api.pauseIndexing).mockResolvedValue(SAVED as never);
	vi.mocked(api.resumeIndexing).mockResolvedValue(SAVED as never);
	vi.mocked(api.queueIndexingDigests).mockResolvedValue(SAVED as never);
});

afterEach(() => {
	seed(null);
	for (const toast of get(toasts)) toasts.remove(toast.id);
	vi.clearAllMocks();
});

describe('IndexingSettingsSection', () => {
	it('renders nothing without a status', () => {
		seed(null);
		mountSection();
		expect(document.body.textContent).not.toContain('Model indexing');
	});

	it('shows the running state with progress, speed, time left, workers and the file being read', () => {
		seed(status(RUNNING));
		mountSection();

		expect(text()).toContain('Model indexing');
		expect(text()).toContain('Indexing in the background');
		expect(text()).toContain('Models are listed and ready to use while this runs.');
		expect(text()).toContain('Indexing');
		expect(text()).toContain('218 of 487 indexed');
		expect(text()).toContain('96 MB/s');
		expect(text()).toContain('about 5 h 12 m');
		expect(text()).toContain('Workers');
		expect(text()).toContain('Reading');
		expect(text()).toContain('neon_wan_lora_v6-190.safetensors');
		expect(text()).toContain('3 new files are being compared with files you already have.');
		expect(button('Pause')).toBeTruthy();
		expect(bar()?.getAttribute('aria-valuenow')).toBe('2');
		expect(bar()?.firstElementChild?.className).toContain('bg-signal-solid');
	});

	it('shows Calculating for the time left until the backend has an estimate', () => {
		seed(status({ ...RUNNING, eta_seconds: null }));
		mountSection();
		expect(text()).toContain('Calculating');
	});

	it('mentions the other files being read when several run at once', () => {
		const current = [
			{ model_id: 'a', filename: 'a.safetensors', size: 10 * MIB, done: MIB },
			{ model_id: 'b', filename: 'b.safetensors', size: 10 * MIB, done: MIB },
			{ model_id: 'c', filename: 'c.safetensors', size: 10 * MIB, done: MIB }
		];
		seed(status({ ...RUNNING, current }));
		mountSection();
		expect(text()).toContain('a.safetensors');
		expect(text()).toContain('and 2 more');
	});

	it('shows the generation pause with a dimmed bar, Waiting copy and the info note', () => {
		seed(status({ ...RUNNING, pause_reason: 'generation', throughput_bps: null, eta_seconds: null, busy: true, current: [] }));
		mountSection();

		expect(text()).toContain('Waiting for generation');
		expect(text()).toContain('Indexing is waiting for your generation to finish');
		expect(text()).toContain('Waiting');
		expect(text()).toContain('0 MB/s');
		expect(alerts()).toContain('Paused while a generation is running. Indexing carries on by itself when it finishes.');
		expect(bar()?.firstElementChild?.className).toContain('bg-fg-subtle');
		expect(text()).not.toContain('Reading');
		expect(button('Pause')).toBeTruthy();
	});

	it('shows a user pause ahead of a generation pause and offers Resume', () => {
		seed(status({ ...RUNNING, paused: true, pause_reason: 'generation', throughput_bps: null, eta_seconds: null }));
		mountSection();

		expect(text()).toContain('Indexing is paused');
		expect(text()).toContain('Paused');
		expect(alerts()).toContain('Paused. Progress is kept, even if you restart PotionUI.');
		expect(button('Resume')).toBeTruthy();
		expect(button('Pause')).toBeUndefined();
	});

	it('warns when Max would not pause for a generation that is running', () => {
		const levelMax = { ...RUNNING, level: 'max' as const, busy: true, workers: 4 };
		seed(status(levelMax));
		mountSection();
		expect(alerts().some((line) => line?.startsWith('A generation is running and Max does not pause for it'))).toBe(true);
		expect(text()).toContain('as fast as your disk allows');
	});

	it('does not warn about Max while no generation is running', () => {
		seed(status({ ...RUNNING, level: 'max' as const, busy: false }));
		mountSection();
		expect(text()).not.toContain('Max does not pause for it');
	});

	it('shows the later state with the not-indexed title, both actions and the explanation', () => {
		seed(status({ policy: 'on_demand', ready: 15, pending_total: 472, deferred: 472, pending_bytes: 1.75 * 1024 * GIB }));
		mountSection();

		expect(text()).toContain('Not started');
		expect(text()).toContain('472 files (1.75 TB) are not indexed yet');
		expect(button('Pick models')).toBeTruthy();
		expect(button('Index all in the background')).toBeTruthy();
		expect(text()).toContain('Indexing lets PotionUI find duplicates');
		expect(text()).toContain('about 5 h 19 m for the rest');
		expect(bar()).toBeNull();
		expect(button('Pause')).toBeUndefined();
	});

	it('shows the awaiting-choice state', () => {
		seed(status(AWAITING));
		mountSection();

		expect(text()).toContain('Waiting for your choice');
		expect(text()).toContain('Choose how to index your models');
		expect(button('Pick models')).toBeTruthy();
	});

	it('shows the up-to-date state with a full bar and nothing left to index', () => {
		seed(status({ ready: 487, ready_bytes: 10 * GIB }));
		mountSection();

		expect(text()).toContain('Up to date');
		expect(text()).toContain('Every model is indexed');
		expect(text()).toContain('487 of 487 indexed');
		expect(bar()?.getAttribute('aria-valuenow')).toBe('100');
		expect(text()).toContain('Nothing left to index.');
		expect(button('Pause')).toBeUndefined();
	});

	it('shows an empty library without progress', () => {
		seed(status());
		mountSection();

		expect(text()).toContain('No models to index yet');
		expect(bar()).toBeNull();
		expect(text()).toContain('Resource use');
	});

	it('lists files that could not be read', () => {
		seed(status({ ...RUNNING, failed: [{ path: '/models/broken.safetensors', error: 'Permission denied' }] }));
		mountSection();

		const toggle = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
			b.textContent?.includes('file could not be read')
		);
		expect(toggle).toBeTruthy();
		toggle!.click();
		return settle().then(() => {
			expect(text()).toContain('/models/broken.safetensors');
			expect(text()).toContain('Permission denied');
		});
	});

	it('keeps the scan content above the digest block while the scan runs', () => {
		seed(status(RUNNING, { state: 'scanning', scanned_roots: ['/mnt/models'] }));
		mountSection();
		expect(text()).toContain('Scanning /mnt/models');
		expect(text()).toContain('Indexing in the background');
	});

	it('does not repeat an up-to-date line next to the not-indexed title', () => {
		seed(status({ ready: 15, pending_total: 472, deferred: 472, pending_bytes: GIB }, { found_on_disk: 487, indexed: 0 }));
		mountSection();
		expect(text()).not.toContain('model files are up to date');
	});

	describe('resource use', () => {
		it('describes the selected level from the backend profile', () => {
			seed(status({ ...RUNNING, level: 'normal' as const }));
			mountSection();
			expect(text()).toContain('Resource use');
			expect(text()).toContain('Applies right away');
			expect(text()).toContain('Two workers at a gentler-than-usual priority, up to 384 MB/s. Pauses while you generate.');
			const selected = Array.from(document.querySelectorAll('button[aria-pressed="true"]')).map((b) => b.textContent?.trim());
			expect(selected).toEqual(['Normal']);
		});

		it('saves a new level, toasts and refetches', async () => {
			seed(status(RUNNING));
			mountSection();

			button('Max')!.click();
			await settle();

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ level: 'max' });
			expect(api.getIndexingStatus).toHaveBeenCalled();
			expect(get(toasts).some((toast) => toast.message === 'Resource use set to Max')).toBe(true);
		});

		it('reports a failed save', async () => {
			seed(status(RUNNING));
			vi.mocked(api.setIndexingSettings).mockResolvedValue({ success: false, message: 'Nope' } as never);
			mountSection();

			button('Normal')!.click();
			await settle();

			expect(get(toasts).some((toast) => toast.type === 'error' && toast.message === 'Nope')).toBe(true);
		});

		it('does nothing when the current level is clicked', async () => {
			seed(status(RUNNING));
			mountSection();
			button('Low')!.click();
			await settle();
			expect(api.setIndexingSettings).not.toHaveBeenCalled();
		});
	});

	describe('actions', () => {
		it('Pause and Resume call their routes', async () => {
			seed(status(RUNNING));
			mountSection();
			button('Pause')!.click();
			await settle();
			expect(api.pauseIndexing).toHaveBeenCalledTimes(1);

			seed(status({ ...RUNNING, paused: true }));
			await settle();
			button('Resume')!.click();
			await settle();
			expect(api.resumeIndexing).toHaveBeenCalledTimes(1);
		});

		it('Index all in the background sets the policy then queues everything', async () => {
			seed(status({ policy: 'on_demand', ready: 15, pending_total: 472, deferred: 472, pending_bytes: GIB }));
			mountSection();

			button('Index all in the background')!.click();
			await settle();

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ policy: 'background' });
			expect(api.queueIndexingDigests).toHaveBeenCalledWith({ mode: 'all' });
			expect(get(toasts).some((toast) => toast.message === 'Indexing started in the background')).toBe(true);
		});

		it('Pick models asks the layout modal to open at the pick step', () => {
			seed(status({ policy: 'on_demand', ready: 15, pending_total: 472, deferred: 472, pending_bytes: GIB }));
			mountSection();
			button('Pick models')!.click();
			expect(indexingStatusStore.pickRequested).toBe(true);
		});
	});
});
