import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { AWAITING, status } from './helpers/indexingFixtures';

vi.mock('$lib/services/api', () => ({
	api: {
		getIndexingStatus: vi.fn(),
		getModelRoots: vi.fn(),
		listModelLayouts: vi.fn()
	}
}));

const { api } = await import('$lib/services/api');
const { indexingStatusStore } = await import('$lib/models-location/indexingStatus.svelte');
const { default: ModelsLocationStep } = await import('../../src/routes/setup/components/ModelsLocationStep.svelte');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const DISMISSED_KEY = 'potionui:setup:modelsLocationDismissed';

const overview = {
	roots: [],
	types: [],
	unplaced: [],
	indexing: { state: 'idle' },
	server_os: 'Linux',
	path_style: 'posix'
};

function seed(next: ReturnType<typeof status>) {
	vi.mocked(api.getIndexingStatus).mockResolvedValue({ success: true, data: next });
	indexingStatusStore.status = next;
}

const text = () => document.body.textContent?.replace(/\s+/g, ' ') ?? '';

beforeEach(() => {
	vi.mocked(api.getModelRoots).mockResolvedValue({ success: true, data: overview } as never);
	vi.mocked(api.listModelLayouts).mockResolvedValue({ layouts: [] } as never);
});

afterEach(() => {
	indexingStatusStore.status = null;
	localStorage.removeItem(DISMISSED_KEY);
	vi.clearAllMocks();
});

describe('setup models step', () => {
	it('tells the admin the models are usable now and the choice comes after setup', async () => {
		seed(status({ ...AWAITING, ready: 15, pending_total: 472 }));
		mountComponent(ModelsLocationStep, {});
		await settle();

		expect(text()).toContain("487 models found. They're usable now; you'll choose how to index them after setup.");
	});

	it('shows the same line once the location step is collapsed', async () => {
		localStorage.setItem(DISMISSED_KEY, '1');
		seed(status({ ...AWAITING, ready: 15, pending_total: 472 }));
		mountComponent(ModelsLocationStep, {});
		await settle();

		expect(text()).toContain('Models location:');
		expect(text()).toContain("487 models found. They're usable now; you'll choose how to index them after setup.");
	});

	it('no longer says indexing has to carry on while a scan runs', async () => {
		localStorage.setItem(DISMISSED_KEY, '1');
		seed(status({ ...AWAITING, ready: 15, pending_total: 472 }, { state: 'scanning' }));
		mountComponent(ModelsLocationStep, {});
		await settle();

		expect(text()).not.toContain('Model indexing continues in the background');
	});

	it('shows no line while there are no models', async () => {
		seed(status());
		mountComponent(ModelsLocationStep, {});
		await settle();

		expect(text()).not.toContain('models found');
		expect(text()).not.toContain('usable now');
	});

	it('does not promise a choice once it has been answered', async () => {
		seed(status({ ready: 487, ready_bytes: 1 }));
		mountComponent(ModelsLocationStep, {});
		await settle();

		expect(text()).toContain("487 models found. They're usable now.");
		expect(text()).not.toContain("you'll choose");
	});
});
