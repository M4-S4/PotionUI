import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { mount, unmount } from 'svelte';
import { writable } from 'svelte/store';
import { settle } from './helpers/settle';

const current = vi.hoisted(() => ({
	model: null as unknown,
	store: null as null | { update: (fn: (value: any) => any) => void },
	refresh: null as null | ReturnType<typeof import('vitest').vi.fn>
}));

const requestModelDigest = vi.hoisted(() => vi.fn());

vi.mock('$lib/services/api/index', () => ({
	api: new Proxy({ requestModelDigest }, { get: (target, key) => (target as any)[key] ?? vi.fn().mockResolvedValue({ success: true, data: {} }) })
}));
vi.mock('$lib/services/admin-api', () => ({
	getCloudModelScope: vi.fn(),
	setCloudModelScope: vi.fn()
}));

vi.mock('$lib/components/modals/model-details/modelDetailsController', () => ({
	createAdminModelDetailsController: () => {
		const model = writable(current.model);
		current.store = model;
		return {
			model,
			loading: writable(false),
			currentImageIndex: writable(0),
			imageFiles: writable([]),
			displayName: writable('Model'),
			selectedTags: writable([]),
			selectedTagIds: writable([]),
			availability: writable(null),
			availabilityLoading: writable(false),
			load: vi.fn(),
			refresh: (current.refresh = vi.fn(async () => {
				model.update((value: any) => ({ ...value, sha256: 'f'.repeat(64) }));
			})),
			saveType: vi.fn(),
			saveDescription: vi.fn().mockResolvedValue(undefined),
			savePromptingGuidance: vi.fn().mockResolvedValue(undefined),
			updateTags: vi.fn(),
			prevImage: vi.fn(),
			nextImage: vi.fn(),
			handlePrimaryPreviewChange: vi.fn()
		};
	}
}));

const { default: BlankStub } = await import('./stubs/BlankStub.svelte');
const stub = () => ({ default: BlankStub });
for (const name of [
	'ModelMediaViewer',
	'ModelPreviewGallery',
	'ModelGenerationsCard',
	'ModelAttributesCard',
	'ModelInfoCard',
	'ModelFilesCard',
	'ModelTechnicalDetailsCard',
	'ModelAvailabilityCard',
	'ModelMirrorsCard'
]) {
	vi.doMock(`$lib/components/modals/model-details/${name}.svelte`, stub);
}
vi.doMock('$lib/components/recipes/ModelOtherVariants.svelte', stub);
vi.doMock('$lib/components/assignment/AssignmentCard.svelte', stub);

const { default: ModelDetailPage } = await import('../../src/routes/admin/components/models/ModelDetailPage.svelte');

function model(overrides: Record<string, unknown> = {}) {
	return {
		id: 'm1',
		filename: 'mystery.safetensors',
		model_type: 'undefined',
		sha256: null,
		is_directory: false,
		type_info: null,
		providers: [],
		files: [],
		description: '',
		prompting_guidance: '',
		model_metadata: {},
		...overrides
	};
}

let instance: object | undefined;
let target: HTMLElement | undefined;

const onMerged = vi.fn();

async function mountPage(m: unknown) {
	current.model = m;
	target = document.createElement('div');
	document.body.appendChild(target);
	instance = mount(ModelDetailPage, {
		target,
		props: { modelId: 'm1', onBack: () => {}, onDeleted: () => {}, onAssignChanged: () => {}, onMerged }
	});
	await settle();
	return target;
}

const select = () => document.querySelector<HTMLSelectElement>('#model-type')!;
const tryAgain = () => Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim() === 'Try again');
const text = () => document.body.textContent?.replace(/\s+/g, ' ') ?? '';

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	if (instance) unmount(instance as never);
	target?.remove();
	instance = undefined;
	document.body.innerHTML = '';
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe('admin model detail for a file that is still being indexed', () => {
	it('asks for the file to be read, says so with the estimate and keeps the type locked', async () => {
		requestModelDigest.mockResolvedValue({ success: true, data: { state: 'queued', eta_seconds: 240 } });
		await mountPage(model());

		expect(requestModelDigest).toHaveBeenCalledWith('m1', { retry: false });
		expect(text()).toContain('Verifying this file… about 4 min left');
		expect(select().disabled).toBe(true);
	});

	it('unlocks the type and reloads the model once the file is ready', async () => {
		requestModelDigest
			.mockResolvedValueOnce({ success: true, data: { state: 'queued', eta_seconds: 5 } })
			.mockResolvedValueOnce({ success: true, data: { state: 'ready', eta_seconds: null } });
		await mountPage(model());
		expect(text()).toContain('Verifying this file…');

		await vi.advanceTimersByTimeAsync(3000);
		await settle();

		expect(current.refresh).toHaveBeenCalledWith('m1');
		expect(text()).not.toContain('Verifying this file…');
		expect(select().disabled).toBe(false);
	});

	it('does not ask when the model already has its content id', async () => {
		await mountPage(model({ sha256: 'a'.repeat(64) }));

		expect(requestModelDigest).not.toHaveBeenCalled();
		expect(text()).not.toContain('Verifying this file…');
		expect(select().disabled).toBe(false);
	});

	it('does not ask for a directory model', async () => {
		await mountPage(model({ is_directory: true }));

		expect(requestModelDigest).not.toHaveBeenCalled();
		expect(select().disabled).toBe(true);
	});

	it('does not ask for a cloud model', async () => {
		await mountPage(model({ model_type: 'cloud', providers: [{ provider: 'cloud.openrouter', tags: [] }] }));

		expect(requestModelDigest).not.toHaveBeenCalled();
		expect(text()).not.toContain('Verifying this file…');
	});

	it('stops waiting and says the file could not be checked when the request fails', async () => {
		requestModelDigest.mockResolvedValue({ success: false });
		await mountPage(model());

		expect(text()).not.toContain('Verifying this file…');
		expect(text()).toContain('This file could not be checked, so its type cannot be set yet.');
		expect(select().disabled).toBe(true);
	});

	it('says so when checking keeps going for too long', async () => {
		requestModelDigest.mockResolvedValue({ success: true, data: { state: 'queued', eta_seconds: 5 } });
		await mountPage(model());

		for (let i = 0; i < 125; i++) await vi.advanceTimersByTimeAsync(3000);
		await settle();

		expect(text()).not.toContain('Verifying this file…');
		expect(text()).toContain('Checking this file is taking longer than expected. Reopen this page to try again.');
		expect(requestModelDigest).toHaveBeenCalledTimes(120);
	});

	it('stops asking when the page is closed', async () => {
		requestModelDigest.mockResolvedValue({ success: true, data: { state: 'queued', eta_seconds: 5 } });
		await mountPage(model());
		expect(requestModelDigest).toHaveBeenCalledTimes(1);

		unmount(instance as never);
		instance = undefined;
		await vi.advanceTimersByTimeAsync(60000);

		expect(requestModelDigest).toHaveBeenCalledTimes(1);
	});

	it('stops at once when every copy of the file failed, shows the message and offers Try again', async () => {
		requestModelDigest.mockResolvedValue({ success: true, data: { state: 'failed', eta_seconds: null } });
		await mountPage(model());

		expect(text()).not.toContain('Verifying this file…');
		expect(text()).toContain('This file could not be checked, so its type cannot be set yet.');
		expect(tryAgain()).toBeTruthy();
		expect(select().disabled).toBe(true);

		await vi.advanceTimersByTimeAsync(120000);
		expect(requestModelDigest).toHaveBeenCalledTimes(1);
	});

	it('does not offer Try again when the request itself failed', async () => {
		requestModelDigest.mockResolvedValue({ success: false });
		await mountPage(model());

		expect(text()).toContain('This file could not be checked');
		expect(tryAgain()).toBeUndefined();
	});

	it('Try again asks with retry, shows the wait again and resumes until the file is ready', async () => {
		requestModelDigest
			.mockResolvedValueOnce({ success: true, data: { state: 'failed', eta_seconds: null } })
			.mockResolvedValueOnce({ success: true, data: { state: 'queued', eta_seconds: 5 } })
			.mockResolvedValueOnce({ success: true, data: { state: 'ready', eta_seconds: null } });
		await mountPage(model());

		tryAgain()!.click();
		await settle();

		expect(requestModelDigest).toHaveBeenNthCalledWith(2, 'm1', { retry: true });
		expect(text()).toContain('Verifying this file…');
		expect(text()).not.toContain('This file could not be checked');
		expect(tryAgain()).toBeUndefined();

		await vi.advanceTimersByTimeAsync(3000);
		await settle();

		expect(requestModelDigest).toHaveBeenNthCalledWith(3, 'm1', { retry: false });
		expect(current.refresh).toHaveBeenCalledWith('m1');
		expect(select().disabled).toBe(false);
	});

	it('goes to the owner instead of reloading when the model was merged into another one', async () => {
		requestModelDigest.mockResolvedValue({ success: true, data: { state: 'ready', eta_seconds: 0, merged_into: 'owner-9' } });
		await mountPage(model());

		expect(onMerged).toHaveBeenCalledWith('owner-9');
		expect(current.refresh).not.toHaveBeenCalled();
		expect(text()).not.toContain('Verifying this file…');
	});
});
