// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { get, type Writable } from 'svelte/store';
import type { PresetInfo } from '$lib/types/api';
import { settle } from './helpers/settle';
import { presetInfo } from './presetFixtures';

type PageStore = Writable<{ url: URL }>;

vi.mock('$lib/services/api/index', async () => {
	const actual = await vi.importActual<typeof import('$lib/services/api/index')>(
		'$lib/services/api/index'
	);
	return {
		...actual,
		api: {
			...actual.api,
			listPresets: vi.fn(),
			getPreset: vi.fn(),
			getPresetConfiguration: vi.fn(),
			getPresetRequirements: vi.fn()
		}
	};
});
vi.mock('$lib/services/admin-api', () => ({
	installPreset: vi.fn(),
	uninstallPreset: vi.fn()
}));

const api = await import('$lib/services/api/index');
const page = (await import('$app/stores')).page as unknown as PageStore;
const { default: PresetsTab } = await import('../../src/routes/admin/components/PresetsTab.svelte');
const { createClassComponent } = await import('svelte/legacy');

const EMPTY_REQUIREMENTS = {
	success: true,
	data: {
		results: [],
		summary: { ok: 0, missing: 0, unknown: 0, optional_missing: 0 },
		checked_at: 1_735_000_000,
		backends: []
	}
};

function preset(overrides: Partial<PresetInfo> = {}): PresetInfo {
	return presetInfo({ id: 'preset-a', name: 'Preset A', ...overrides });
}

function setUrl(search: string) {
	page.update((current) => ({ ...current, url: new URL(`http://localhost/admin${search}`) }));
}

function currentSearch(): URLSearchParams {
	return get(page).url.searchParams;
}

function mount() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: PresetsTab as never, target, props: {} });
	const nav = () => target.querySelector('nav[aria-label="Preset details"]');
	return {
		target,
		activeTab: () => nav()?.querySelector('[aria-current="page"]')?.textContent?.trim() ?? null,
		tab: (label: string) =>
			Array.from(nav()?.querySelectorAll<HTMLButtonElement>('button') ?? []).find((button) =>
				button.textContent?.includes(label)
			),
		back: () => target.querySelector<HTMLButtonElement>('button[data-detail-back]'),
		card: (name: string) => target.querySelector<HTMLElement>(`[data-preset-card][aria-label="${name}"]`),
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

function serve(presets: PresetInfo[], configurationEntries: unknown[] = []) {
	vi.mocked(api.api.listPresets).mockResolvedValue({ success: true, data: presets });
	vi.mocked(api.api.getPreset).mockImplementation(
		async (id: string) => ({ success: true, data: presets.find((p) => p.id === id) }) as never
	);
	vi.mocked(api.api.getPresetConfiguration).mockResolvedValue({
		success: true,
		data: { entries: configurationEntries }
	} as never);
	vi.mocked(api.api.getPresetRequirements).mockResolvedValue(EMPTY_REQUIREMENTS as never);
}

let mounted: ReturnType<typeof mount> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	vi.clearAllMocks();
	setUrl('');
});

describe('PresetsTab detail tab deep link', () => {
	it('opens on the Requirements tab named in the URL', async () => {
		serve([preset()]);
		setUrl('?tab=presets&id=preset-a&view=requirements');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Requirements');
		expect(vi.mocked(api.api.getPresetRequirements).mock.calls[0][0]).toBe('preset-a');
	});

	it('writes the selected tab to the URL and drops it again on Overview', async () => {
		serve([preset()]);
		setUrl('?tab=presets&id=preset-a');
		mounted = mount();
		await settle();
		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);

		mounted.tab('Requirements')?.click();
		await settle();
		expect(mounted.activeTab()).toBe('Requirements');
		expect(currentSearch().get('view')).toBe('requirements');
		expect(currentSearch().get('id')).toBe('preset-a');
		expect(currentSearch().get('tab')).toBe('presets');

		mounted.tab('Overview')?.click();
		await settle();
		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);
		expect(currentSearch().get('id')).toBe('preset-a');
	});

	it('starts another preset on Overview instead of carrying the tab over', async () => {
		serve([preset(), preset({ id: 'preset-b', name: 'Preset B' })]);
		setUrl('?tab=presets&id=preset-a&view=requirements');
		mounted = mount();
		await settle();
		expect(mounted.activeTab()).toBe('Requirements');

		setUrl('?tab=presets&id=preset-b');
		await settle();
		expect(mounted.activeTab()).toBe('Overview');
	});

	it('opens another preset from the grid on Overview with no tab in the URL', async () => {
		serve([preset(), preset({ id: 'preset-b', name: 'Preset B' })]);
		setUrl('?tab=presets&id=preset-a&view=requirements');
		mounted = mount();
		await settle();
		expect(mounted.activeTab()).toBe('Requirements');

		mounted.back()?.click();
		await settle();
		expect(currentSearch().has('id')).toBe(false);
		expect(currentSearch().has('view')).toBe(false);

		mounted.card('Preset B')?.click();
		await settle();
		expect(currentSearch().get('id')).toBe('preset-b');
		expect(currentSearch().has('view')).toBe(false);
		expect(mounted.activeTab()).toBe('Overview');
	});

	it('clears a tab the preset does not offer from the URL', async () => {
		serve([preset({ installed: false })]);
		setUrl('?tab=presets&id=preset-a&view=form');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);
		expect(currentSearch().get('id')).toBe('preset-a');
	});

	it('falls back to Overview and clears the URL when a requested Configuration tab has no entries', async () => {
		serve([preset({ installed: true })], []);
		setUrl('?tab=presets&id=preset-a&view=configuration');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);
	});

	it('keeps the Configuration tab when the preset has entries', async () => {
		serve([preset({ installed: true })], [{ key: 'tags', label: 'Tags', type: 'model_tags', value: [] }]);
		setUrl('?tab=presets&id=preset-a&view=configuration');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Configuration');
		expect(currentSearch().get('view')).toBe('configuration');
		expect(api.api.getPresetConfiguration).toHaveBeenCalledTimes(1);
	});

	it('falls back to Overview and clears the URL when the configuration fetch throws', async () => {
		serve([preset({ installed: true })]);
		vi.mocked(api.api.getPresetConfiguration).mockRejectedValue(new Error('unreachable'));
		setUrl('?tab=presets&id=preset-a&view=configuration');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);
		expect(api.api.getPresetConfiguration).toHaveBeenCalledTimes(1);
	});

	it('falls back to Overview and clears the URL when the configuration fetch is refused', async () => {
		serve([preset({ installed: true })]);
		vi.mocked(api.api.getPresetConfiguration).mockResolvedValue({ success: false, message: 'denied' } as never);
		setUrl('?tab=presets&id=preset-a&view=configuration');
		mounted = mount();
		await settle();

		expect(mounted.activeTab()).toBe('Overview');
		expect(currentSearch().has('view')).toBe(false);
		expect(api.api.getPresetConfiguration).toHaveBeenCalledTimes(1);
	});
});
