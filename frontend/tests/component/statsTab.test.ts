// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { flushSync } from 'svelte';

const api = {
	getStatsOverview: vi.fn(),
	getStatsTimeseries: vi.fn(),
	getStatsDurations: vi.fn(),
	getStatsStorage: vi.fn(),
	getStatsBreakdown: vi.fn(),
	getStatsPresetTiming: vi.fn(),
	getStatsPresetResources: vi.fn(),
	getStatsSpend: vi.fn()
};

vi.mock('$lib/services/api/index', () => ({ api }));

const { default: StatsTab } = await import('../../src/routes/admin/components/StatsTab.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { statsRowLimits } = await import('$lib/stores/statsRowLimits');

const ok = (data: unknown) => ({ success: true, data });

function flush(times = 12) {
	let p: Promise<void> = Promise.resolve();
	for (let i = 0; i < times; i++) p = p.then(() => Promise.resolve());
	return p;
}

let target: HTMLDivElement;
let component: ReturnType<typeof createClassComponent> | undefined;

async function mountTab() {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: StatsTab as never, target, props: {} });
	await flush();
	flushSync();
}

function cardTitles(): string[] {
	return Array.from(target.querySelectorAll('h3')).map((h) => h.textContent?.trim() ?? '');
}

beforeEach(() => {
	vi.clearAllMocks();
	for (const section of ['presets', 'models', 'samplers', 'denoises'] as const) statsRowLimits.setLimit(section, 8);
	api.getStatsOverview.mockResolvedValue({ success: false, message: 'x' });
	api.getStatsTimeseries.mockResolvedValue(ok(null));
	api.getStatsDurations.mockResolvedValue(ok(null));
	api.getStatsStorage.mockResolvedValue(ok(null));
	api.getStatsPresetTiming.mockResolvedValue(ok({ items: [] }));
	api.getStatsPresetResources.mockResolvedValue(ok({ items: [] }));
	api.getStatsSpend.mockResolvedValue(ok(null));
	api.getStatsBreakdown.mockImplementation(async (dimension: string) =>
		ok({ dimension, total_distinct: 1, items: [{ label: `${dimension}-a`, count: 3, group: 'sdxl' }] })
	);
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
});

describe('StatsTab breakdown cards', () => {
	it('fetches and renders all eight breakdowns in layout order', async () => {
		await mountTab();
		expect(api.getStatsBreakdown.mock.calls.map((c) => c[0]).sort()).toEqual(
			['cfg', 'denoise', 'model', 'preset', 'resolution', 'sampler', 'scheduler', 'steps']
		);
		const titles = cardTitles();
		expect(titles).toEqual(
			expect.arrayContaining(['Top presets', 'Top models', 'Top resolutions', 'Samplers', 'Schedulers', 'Steps', 'CFG scale', 'Denoise'])
		);
		expect(titles.indexOf('Top presets')).toBeLessThan(titles.indexOf('Top models'));
		expect(titles.indexOf('Top resolutions')).toBeLessThan(titles.indexOf('Samplers'));
	});

	it('passes the stored row limit of each section to its fetch', async () => {
		statsRowLimits.setLimit('samplers', 20);
		await mountTab();
		const call = api.getStatsBreakdown.mock.calls.find((c) => c[0] === 'sampler');
		expect(call?.[1]).toBe(20);
	});

	it('refetches only the changed section when its row limit changes', async () => {
		await mountTab();
		api.getStatsBreakdown.mockClear();
		const select = target.querySelectorAll<HTMLSelectElement>('select[aria-label="Rows to show"]')[1];
		select.value = '50';
		select.dispatchEvent(new Event('change', { bubbles: true }));
		await flush();
		flushSync();
		expect(api.getStatsBreakdown).toHaveBeenCalledTimes(1);
		expect(api.getStatsBreakdown.mock.calls[0][0]).toBe('model');
		expect(api.getStatsBreakdown.mock.calls[0][1]).toBe(50);
	});

	it('hides a card whose fetch failed and keeps the others', async () => {
		api.getStatsBreakdown.mockImplementation(async (dimension: string) =>
			dimension === 'cfg' ? { success: false, message: 'no' } : ok({ dimension, total_distinct: 1, items: [{ label: 'a', count: 1 }] })
		);
		await mountTab();
		const titles = cardTitles();
		expect(titles).not.toContain('CFG scale');
		expect(titles).toContain('Steps');
	});

	it('shows the Type column only on the models table', async () => {
		await mountTab();
		const toggles = Array.from(target.querySelectorAll<HTMLButtonElement>('button[aria-label="Show table view"]'));
		for (const t of toggles) t.click();
		flushSync();
		const headers = Array.from(target.querySelectorAll('thead')).map((h) => h.textContent?.replace(/\s+/g, ' ').trim() ?? '');
		expect(headers.filter((h) => h.includes('Type'))).toHaveLength(1);
		expect(headers.find((h) => h.includes('Type'))).toContain('Model');
	});
});
