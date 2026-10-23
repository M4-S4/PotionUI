// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { writable } from 'svelte/store';
import type { ActiveGrid, GridCell } from '$lib/generation/compare/compareStore.svelte';

vi.mock('$lib/generation/compare/view/gridApi', () => ({
	openGridId: writable<string | null>(null),
	fetchGrid: vi.fn(),
	retryGridFailed: vi.fn()
}));

const gridApi = await import('$lib/generation/compare/view/gridApi');
const { default: HistoryGridView } = await import('../../src/routes/history/components/HistoryGridView.svelte');
const { createClassComponent } = await import('svelte/legacy');

function cell(status: GridCell['status'], x: number): GridCell {
	return {
		x,
		y: 0,
		generationId: null,
		status,
		axisValues: {},
		seed: null,
		thumbnailUrl: null,
		mediaType: null,
		error: null,
		elapsedSeconds: null,
		progress: null,
		previewUrl: null
	};
}

function grid(statuses: GridCell['status'][]): ActiveGrid {
	return {
		id: 'grid-1',
		config: { armed: false, x: null, y: null, lockSeed: true },
		cells: statuses.map((status, index) => cell(status, index)),
		cols: statuses.length,
		rows: 1
	};
}

async function flush() {
	for (let i = 0; i < 6; i++) await vi.advanceTimersByTimeAsync(0);
}

async function mountView() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: HistoryGridView as never,
		target,
		props: { gridId: 'grid-1', onClose: vi.fn() }
	});
	await flush();
	return () => {
		component.$destroy();
		target.remove();
	};
}

afterEach(() => {
	vi.useRealTimers();
	vi.clearAllMocks();
	document.body.innerHTML = '';
});

describe('HistoryGridView polling', () => {
	it('re-fetches the grid every 2.5s while a cell is still queued or running', async () => {
		vi.useFakeTimers();
		vi.mocked(gridApi.fetchGrid).mockResolvedValue(grid(['completed', 'running']));

		const destroy = await mountView();
		expect(gridApi.fetchGrid).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(2500);
		expect(gridApi.fetchGrid).toHaveBeenCalledTimes(2);
		destroy();
	});

	it('stops polling once every cell is settled', async () => {
		vi.useFakeTimers();
		vi.mocked(gridApi.fetchGrid)
			.mockResolvedValueOnce(grid(['completed', 'running']))
			.mockResolvedValue(grid(['completed', 'completed']));

		const destroy = await mountView();
		await vi.advanceTimersByTimeAsync(2500);
		await flush();
		expect(gridApi.fetchGrid).toHaveBeenCalledTimes(2);

		await vi.advanceTimersByTimeAsync(20000);
		expect(gridApi.fetchGrid).toHaveBeenCalledTimes(2);
		destroy();
	});

	it('never polls a grid that is already settled', async () => {
		vi.useFakeTimers();
		vi.mocked(gridApi.fetchGrid).mockResolvedValue(grid(['completed', 'failed']));

		const destroy = await mountView();
		await vi.advanceTimersByTimeAsync(20000);

		expect(gridApi.fetchGrid).toHaveBeenCalledTimes(1);
		destroy();
	});
});
