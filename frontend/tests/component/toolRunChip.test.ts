// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { settle } from './helpers/settle';

const startMediaToolRun = vi.fn();
const getMediaToolRun = vi.fn();
const cancelMediaToolRun = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: vi.fn(),
	startMediaToolRun: (...args: unknown[]) => startMediaToolRun(...args),
	getMediaToolRun: (...args: unknown[]) => getMediaToolRun(...args),
	cancelMediaToolRun: (...args: unknown[]) => cancelMediaToolRun(...args),
	previewMediaTool: vi.fn()
}));

const { mediaToolRuns } = await import('$lib/media-tools/runs');
const { page } = await import('./stubs/appStores');
const { default: ToolRunChip } = await import('$lib/components/tools/ToolRunChip.svelte');

let target: HTMLElement;
let instance: ReturnType<typeof mount>;

function mountChip() {
	target = document.createElement('div');
	document.body.appendChild(target);
	instance = mount(ToolRunChip, { target });
	flushSync();
}

function goTo(path: string) {
	page.update((current) => ({ ...current, url: new URL(path, 'http://localhost') }));
	flushSync();
}

async function startRun(keys: string[], runId = 'run-1', progress = 0.4) {
	startMediaToolRun.mockResolvedValueOnce({ run_id: runId });
	getMediaToolRun.mockResolvedValue({
		run_id: runId,
		status: 'running',
		progress,
		label: 'Encoding',
		items: [],
		replaced: false,
		error: null
	});
	await mediaToolRuns.start({
		keys,
		toolId: 'trim',
		label: 'Trim',
		body: { steps: [{ tool: 'trim', params: {} }], inputs: [{ item_id: 'i' }], output: { mode: 'new' } }
	});
	await settle(3);
	flushSync();
}

beforeEach(() => {
	startMediaToolRun.mockReset();
	getMediaToolRun.mockReset();
	cancelMediaToolRun.mockReset();
	cancelMediaToolRun.mockResolvedValue(undefined);
	goTo('/history');
});

afterEach(() => {
	unmount(instance);
	target.remove();
	mediaToolRuns.destroy();
	document.body.innerHTML = '';
});

describe('ToolRunChip', () => {
	it('renders nothing while no run is active', () => {
		mountChip();
		expect(target.querySelector('[data-tool-run-chip]')).toBeNull();
	});

	it('shows one chip with the label and percent for an active run on History', async () => {
		mountChip();
		await startRun(['g:0']);
		const chips = target.querySelectorAll('[data-tool-run-chip]');
		expect(chips).toHaveLength(1);
		expect(target.querySelector('[data-tool-run-chip-label]')!.textContent).toBe('Trim');
		expect(target.querySelector('[data-tool-run-chip-percent]')!.textContent).toBe('40%');
	});

	it('shows one chip for a run that covers several items', async () => {
		mountChip();
		await startRun(['a', 'b', 'c']);
		expect(target.querySelectorAll('[data-tool-run-chip]')).toHaveLength(1);
	});

	it('shows on Library and hides elsewhere', async () => {
		mountChip();
		await startRun(['a']);
		goTo('/library');
		expect(target.querySelector('[data-tool-run-chip]')).not.toBeNull();
		goTo('/generate');
		expect(target.querySelector('[data-tool-run-chip]')).toBeNull();
	});

	it('cancels the run from the chip', async () => {
		mountChip();
		await startRun(['a']);
		target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Trim"]')!.click();
		await settle(2);
		expect(cancelMediaToolRun).toHaveBeenCalledWith('run-1');
	});
});
