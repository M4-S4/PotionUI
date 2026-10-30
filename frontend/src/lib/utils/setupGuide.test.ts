import { describe, it, expect } from 'vitest';
import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';
import {
	canNavigateToStep,
	deriveCurrentStep,
	diskStatus,
	machineVerdict,
	orderStarters,
	recommendStarter,
	reviewDownloadLabel,
	runPhase,
	showsSkipGuide,
	skipToastMessage,
	starterFit,
	starterNeedText,
	stepStatus,
	type SetupStepId
} from './setupGuide';

const GIB = 1024 ** 3;

function recipe(overrides: Partial<SetupRecipe>): SetupRecipe {
	return {
		id: 'r',
		name: 'R',
		summary: '',
		description: '',
		engine: 'native',
		category: 'image',
		artifact_count: 1,
		total_download_bytes: 10 * GIB,
		presets: [],
		last_completed_at: null,
		blurb: null,
		min_vram_gb: null,
		recommended_vram_gb: null,
		sources: ['huggingface.co'],
		first_run: true,
		...overrides
	};
}

const ZIMAGE = recipe({
	id: 'zimage-starter',
	min_vram_gb: 12,
	recommended_vram_gb: 16,
	total_download_bytes: 19.27 * GIB
});
const SDXL = recipe({ id: 'sdxl-starter', min_vram_gb: 8, total_download_bytes: 6.62 * GIB });
const QWEN = recipe({
	id: 'qwen-image-starter',
	recommended_vram_gb: 16,
	total_download_bytes: 28 * GIB
});
const KLEIN = recipe({ id: 'flux2-klein', min_vram_gb: 8, recommended_vram_gb: 24, first_run: false });
const RECIPES = [QWEN, SDXL, KLEIN, ZIMAGE];

function machine(vram: number | null, overrides: Partial<SetupMachine> = {}): SetupMachine {
	return {
		gpu:
			vram === null
				? null
				: {
						name: 'NVIDIA GeForce',
						vram_bytes: vram * GIB,
						compute_capability: '8.9',
						supported: true
					},
		ram_total_bytes: 32 * GIB,
		models_disk: { path: '/models', free_bytes: 100 * GIB, total_bytes: 500 * GIB },
		...overrides
	};
}

describe('recommendStarter', () => {
	it.each([
		[32, 'zimage-starter'],
		[31.4, 'zimage-starter'],
		[24, 'zimage-starter'],
		[16, 'zimage-starter'],
		[15.6, 'zimage-starter'],
		[12, 'sdxl-starter'],
		[8, 'sdxl-starter'],
		[6, null]
	])('recommends for %s GB', (vram, expected) => {
		expect(recommendStarter(RECIPES, machine(vram))).toBe(expected);
	});

	it('recommends nothing without a GPU', () => {
		expect(recommendStarter(RECIPES, machine(null))).toBeNull();
	});

	it('recommends nothing when detection failed or the GPU is unsupported', () => {
		expect(recommendStarter(RECIPES, null)).toBeNull();
		const unsupported = machine(24);
		unsupported.gpu!.supported = false;
		expect(recommendStarter(RECIPES, unsupported)).toBeNull();
	});

	it('ignores recipes that are not first-run starters', () => {
		expect(recommendStarter([KLEIN], machine(32))).toBeNull();
	});

	it('prefers a recommended figure on a threshold tie, then the smaller download', () => {
		const a = recipe({ id: 'a', min_vram_gb: 16 });
		const b = recipe({ id: 'b', recommended_vram_gb: 16 });
		expect(recommendStarter([a, b], machine(16))).toBe('b');
		expect(recommendStarter([QWEN, ZIMAGE], machine(16))).toBe('zimage-starter');
	});
});

describe('orderStarters', () => {
	it('puts the recommendation first then smaller downloads, dropping non-first-run', () => {
		expect(orderStarters(RECIPES, 'zimage-starter').map((r) => r.id)).toEqual([
			'zimage-starter',
			'sdxl-starter',
			'qwen-image-starter'
		]);
	});

	it('orders by size when nothing is recommended', () => {
		expect(orderStarters(RECIPES, null).map((r) => r.id)).toEqual([
			'sdxl-starter',
			'zimage-starter',
			'qwen-image-starter'
		]);
	});
});

describe('starterFit', () => {
	it('is good when the card meets the figure', () => {
		expect(starterFit(ZIMAGE, machine(24))).toEqual({ tone: 'good', text: 'Fits your card' });
	});

	it('says wants for a recommended figure and needs for a minimum', () => {
		expect(starterFit(ZIMAGE, machine(8)).text).toBe('Wants 16 GB, you have 8 GB');
		expect(starterFit(SDXL, machine(6)).text).toBe('Needs 8 GB, you have 6 GB');
	});

	it('is neutral without a declared figure or a readable machine', () => {
		expect(starterFit(recipe({}), machine(24)).tone).toBe('dim');
		expect(starterFit(ZIMAGE, null).tone).toBe('dim');
	});

	it('flags a missing GPU', () => {
		expect(starterFit(ZIMAGE, machine(null))).toEqual({ tone: 'low', text: 'Needs an NVIDIA GPU' });
	});
});

describe('starterNeedText', () => {
	it('describes the declared figures', () => {
		expect(starterNeedText(ZIMAGE)).toBe('12 to 16 GB suggested');
		expect(starterNeedText(SDXL)).toBe('8 GB minimum');
		expect(starterNeedText(QWEN)).toBe('16 GB recommended');
		expect(starterNeedText(recipe({}))).toBe('Not yet measured');
	});
});

describe('reviewDownloadLabel', () => {
	it('states the size when known', () => {
		expect(reviewDownloadLabel(SDXL)).toBe('Review 6.62 GB download');
		expect(reviewDownloadLabel(null)).toBe('Review download');
	});
});

describe('machineVerdict', () => {
	it('covers every machine state', () => {
		expect(machineVerdict(null).kind).toBe('unknown');
		expect(machineVerdict(machine(null)).kind).toBe('no-gpu');
		expect(machineVerdict(machine(6)).kind).toBe('low-memory');
		expect(machineVerdict(machine(8)).kind).toBe('sdxl');
		expect(machineVerdict(machine(24)).kind).toBe('good');
	});
});

describe('diskStatus', () => {
	it('compares free space with the largest starter plus headroom', () => {
		expect(diskStatus(100 * GIB, RECIPES).tone).toBe('success');
		expect(diskStatus(30 * GIB, RECIPES).tone).toBe('warning');
	});
});

describe('deriveCurrentStep', () => {
	const base = {
		ownerExists: true,
		welcomeSeen: true,
		computerSeen: false,
		modelsDone: false,
		run: null,
		instanceReady: false
	};

	it('walks welcome and account before an owner exists', () => {
		expect(deriveCurrentStep({ ...base, ownerExists: false, welcomeSeen: false })).toBe('welcome');
		expect(deriveCurrentStep({ ...base, ownerExists: false })).toBe('account');
	});

	it('moves through computer, models and starter from the persisted flags', () => {
		expect(deriveCurrentStep(base)).toBe('computer');
		expect(deriveCurrentStep({ ...base, computerSeen: true })).toBe('models');
		expect(deriveCurrentStep({ ...base, computerSeen: true, modelsDone: true })).toBe('starter');
	});

	it('follows the run: install while active, consent or failed, ready when completed', () => {
		for (const status of ['pending', 'running', 'paused', 'awaiting_consent', 'failed'] as const) {
			expect(deriveCurrentStep({ ...base, run: { status } })).toBe('install');
		}
		expect(deriveCurrentStep({ ...base, run: { status: 'completed' } })).toBe('ready');
	});

	it('treats a cancelled run as no run', () => {
		expect(
			deriveCurrentStep({ ...base, computerSeen: true, modelsDone: true, run: { status: 'cancelled' } })
		).toBe('starter');
	});

	it('lands on ready when the instance is already ready', () => {
		expect(deriveCurrentStep({ ...base, instanceReady: true })).toBe('ready');
	});
});

describe('runPhase and step status', () => {
	it('maps run statuses', () => {
		expect(runPhase(null)).toBe('none');
		expect(runPhase({ status: 'awaiting_consent' })).toBe('consent');
	});

	it('labels the install step by run state', () => {
		expect(stepStatus('install', { status: 'awaiting_consent' })?.text).toBe('Needs your go-ahead');
		expect(stepStatus('install', { status: 'running' })?.text).toBe('Running');
		expect(stepStatus('ready')?.tone).toBe('success');
		expect(stepStatus('welcome')).toBeNull();
	});
});

describe('canNavigateToStep and skip visibility', () => {
	it('allows only earlier steps and never the locked ones once an owner exists', () => {
		expect(canNavigateToStep('models', 'starter', true)).toBe(true);
		expect(canNavigateToStep('starter', 'starter', true)).toBe(false);
		expect(canNavigateToStep('computer', 'starter', true)).toBe(true);
		expect(canNavigateToStep('account', 'starter', true)).toBe(false);
		expect(canNavigateToStep('welcome', 'account', false)).toBe(true);
	});

	it('shows skip on steps 3 to 6 only', () => {
		const ids: SetupStepId[] = ['welcome', 'account', 'computer', 'models', 'starter', 'install', 'ready'];
		expect(ids.map(showsSkipGuide)).toEqual([false, false, true, true, true, true, false]);
	});
});

describe('skipToastMessage', () => {
	it('uses the default message away from the install step', () => {
		expect(skipToastMessage({ mobile: false, step: 'starter', runStatus: null })).toBe(
			'Guide skipped. Pick it up again any time from Resume setup.'
		);
	});

	it('points mobile users at Settings', () => {
		expect(skipToastMessage({ mobile: true, step: 'computer', runStatus: null })).toBe(
			'Guide skipped. Pick it up again any time from Settings, under Resume setup.'
		);
	});

	it('says nothing downloads while parked at consent', () => {
		expect(skipToastMessage({ mobile: false, step: 'install', runStatus: 'awaiting_consent' })).toBe(
			'Guide skipped. Nothing downloads until you approve it in Resume setup.'
		);
	});

	it('says the install keeps running while a run is active', () => {
		expect(skipToastMessage({ mobile: true, step: 'install', runStatus: 'running' })).toBe(
			'Guide skipped. Your install keeps running, and Settings, under Resume setup shows its progress.'
		);
	});

	it('falls back to the default for a failed run', () => {
		expect(skipToastMessage({ mobile: false, step: 'install', runStatus: 'failed' })).toContain(
			'Pick it up again'
		);
	});
});
