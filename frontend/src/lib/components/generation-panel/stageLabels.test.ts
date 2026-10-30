import { describe, it, expect } from 'vitest';
import { FALLBACK_STAGE_LABEL, resolveStageStatus, stageLabel } from './stageLabels';

const RAW = /<<|>>|PIPE:/;

describe('stageLabel', () => {
	it('maps known pipe families to friendly labels', () => {
		expect(stageLabel('generator')).toBe('Generating');
		expect(stageLabel('checkpoint_loader/sdxl')).toBe('Loading model');
		expect(stageLabel('upscaler')).toBe('Upscaling');
		expect(stageLabel('tiled_detailer/sdxl')).toBe('Refining');
		expect(stageLabel('gallery')).toBe('Saving');
	});

	it('falls back to a known title when the pipe name is absent', () => {
		expect(stageLabel(undefined, 'Upscaling')).toBe('Upscaling');
	});

	it('never returns an unknown pipe name or title', () => {
		expect(stageLabel('acme_secret_pipe')).toBe(FALLBACK_STAGE_LABEL);
		expect(stageLabel('acme_secret_pipe', 'Acme secret pipe')).toBe(FALLBACK_STAGE_LABEL);
		expect(stageLabel()).toBe(FALLBACK_STAGE_LABEL);
	});
});

describe('resolveStageStatus', () => {
	const input = {
		currentStep: 'Loading checkpoint <<MODEL:flux-dev.safetensors>> <<RESOLUTION:1024x768>> <<PROGRESS:3/30>>',
		message: '<<PIPE:acme_pipe:bolt>>',
		pipeName: 'checkpoint_loader/flux',
		pipeId: 4
	};

	it('hides raw markers, models and pipe identity from non-admins', () => {
		const status = resolveStageStatus({ ...input, isAdmin: false });
		expect(status.label).toBe('Loading model');
		expect(status.stepCount).toBe('3/30');
		expect(status.meta).toEqual([{ type: 'RESOLUTION', value: '1024x768' }]);
		expect(status.technical).toBeNull();
		const visible = [status.label, ...status.meta.map((m) => m.value)].join(' ');
		expect(visible).not.toMatch(RAW);
		expect(visible).not.toMatch(/acme|flux|checkpoint_loader/);
	});

	it('gives non-admins the fallback for an unknown pipe', () => {
		const status = resolveStageStatus({ currentStep: '<<PIPE:acme_pipe:bolt>>', pipeName: 'acme_pipe', isAdmin: false });
		expect(status.label).toBe(FALLBACK_STAGE_LABEL);
		expect(status.technical).toBeNull();
	});

	it('gives admins the same label plus technical detail', () => {
		const status = resolveStageStatus({ ...input, isAdmin: true });
		expect(status.label).toBe('Loading model');
		expect(status.meta.map((m) => m.type)).toEqual(['MODEL', 'RESOLUTION']);
		expect(status.technical).toContain('Pipe: acme_pipe');
		expect(status.technical).toContain('Name: checkpoint_loader/flux');
		expect(status.technical).toContain('Index: 4');
		expect(status.technical).toContain('<<MODEL:flux-dev.safetensors>>');
	});

	it('has no technical detail for admins when nothing is known', () => {
		expect(resolveStageStatus({ isAdmin: true }).technical).toBeNull();
	});
});
