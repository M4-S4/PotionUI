import { describe, it, expect } from 'vitest';
import type { PresetInfo } from '$lib/types/api';
import { presetSections, presetIssue, presetUnavailableReason, matchesPreset, engineLabel } from './presetDropdownModel';

function preset(id: string, extra: Partial<PresetInfo> = {}): PresetInfo {
	return { id, name: id.toUpperCase(), version: '1.0.0', tags: [], category: 'image', engine: 'native', ...extra };
}

const CATALOG = [
	preset('sdxl', { tags: ['photo'] }),
	preset('flux1', { engine: 'comfyui', engine_available: false }),
	preset('krea'),
	preset('wan', { category: 'video' }),
	preset('seedvr2', { category: 'utility' }),
	preset('pony', { requirements_summary: { ok: 1, missing: 1, unknown: 0, optional_missing: 0 } })
];

describe('presetDropdownModel', () => {
	it('lists Recent (without the current preset, up to three), Pinned, then the families in order', () => {
		const sections = presetSections(CATALOG, '', { pinned: ['seedvr2'], recent: ['sdxl', 'krea', 'wan', 'flux1', 'pony'] }, 'sdxl');
		expect(sections.map((s) => [s.label, s.presets.map((p) => p.id)])).toEqual([
			['Recent', ['krea', 'wan', 'flux1']],
			['Pinned', ['seedvr2']],
			['Image', ['sdxl', 'krea', 'flux1', 'pony']],
			['Video', ['wan']],
			['Tools', ['seedvr2']]
		]);
	});

	it('moves presets that cannot run to the end of their family', () => {
		const image = presetSections(CATALOG, '', { pinned: [], recent: [] }, null)[0];
		expect(image.presets.map((p) => p.id)).toEqual(['sdxl', 'krea', 'flux1', 'pony']);
	});

	it('searches names, families by label, engines and tags, and drops Recent and Pinned', () => {
		expect(matchesPreset(CATALOG[4], 'tools')).toBe(true);
		expect(matchesPreset(CATALOG[1], 'comfy')).toBe(true);
		expect(matchesPreset(CATALOG[0], 'photo')).toBe(true);
		expect(matchesPreset(CATALOG[2], 'video')).toBe(false);
		const sections = presetSections(CATALOG, 'comfy', { pinned: ['flux1'], recent: ['flux1'] }, null);
		expect(sections.map((s) => s.label)).toEqual(['Image']);
		expect(sections[0].presets.map((p) => p.id)).toEqual(['flux1']);
	});

	it('explains why a preset cannot run, and flags a GPU that is too small without blocking', () => {
		expect(presetIssue(CATALOG[1])).toMatchObject({ kind: 'backend', label: 'No ComfyUI backend', blocking: true });
		expect(presetIssue(CATALOG[5])).toMatchObject({ kind: 'missing', label: '1 missing', blocking: true });
		expect(presetIssue(preset('big', { requires: { min_vram_gb: 24 } }), 16)).toEqual({
			kind: 'vram',
			label: 'needs 24 GB · this GPU 16 GB',
			detail: 'Needs about 24 GB of VRAM; this GPU has 16 GB.',
			blocking: false
		});
		expect(presetIssue(preset('big', { requires: { min_vram_gb: 24 } }), 32)).toBeNull();
		expect(presetIssue(CATALOG[0])).toBeNull();
	});

	it('gives Generate a reason only for presets that cannot run', () => {
		expect(presetUnavailableReason(CATALOG[1])).toBe('No ComfyUI backend is set up');
		expect(presetUnavailableReason(CATALOG[5])).toBe('1 required file is missing');
		expect(presetUnavailableReason(preset('two', { requirements_summary: { ok: 0, missing: 2, unknown: 0, optional_missing: 0 } }))).toBe(
			'2 required files are missing'
		);
		expect(presetUnavailableReason(CATALOG[0])).toBeUndefined();
		expect(presetUnavailableReason(null)).toBeUndefined();
		expect(engineLabel('native')).toBe('native');
	});
});
