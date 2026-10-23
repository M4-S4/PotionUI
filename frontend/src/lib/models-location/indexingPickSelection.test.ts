import { describe, expect, it } from 'vitest';
import {
	emptyPickSelection,
	pickAllMatchingSelected,
	pickBytes,
	pickCount,
	pickIsApproximate,
	pickIsSelected,
	pickRequests,
	pickSelectAllMatching,
	pickToggle
} from './indexingPickSelection';
import type { IndexingPendingItem } from '$lib/services/api/models';

function item(id: string, filename: string, model_type: string, size: number): IndexingPendingItem {
	return {
		model_id: id,
		location_id: `loc-${id}`,
		filename,
		model_type,
		size,
		root_label: 'Library',
		rel_path: filename,
		queued: false
	};
}

const XL_A = item('a', 'sd_xl_base.safetensors', 'checkpoint', 100);
const XL_B = item('b', 'juggernaut-XL.safetensors', 'checkpoint', 200);
const XL_LORA = item('c', 'neon_xl_lora.safetensors', 'lora', 10);
const FLUX = item('d', 'flux_dev.safetensors', 'diffusion_model', 1000);

const XL = { query: 'xl', modelType: null };

describe('ticking files', () => {
	it('adds and removes single files with their sizes', () => {
		let selection = pickToggle(emptyPickSelection(), XL_A, true);
		selection = pickToggle(selection, FLUX, true);
		expect(pickCount(selection)).toBe(2);
		expect(pickBytes(selection)).toBe(1100);
		expect(pickIsSelected(selection, XL_A)).toBe(true);
		selection = pickToggle(selection, XL_A, false);
		expect(pickIsSelected(selection, XL_A)).toBe(false);
		expect(pickCount(selection)).toBe(1);
	});

	it('asks for the ticked files by id', () => {
		const selection = pickToggle(pickToggle(emptyPickSelection(), XL_A, true), FLUX, true);
		expect(pickRequests(selection)).toEqual([{ mode: 'selected', model_ids: ['a', 'd'] }]);
	});

	it('sends nothing for an empty selection', () => {
		expect(pickRequests(emptyPickSelection())).toEqual([]);
	});
});

describe('select all matching', () => {
	it('selects every match by count and size, including rows that are not loaded', () => {
		const selection = pickSelectAllMatching(emptyPickSelection(), XL, 101, 5000);
		expect(pickCount(selection)).toBe(101);
		expect(pickBytes(selection)).toBe(5000);
		expect(pickIsSelected(selection, XL_A)).toBe(true);
		expect(pickIsSelected(selection, FLUX)).toBe(false);
	});

	it('matches names case-insensitively and by the type filter', () => {
		const selection = pickSelectAllMatching(emptyPickSelection(), { query: 'XL', modelType: 'checkpoint' }, 2, 300);
		expect(pickIsSelected(selection, XL_A)).toBe(true);
		expect(pickIsSelected(selection, XL_LORA)).toBe(false);
	});

	it('asks for the matches by filter', () => {
		const selection = pickSelectAllMatching(emptyPickSelection(), { query: ' xl ', modelType: 'lora' }, 3, 30);
		expect(pickRequests(selection)).toEqual([{ mode: 'matching', query: 'xl', model_type: 'lora' }]);
		const bare = pickSelectAllMatching(emptyPickSelection(), { query: '', modelType: null }, 472, 1);
		expect(pickRequests(bare)).toEqual([{ mode: 'matching' }]);
	});

	it('keeps an unticked match out of the count, the size and the request', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), XL, 101, 5000);
		selection = pickToggle(selection, XL_B, false);
		expect(pickIsSelected(selection, XL_B)).toBe(false);
		expect(pickCount(selection)).toBe(100);
		expect(pickBytes(selection)).toBe(4800);
		expect(pickRequests(selection)).toEqual([{ mode: 'matching', query: 'xl', exclude_model_ids: ['b'] }]);
		selection = pickToggle(selection, XL_B, true);
		expect(pickCount(selection)).toBe(101);
		expect(pickRequests(selection)).toEqual([{ mode: 'matching', query: 'xl' }]);
	});

	it('sends a file outside the filter as its own selected request', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), XL, 101, 5000);
		selection = pickToggle(selection, FLUX, true);
		expect(pickCount(selection)).toBe(102);
		expect(pickBytes(selection)).toBe(6000);
		expect(pickRequests(selection)).toEqual([
			{ mode: 'matching', query: 'xl' },
			{ mode: 'selected', model_ids: ['d'] }
		]);
	});

	it('absorbs files ticked earlier that the new filter covers', () => {
		let selection = pickToggle(pickToggle(emptyPickSelection(), XL_A, true), FLUX, true);
		selection = pickSelectAllMatching(selection, XL, 101, 5000);
		expect(pickCount(selection)).toBe(102);
		expect(pickRequests(selection)).toEqual([
			{ mode: 'matching', query: 'xl' },
			{ mode: 'selected', model_ids: ['d'] }
		]);
	});

	it('a second select all under another type adds to the first instead of replacing it', () => {
		const checkpoints = { query: '', modelType: 'checkpoint' };
		const loras = { query: '', modelType: 'lora' };
		let selection = pickSelectAllMatching(emptyPickSelection(), checkpoints, 65, 6500);
		selection = pickToggle(selection, FLUX, true);
		selection = pickSelectAllMatching(selection, loras, 301, 900);

		expect(pickCount(selection)).toBe(367);
		expect(pickBytes(selection)).toBe(8400);
		expect(pickIsSelected(selection, XL_A)).toBe(true);
		expect(pickIsSelected(selection, XL_LORA)).toBe(true);
		expect(pickIsSelected(selection, FLUX)).toBe(true);
		expect(pickRequests(selection)).toEqual([
			{ mode: 'matching', model_type: 'checkpoint' },
			{ mode: 'matching', model_type: 'lora' },
			{ mode: 'selected', model_ids: ['d'] }
		]);
		expect(pickIsApproximate(selection)).toBe(false);
	});

	it('unticking a file only excludes it from the scope it belongs to', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: '', modelType: 'checkpoint' }, 65, 6500);
		selection = pickSelectAllMatching(selection, { query: '', modelType: 'lora' }, 301, 900);
		selection = pickToggle(selection, XL_LORA, false);

		expect(pickIsSelected(selection, XL_LORA)).toBe(false);
		expect(pickIsSelected(selection, XL_A)).toBe(true);
		expect(pickCount(selection)).toBe(365);
		expect(pickRequests(selection)).toEqual([
			{ mode: 'matching', model_type: 'checkpoint' },
			{ mode: 'matching', model_type: 'lora', exclude_model_ids: ['c'] }
		]);
	});

	it('a narrower select all inside a wider one adds nothing and re-includes what was unticked there', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: '', modelType: 'checkpoint' }, 65, 6500);
		selection = pickToggle(selection, XL_A, false);
		selection = pickToggle(selection, XL_B, false);
		expect(pickCount(selection)).toBe(63);

		selection = pickSelectAllMatching(selection, { query: 'juggernaut', modelType: 'checkpoint' }, 1, 200);

		expect(selection.scopes).toHaveLength(1);
		expect(pickCount(selection)).toBe(64);
		expect(pickIsSelected(selection, XL_B)).toBe(true);
		expect(pickIsSelected(selection, XL_A)).toBe(false);
		expect(pickRequests(selection)).toEqual([{ mode: 'matching', model_type: 'checkpoint', exclude_model_ids: ['a'] }]);
	});

	it('a wider select all swallows the narrower ones it covers', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: 'xl', modelType: 'checkpoint' }, 2, 300);
		selection = pickSelectAllMatching(selection, { query: '', modelType: null }, 472, 9000);

		expect(selection.scopes).toHaveLength(1);
		expect(pickCount(selection)).toBe(472);
		expect(pickRequests(selection)).toEqual([{ mode: 'matching' }]);
	});

	it('flags the totals as approximate when two scopes may overlap', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: 'xl', modelType: null }, 101, 5000);
		expect(pickIsApproximate(selection)).toBe(false);
		selection = pickSelectAllMatching(selection, { query: '', modelType: 'lora' }, 301, 900);

		expect(selection.scopes).toHaveLength(2);
		expect(pickIsApproximate(selection)).toBe(true);
		expect(pickIsApproximate(pickSelectAllMatching(emptyPickSelection(), XL, 1, 1))).toBe(false);
	});
});

describe('pickAllMatchingSelected', () => {
	it('is true when a scope covers the filter and nothing in it is unticked', () => {
		const selection = pickSelectAllMatching(emptyPickSelection(), XL, 101, 5000);
		expect(pickAllMatchingSelected(selection, { query: ' XL', modelType: null }, 101)).toBe(true);
		expect(pickAllMatchingSelected(selection, { query: 'xl_base', modelType: 'checkpoint' }, 3)).toBe(true);
	});

	it('is false for another filter, a wider filter, an unticked match or no matches', () => {
		const selection = pickSelectAllMatching(emptyPickSelection(), XL, 101, 5000);
		expect(pickAllMatchingSelected(selection, { query: 'flux', modelType: null }, 4)).toBe(false);
		expect(pickAllMatchingSelected(selection, { query: '', modelType: null }, 400)).toBe(false);
		expect(pickAllMatchingSelected(pickToggle(selection, XL_A, false), XL, 101)).toBe(false);
		expect(pickAllMatchingSelected(selection, XL, 0)).toBe(false);
		expect(pickAllMatchingSelected(emptyPickSelection(), XL, 101)).toBe(false);
	});

	it('ignores an unticked file that the filter does not match', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: '', modelType: 'checkpoint' }, 65, 6500);
		selection = pickToggle(selection, XL_A, false);
		expect(pickAllMatchingSelected(selection, { query: 'juggernaut', modelType: 'checkpoint' }, 1)).toBe(true);
	});

	it('checks each scope on its own', () => {
		let selection = pickSelectAllMatching(emptyPickSelection(), { query: '', modelType: 'checkpoint' }, 65, 6500);
		selection = pickSelectAllMatching(selection, { query: '', modelType: 'lora' }, 301, 900);
		expect(pickAllMatchingSelected(selection, { query: '', modelType: 'lora' }, 301)).toBe(true);
		expect(pickAllMatchingSelected(selection, { query: '', modelType: 'diffusion_model' }, 58)).toBe(false);
	});
});
