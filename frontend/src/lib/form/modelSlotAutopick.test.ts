import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
	autopickValue,
	collectAutopickSlots,
	createModelSlotAutopick,
	decideAutopick,
	forgetClearedSlots,
	isSlotClearEdit,
	resolveSlotTagIds,
	markSlotCleared,
	slotSearchRequest,
	type AutopickSlot
} from './modelSlotAutopick';

const model = (id: string) => ({ id, filename: `${id}.safetensors` });

const base = {
	hasValue: false,
	restored: false,
	cleared: false,
	result: { models: [model('a')], total: 1 }
};

describe('decideAutopick', () => {
	it('picks the single candidate of an empty required slot', () => {
		expect(decideAutopick(base)).toEqual(model('a'));
	});

	it('does nothing with zero or two candidates', () => {
		expect(decideAutopick({ ...base, result: { models: [], total: 0 } })).toBeNull();
		expect(decideAutopick({ ...base, result: { models: [model('a'), model('b')], total: 2 } })).toBeNull();
	});

	it('does nothing with two rows and no reported total', () => {
		expect(decideAutopick({ ...base, result: { models: [model('a'), model('b')] } })).toBeNull();
	});

	it('does nothing when the page holds one row but more exist', () => {
		expect(decideAutopick({ ...base, result: { models: [model('a')], total: 3 } })).toBeNull();
	});

	it('never overwrites a value', () => {
		expect(decideAutopick({ ...base, hasValue: true })).toBeNull();
	});

	it('skips restored and user-cleared slots', () => {
		expect(decideAutopick({ ...base, restored: true })).toBeNull();
		expect(decideAutopick({ ...base, cleared: true })).toBeNull();
	});

	it('ignores a failed search and rows without an id', () => {
		expect(decideAutopick({ ...base, result: null })).toBeNull();
		expect(decideAutopick({ ...base, result: { models: [{ filename: 'x' }] } })).toBeNull();
	});
});

describe('collectAutopickSlots', () => {
	const field = (over: Record<string, unknown>) => ({
		type: 'model',
		required: true,
		configuration: { model_type: 'vae' },
		...over
	});

	it('collects empty required model fields, nested, with filters', () => {
		const slots = collectAutopickSlots(
			[
				{
					type: 'group',
					children: [
						field({ name: 'vae' }),
						field({ name: 'tools', configuration: { model_types: ['unet', 'diffusion_model'], unquantized_only: true, filter_tags: ['t1'], tags: ['x'] } })
					]
				}
			] as any,
			{ vae: { modelPath: '', tagFilters: [] } },
			'p1'
		);
		expect(slots.map((s) => s.name)).toEqual(['vae', 'tools']);
		expect(slots[0]).toMatchObject({ presetId: 'p1', modelType: 'vae', unquantizedOnly: false });
		expect(slots[1]).toMatchObject({
			modelType: 'unet',
			modelTypes: ['unet', 'diffusion_model'],
			unquantizedOnly: true,
			filterTagIds: ['t1'],
			tagFilters: ['x']
		});
	});

	it('skips optional, filled, hidden, readonly, disabled and non-model fields', () => {
		const slots = collectAutopickSlots(
			[
				field({ name: 'optional', required: false }),
				field({ name: 'filled' }),
				field({ name: 'hidden', visible: false }),
				field({ name: 'locked', readonly: true }),
				field({ name: 'off', disabled: true }),
				field({ name: 'lora', type: 'lora_picker' })
			] as any,
			{ filled: { modelPath: 'model:1', tagFilters: [] } },
			'p1'
		);
		expect(slots).toEqual([]);
	});

	it('prefers the field preset_id and skips slots with no preset scope', () => {
		expect(collectAutopickSlots([field({ name: 'a', preset_id: 'own' })] as any, {}, 'p1')[0].presetId).toBe('own');
		expect(collectAutopickSlots([field({ name: 'a' })] as any, {}, '')).toEqual([]);
	});
});

describe('resolveSlotTagIds', () => {
	const tags = [{ id: 't1', name: 'a' }, { id: 't2', name: 'b' }];

	it('maps names to ids', () => {
		expect(resolveSlotTagIds(['b', 'a'], tags)).toEqual(['t2', 't1']);
		expect(resolveSlotTagIds([], null)).toEqual([]);
	});

	it('returns null when the lookup failed or any name is unknown', () => {
		expect(resolveSlotTagIds(['a'], null)).toBeNull();
		expect(resolveSlotTagIds(['a', 'zzz'], tags)).toBeNull();
	});
});

describe('isSlotClearEdit', () => {
	const filled = { modelPath: 'model:1', tagFilters: [] };
	const empty = { modelPath: '', tagFilters: [] };

	it('is true only when a model field goes from a value to empty', () => {
		expect(isSlotClearEdit(true, filled, empty)).toBe(true);
		expect(isSlotClearEdit(true, filled, filled)).toBe(false);
		expect(isSlotClearEdit(true, empty, empty)).toBe(false);
		expect(isSlotClearEdit(true, empty, filled)).toBe(false);
		expect(isSlotClearEdit(false, filled, empty)).toBe(false);
	});
});

describe('autopickValue and slotSearchRequest', () => {
	const slot: AutopickSlot = {
		name: 'vae',
		presetId: 'p1',
		modelType: 'vae',
		modelTypes: ['vae', 'unet'],
		unquantizedOnly: true,
		filterTagIds: ['t1'],
		tagFilters: ['x']
	};

	it('stores the same shape a manual pick does', () => {
		expect(autopickValue(slot, model('m1'))).toEqual({ modelPath: 'model:m1', tagFilters: ['x'] });
	});

	it('asks the preset-scoped endpoint with the field filters and a page of two', () => {
		expect(slotSearchRequest(slot, ['id1'])).toEqual({
			kind: 'preset',
			presetId: 'p1',
			modelType: 'vae',
			search: undefined,
			opts: {
				limit: 2,
				tagIds: 'id1',
				anyTagIds: 't1',
				favoritesOnly: undefined,
				modelTypes: 'vae,unet',
				unquantizedOnly: true
			}
		});
	});
});

describe('createModelSlotAutopick', () => {
	const slot = (name: string): AutopickSlot => ({
		name,
		presetId: 'p1',
		modelType: name,
		modelTypes: [],
		unquantizedOnly: false,
		filterTagIds: [],
		tagFilters: []
	});

	beforeEach(() => {
		vi.useFakeTimers();
		forgetClearedSlots();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	function setup(results: Record<string, { models: any[]; total?: number }>) {
		const search = vi.fn(async (s: AutopickSlot) => results[s.name] ?? null);
		const values: Record<string, string> = {};
		const picks: string[] = [];
		const pinned = new Set<string>();
		const controller = createModelSlotAutopick({ tabKey: 'tab1', search });
		const sync = (loadKey = 'k1', names = ['diffusion_model', 'text_encoder', 'vae']) =>
			controller.sync({
				loadKey,
				slots: names.filter((n) => !values[n]).map(slot),
				isPinned: (n) => pinned.has(n),
				isEmpty: (n) => !values[n],
				onPick: (s, m) => {
					values[s.name] = m.id;
					picks.push(s.name);
				}
			});
		return { search, values, picks, pinned, controller, sync };
	}

	const one = { models: [model('only')], total: 1 };

	it('issues one request per slot however often the form re-syncs', async () => {
		const t = setup({ diffusion_model: one, text_encoder: one, vae: one });
		for (let i = 0; i < 25; i++) t.sync();
		await vi.advanceTimersByTimeAsync(500);
		for (let i = 0; i < 25; i++) t.sync();
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).toHaveBeenCalledTimes(3);
		expect(t.picks.sort()).toEqual(['diffusion_model', 'text_encoder', 'vae']);
	});

	it('debounces the first request', async () => {
		const t = setup({ vae: one });
		t.sync('k1', ['vae']);
		expect(t.search).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).toHaveBeenCalledTimes(1);
	});

	it('does not refill a slot that stayed ambiguous and does not loop', async () => {
		const two = { models: [model('a'), model('b')], total: 2 };
		const t = setup({ vae: two });
		t.sync('k1', ['vae']);
		await vi.advanceTimersByTimeAsync(500);
		t.sync('k1', ['vae']);
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).toHaveBeenCalledTimes(1);
		expect(t.picks).toEqual([]);
	});

	it('does not request restored slots', async () => {
		const t = setup({ vae: one });
		t.pinned.add('vae');
		t.sync('k1', ['vae']);
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).not.toHaveBeenCalled();
	});

	it('does not refill a slot the user cleared, across loads', async () => {
		const t = setup({ vae: one });
		markSlotCleared('tab1', 'vae');
		t.sync('k1', ['vae']);
		t.sync('k2', ['vae']);
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).not.toHaveBeenCalled();
	});

	it('drops a result when the user filled the slot while it was in flight', async () => {
		const t = setup({ vae: one });
		t.sync('k1', ['vae']);
		t.values.vae = 'manual';
		await vi.advanceTimersByTimeAsync(500);
		expect(t.picks).toEqual([]);
		expect(t.values.vae).toBe('manual');
	});

	it('drops a result when the slot was cleared while it was in flight', async () => {
		const t = setup({ vae: one });
		t.sync('k1', ['vae']);
		markSlotCleared('tab1', 'vae');
		await vi.advanceTimersByTimeAsync(500);
		expect(t.picks).toEqual([]);
	});

	it('drops a result from a previous load and re-requests for the new one', async () => {
		const t = setup({ vae: one });
		t.sync('k1', ['vae']);
		await vi.advanceTimersByTimeAsync(10);
		t.sync('k2', ['vae']);
		await vi.advanceTimersByTimeAsync(500);
		expect(t.search).toHaveBeenCalledTimes(1);
		expect(t.picks).toEqual(['vae']);
	});

	it('ignores an in-flight result from a previous load', async () => {
		let resolveFirst: (v: { models: any[]; total: number }) => void = () => {};
		const search = vi
			.fn()
			.mockImplementationOnce(() => new Promise((r) => (resolveFirst = r)))
			.mockResolvedValueOnce({ models: [model('second')], total: 1 });
		const controller = createModelSlotAutopick({ tabKey: 'tab1', search });
		const onPick = vi.fn();
		const input = (loadKey: string) => ({ loadKey, slots: [slot('vae')], isPinned: () => false, isEmpty: () => true, onPick });
		controller.sync(input('k1'));
		await vi.advanceTimersByTimeAsync(500);
		expect(search).toHaveBeenCalledTimes(1);
		controller.sync(input('k2'));
		resolveFirst({ models: [model('first')], total: 1 });
		await vi.advanceTimersByTimeAsync(0);
		expect(onPick).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(500);
		expect(search).toHaveBeenCalledTimes(2);
		expect(onPick).toHaveBeenCalledTimes(1);
		expect(onPick.mock.calls[0][1].id).toBe('second');
	});

	it('survives a failing search and stops after dispose', async () => {
		const search = vi.fn(async () => {
			throw new Error('boom');
		});
		const controller = createModelSlotAutopick({ tabKey: 'tab1', search });
		const onPick = vi.fn();
		const input = { loadKey: 'k', slots: [slot('vae')], isPinned: () => false, isEmpty: () => true, onPick };
		controller.sync(input);
		await vi.advanceTimersByTimeAsync(500);
		expect(onPick).not.toHaveBeenCalled();
		controller.dispose();
		controller.sync({ ...input, loadKey: 'k2' });
		await vi.advanceTimersByTimeAsync(500);
		expect(search).toHaveBeenCalledTimes(1);
	});
});
