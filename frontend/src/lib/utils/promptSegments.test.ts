import { describe, expect, it, vi, beforeEach } from 'vitest';
import type { ChipData } from '$lib/types/segments';

const searchPhrasebook = vi.fn();
vi.mock('$lib/services/api/index', () => ({
	api: { searchPhrasebook: (...args: unknown[]) => searchPhrasebook(...args) }
}));

import {
	applySegmentAdd,
	applySegmentDelete,
	applySegmentUpdate,
	locateSegmentInLists,
	restoreSegment,
	locateSegmentIndex,
	mergeChipSelections,
	resolvePromptSegments
} from './promptSegments';

function makeChip(overrides: Partial<ChipData> = {}): ChipData {
	return {
		id: overrides.id ?? 'chip-1',
		categoryPath: overrides.categoryPath ?? 'style.lighting',
		valueId: overrides.valueId ?? 'val-1',
		label: overrides.label ?? 'Soft light',
		value: overrides.value ?? 'soft light',
		allValues: overrides.allValues ?? [],
		shuffle: overrides.shuffle ?? false,
		autoRegen: overrides.autoRegen ?? false
	};
}

describe('resolvePromptSegments', () => {
	it('joins enabled content segments with a space', () => {
		expect(
			resolvePromptSegments([
				{ id: 'one', content: 'portrait' },
				{ id: 'two', content: 'studio light' }
			])
		).toBe('portrait studio light');
	});

	it('omits disabled segments but treats collapsed state as presentation-only', () => {
		expect(
			resolvePromptSegments([
				{ id: 'one', content: 'visible' },
				{ id: 'two', content: 'disabled', isDisabled: true },
				{ id: 'three', content: 'collapsed', isCollapsed: true }
			])
		).toBe('visible collapsed');
	});

	it('joins with a blank line instead of a space when the paragraph join is requested', () => {
		expect(
			resolvePromptSegments(
				[
					{ id: 'verse', content: '[Verse]\nrain on the window' },
					{ id: 'chorus', content: '[Chorus]\nnowhere to go' }
				],
				'paragraph'
			)
		).toBe('[Verse]\nrain on the window\n\n[Chorus]\nnowhere to go');
	});
});

describe('locateSegmentIndex', () => {
	const segments = [{ id: 'a', content: '' }, { id: 'b', content: '' }, { id: 'c', content: '' }];

	it('matches by id even when the proposed index is stale', () => {
		expect(locateSegmentIndex(segments, { segmentId: 'c', segmentIndex: 0 })).toBe(2);
	});

	it('falls back to the index when the id is not found', () => {
		expect(locateSegmentIndex(segments, { segmentId: 'missing', segmentIndex: 1 })).toBe(1);
	});

	it('returns -1 when neither the id nor the index resolve (e.g. an empty array)', () => {
		expect(locateSegmentIndex([], { segmentId: 'missing', segmentIndex: 0 })).toBe(-1);
		expect(locateSegmentIndex(segments, { segmentId: 'missing', segmentIndex: 99 })).toBe(-1);
	});
});

describe('mergeChipSelections', () => {
	it('carries the prior value/shuffle/autoRegen onto a re-parsed chip with the same category path', () => {
		const oldChips = {
			'old-1': makeChip({ id: 'old-1', categoryPath: 'style.lighting', valueId: 'val-dramatic', label: 'Dramatic', value: 'dramatic light', shuffle: true, autoRegen: true })
		};
		const newChips = {
			'new-1': makeChip({ id: 'new-1', categoryPath: 'style.lighting', valueId: 'val-default', label: 'Soft light', value: 'soft light', shuffle: false, autoRegen: false })
		};

		const merged = mergeChipSelections(oldChips, newChips);

		expect(merged['new-1']).toMatchObject({
			valueId: 'val-dramatic',
			label: 'Dramatic',
			value: 'dramatic light',
			shuffle: true,
			autoRegen: true
		});
	});

	it('leaves a newly introduced marker (no matching category path) as freshly parsed', () => {
		const oldChips = { 'old-1': makeChip({ id: 'old-1', categoryPath: 'style.lighting' }) };
		const newChips = { 'new-1': makeChip({ id: 'new-1', categoryPath: 'style.camera-angle' }) };

		expect(mergeChipSelections(oldChips, newChips)).toEqual(newChips);
	});

	it('consumes each old chip at most once when multiple new markers share a category path', () => {
		const oldChips = {
			'old-1': makeChip({ id: 'old-1', categoryPath: 'style.lighting', valueId: 'val-a', label: 'A' }),
			'old-2': makeChip({ id: 'old-2', categoryPath: 'style.lighting', valueId: 'val-b', label: 'B' })
		};
		const newChips = {
			'new-1': makeChip({ id: 'new-1', categoryPath: 'style.lighting', valueId: 'val-default' }),
			'new-2': makeChip({ id: 'new-2', categoryPath: 'style.lighting', valueId: 'val-default' })
		};

		const merged = mergeChipSelections(oldChips, newChips);

		expect([merged['new-1'].valueId, merged['new-2'].valueId].sort()).toEqual(['val-a', 'val-b']);
	});
});

describe('applySegmentUpdate', () => {
	beforeEach(() => {
		searchPhrasebook.mockReset();
	});

	it('returns null and leaves the source array untouched when the target segment cannot be resolved', async () => {
		const segments = [{ id: 'a', content: 'one' }];

		const result = await applySegmentUpdate(segments, {
			segmentId: 'missing',
			segmentIndex: 99,
			content: 'two'
		});

		expect(result).toBeNull();
		expect(searchPhrasebook).not.toHaveBeenCalled();
		expect(segments).toEqual([{ id: 'a', content: 'one' }]);
	});

	it('writes the new content on a plain-text update without touching other segments or chips', async () => {
		const segments = [
			{ id: 'a', content: 'portrait' },
			{ id: 'b', content: 'studio light' }
		];

		const result = await applySegmentUpdate(segments, {
			segmentId: 'b',
			segmentIndex: 1,
			content: 'dramatic light'
		});

		expect(searchPhrasebook).not.toHaveBeenCalled();
		expect(result).toEqual({
			index: 1,
			segments: [
				{ id: 'a', content: 'portrait' },
				{ id: 'b', content: 'dramatic light', chips: {} }
			]
		});
		// the input array is never mutated in place
		expect(segments[1].content).toBe('studio light');
	});

	it('leaves the affixes on a segment the chat rewrote', async () => {
		const segments = [{ id: 'a', content: 'portrait', prefix: '(', suffix: ':1.2)' }];

		const result = await applySegmentUpdate(segments, {
			segmentId: 'a',
			segmentIndex: 0,
			content: 'close-up portrait'
		});

		expect(result?.segments[0]).toMatchObject({
			content: 'close-up portrait',
			prefix: '(',
			suffix: ':1.2)'
		});
	});

	it('hydrates a #category.path marker in the new content and carries forward the prior chip choice', async () => {
		searchPhrasebook.mockResolvedValue({
			success: true,
			data: {
				values: [
					{ id: 'val-default', label: 'Soft light', value: 'soft light' },
					{ id: 'val-dramatic', label: 'Dramatic', value: 'dramatic light' }
				]
			}
		});
		const oldChip = makeChip({
			id: 'old-1',
			categoryPath: 'style.lighting',
			valueId: 'val-dramatic',
			label: 'Dramatic',
			value: 'dramatic light',
			shuffle: true
		});
		const segments = [{ id: 'a', content: '#style.lighting', chips: { 'old-1': oldChip } }];

		const result = await applySegmentUpdate(segments, {
			segmentId: 'a',
			segmentIndex: 0,
			content: '#style.lighting'
		});

		expect(searchPhrasebook).toHaveBeenCalledWith('style.lighting');
		expect(result?.index).toBe(0);
		const chips = result?.segments[0].chips as Record<string, ChipData>;
		expect(Object.keys(chips)).toHaveLength(1);
		const chip = Object.values(chips)[0];
		expect(chip).toMatchObject({
			categoryPath: 'style.lighting',
			// the freshly parsed chip carries the user's prior selection forward
			valueId: 'val-dramatic',
			label: 'Dramatic',
			value: 'dramatic light',
			shuffle: true
		});
	});
});

describe('applySegmentUpdate resource tokens', () => {
	const specs = [{ field: 'references', kind: 'image' as const, label: 'Pictures', token: '<Picture @>' }];
	const formValues = { references: [{ path: 'uploads/a.png' }, { path: 'uploads/b.png' }] };

	beforeEach(() => {
		searchPhrasebook.mockReset();
	});

	it('links plain tokens to the items at their positions and records the references', async () => {
		const result = await applySegmentUpdate(
			[{ id: 'a', content: 'old' }],
			{ segmentId: 'a', segmentIndex: 0, content: '<Picture 2> hugs <Picture 1>, then <Picture 5>' },
			{ specs, formValues }
		);

		expect(result?.segments[0].content).toBe(
			'@[references:uploads/b.png] hugs @[references:uploads/a.png], then <Picture 5>'
		);
		expect(Object.values(result?.segments[0].resources ?? {})).toEqual([
			{ field: 'references', item_key: 'uploads/b.png' },
			{ field: 'references', item_key: 'uploads/a.png' }
		]);
	});

	it('leaves tokens as text without a resource context', async () => {
		const result = await applySegmentUpdate([{ id: 'a', content: 'old' }], {
			segmentId: 'a',
			segmentIndex: 0,
			content: '<Picture 1> smiles'
		});

		expect(result?.segments[0].content).toBe('<Picture 1> smiles');
		expect(result?.segments[0].resources).toBeUndefined();
	});
});

describe('segment list add, delete and undo', () => {
	beforeEach(() => {
		searchPhrasebook.mockReset();
	});

	const lists = () => ({
		positive: [
			{ id: 'a', content: 'one' },
			{ id: 'b', content: 'two' },
			{ id: 'c', content: 'three' }
		],
		negative: [
			{ id: 'n1', content: 'blurry' },
			{ id: 'n2', content: 'watermark' }
		]
	});

	it('finds a segment by id in either list and falls back to the flat index the model saw', () => {
		expect(locateSegmentInLists(lists(), { segmentId: 'n2', segmentIndex: 0 })).toEqual({ list: 'negative', index: 1 });
		expect(locateSegmentInLists(lists(), { segmentId: 'gone', segmentIndex: 1 })).toEqual({ list: 'positive', index: 1 });
		expect(locateSegmentInLists(lists(), { segmentId: '', segmentIndex: 4 })).toEqual({ list: 'negative', index: 1 });
		expect(locateSegmentInLists(lists(), { segmentId: 'gone', segmentIndex: 9 })).toBeNull();
	});

	it('inserts a new segment right after the segment named by id', async () => {
		const result = await applySegmentAdd(lists(), { segmentId: 'a', segmentIndex: -1, content: 'fresh' });

		expect(result!.list).toBe('positive');
		expect(result!.index).toBe(1);
		expect(result!.lists.positive.map((s) => s.content)).toEqual(['one', 'fresh', 'two', 'three']);
		expect(result!.lists.negative).toEqual(lists().negative);
		expect(result!.lists.positive[1].id).not.toBe('a');
	});

	it('uses the index when the id is missing and appends when neither resolves', async () => {
		const byIndex = await applySegmentAdd(lists(), { segmentId: 'gone', segmentIndex: 1, content: 'fresh' });
		expect(byIndex!.lists.positive.map((s) => s.content)).toEqual(['one', 'two', 'fresh', 'three']);

		const appended = await applySegmentAdd(lists(), { segmentId: '', segmentIndex: -1, content: 'fresh' });
		expect(appended!.lists.positive.map((s) => s.content)).toEqual(['one', 'two', 'three', 'fresh']);
		expect(appended!.index).toBe(3);
	});

	it('targets the negative list when asked, ignoring an anchor from the other list', async () => {
		const result = await applySegmentAdd(lists(), {
			segmentId: 'a',
			segmentIndex: 0,
			content: 'extra fingers',
			list: 'negative'
		});

		expect(result!.list).toBe('negative');
		expect(result!.lists.negative.map((s) => s.content)).toEqual(['blurry', 'watermark', 'extra fingers']);
		expect(result!.lists.positive).toEqual(lists().positive);
	});

	it('follows the anchor into the negative list when no list is given', async () => {
		const result = await applySegmentAdd(lists(), { segmentId: 'n1', segmentIndex: -1, content: 'lowres' });

		expect(result!.list).toBe('negative');
		expect(result!.lists.negative.map((s) => s.content)).toEqual(['blurry', 'lowres', 'watermark']);
	});

	it('replaces a lone blank segment instead of leaving an empty one behind', async () => {
		const blank = { positive: [{ id: 'blank', content: '', chips: {}, enabled: true, type: 'content' as const }], negative: [] };

		const result = await applySegmentAdd(blank, { segmentId: '', segmentIndex: -1, content: 'fresh' });

		expect(result!.lists.positive.map((s) => s.content)).toEqual(['fresh']);
	});

	it('refuses an empty addition', async () => {
		expect(await applySegmentAdd(lists(), { segmentId: 'a', segmentIndex: 0, content: '   ' })).toBeNull();
	});

	it('removes the segment by id or index and reports what was removed', () => {
		const byId = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 });
		expect(byId!.lists.positive.map((s) => s.id)).toEqual(['a', 'c']);
		expect(byId!.removal).toEqual({
			list: 'positive',
			index: 1,
			segment: { id: 'b', content: 'two' },
			previousId: 'a',
			nextId: 'c'
		});

		const byIndex = applySegmentDelete(lists(), { segmentId: 'gone', segmentIndex: 3 });
		expect(byIndex!.lists.negative.map((s) => s.id)).toEqual(['n2']);
		expect(applySegmentDelete(lists(), { segmentId: 'gone', segmentIndex: 9 })).toBeNull();
	});

	it('keeps one blank segment when the last one is deleted', () => {
		const result = applySegmentDelete({ positive: [{ id: 'only', content: 'solo' }], negative: [] }, { segmentId: 'only', segmentIndex: 0 });

		expect(result!.lists.positive).toHaveLength(1);
		expect(result!.lists.positive[0].content).toBe('');
		expect(result!.removal.segment.content).toBe('solo');
	});

	it('undoes a delete by putting the segment back at its position', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 })!;

		const restored = restoreSegment(deleted.lists, deleted.removal);

		expect(restored).toEqual(lists());
	});

	it('undoes a delete after an add before it shifted the indexes', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'c', segmentIndex: 0 })!;
		const shifted = {
			...deleted.lists,
			positive: [deleted.lists.positive[0], { id: 'x', content: 'new' }, ...deleted.lists.positive.slice(1)]
		};

		expect(restoreSegment(shifted, deleted.removal).positive.map((s) => s.id)).toEqual(['a', 'x', 'b', 'c']);
	});

	it('undoes a delete after an add following its previous neighbour', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 })!;
		const shifted = {
			...deleted.lists,
			positive: [deleted.lists.positive[0], { id: 'x', content: 'new' }, deleted.lists.positive[1]]
		};

		expect(restoreSegment(shifted, deleted.removal).positive.map((s) => s.id)).toEqual(['a', 'b', 'x', 'c']);
	});

	it('restores before the next neighbour when the previous one is gone', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 })!;
		const withoutPrevious = { ...deleted.lists, positive: [{ id: 'z', content: 'z' }, { id: 'y', content: 'y' }, deleted.lists.positive[1]] };

		expect(restoreSegment(withoutPrevious, deleted.removal).positive.map((s) => s.id)).toEqual(['z', 'y', 'b', 'c']);
	});

	it('falls back to the clamped old index when both neighbours are gone', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 })!;
		const bothGone = { ...deleted.lists, positive: [{ id: 'y', content: 'y' }, { id: 'z', content: 'z' }, { id: 'w', content: 'w' }] };

		expect(restoreSegment(bothGone, deleted.removal).positive.map((s) => s.id)).toEqual(['y', 'b', 'z', 'w']);
		expect(restoreSegment({ ...deleted.lists, positive: [{ id: 'y', content: 'y' }] }, deleted.removal).positive.map((s) => s.id)).toEqual(['y', 'b']);
	});

	it('undoes the deletion of the last segment by replacing the placeholder blank', () => {
		const deleted = applySegmentDelete({ positive: [{ id: 'only', content: 'solo' }], negative: [] }, { segmentId: 'only', segmentIndex: 0 })!;

		const restored = restoreSegment(deleted.lists, deleted.removal);

		expect(restored.positive).toEqual([{ id: 'only', content: 'solo' }]);
	});

	it('does not duplicate a segment that is already back', () => {
		const deleted = applySegmentDelete(lists(), { segmentId: 'b', segmentIndex: 0 })!;
		const once = restoreSegment(deleted.lists, deleted.removal);

		expect(restoreSegment(once, deleted.removal)).toBe(once);
	});
});
