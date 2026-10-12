import { describe, expect, it } from 'vitest';
import { batchCaption, groupByPrompt, isPromptBatch, type PromptEntry } from './promptGroups';

function image(promptIndex: number | undefined, promptText?: string) {
	return { url: `/${promptIndex}.png`, promptIndex, promptText };
}

function entries(items: ReturnType<typeof image>[]): PromptEntry<ReturnType<typeof image>>[] {
	return items.map((item, index) => ({ item, index, kind: 'image' }));
}

describe('groupByPrompt', () => {
	it('groups tiles by authored prompt and numbers images within each prompt', () => {
		const items = [image(1, 'fox'), image(0, 'lighthouse'), image(1, 'fox'), image(0, 'lighthouse')];

		const groups = groupByPrompt(entries(items));

		expect(groups.map((group) => group.promptIndex)).toEqual([0, 1]);
		expect(groups[0].promptText).toBe('lighthouse');
		expect(groups[1].tiles.map((tile) => [tile.index, tile.imageNumber])).toEqual([
			[0, 1],
			[2, 2]
		]);
	});

	it('skips outputs that carry no prompt index', () => {
		expect(groupByPrompt(entries([image(undefined), image(0, 'a')]))).toHaveLength(1);
	});
});

describe('isPromptBatch', () => {
	it('needs at least two distinct prompts', () => {
		expect(isPromptBatch([image(0), image(0)])).toBe(false);
		expect(isPromptBatch([image(undefined), image(undefined)])).toBe(false);
		expect(isPromptBatch([image(0), image(1)])).toBe(true);
	});
});

describe('batchCaption', () => {
	it('names the prompt and the image position within it', () => {
		const items = [image(0, 'a'), image(1, 'fox'), image(1, 'fox')];

		expect(batchCaption(items, items[2])).toEqual({ title: 'Prompt 2 · image 2 of 2', prompt: 'fox' });
	});

	it('is absent outside a batch', () => {
		const items = [image(undefined), image(undefined)];

		expect(batchCaption(items, items[0])).toBeNull();
		expect(batchCaption(items, null)).toBeNull();
	});
});
