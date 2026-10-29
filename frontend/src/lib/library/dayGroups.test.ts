import { describe, expect, it } from 'vitest';
import { groupByDay } from './dayGroups';

function daysAgo(days: number): string {
	const date = new Date();
	date.setHours(12, 0, 0, 0);
	date.setDate(date.getDate() - days);
	return date.toISOString();
}

const row = (id: string, days: number | null) => ({ id, created_at: days === null ? undefined : daysAgo(days) });

const shape = (groups: ReturnType<typeof groupByDay<ReturnType<typeof row>>>) =>
	groups.map((group) => [group.label, group.items.map((item) => item.id)]);

describe('groupByDay', () => {
	it('groups consecutive items of the same day under that day', () => {
		const groups = groupByDay([row('a', 0), row('b', 0), row('c', 1)]);

		expect(shape(groups)).toEqual([
			['Today', ['a', 'b']],
			['Yesterday', ['c']]
		]);
	});

	it('gives every group a different key', () => {
		const groups = groupByDay([row('a', 0), row('b', 1), row('c', 2)]);

		expect(new Set(groups.map((group) => group.key)).size).toBe(3);
	});

	it('opens an Undated group for an item with no date', () => {
		expect(shape(groupByDay([row('a', 0), row('b', null)]))).toEqual([
			['Today', ['a']],
			['Undated', ['b']]
		]);
	});

	it('never repeats a key when the same day shows up again later in the list', () => {
		const groups = groupByDay([row('a', 0), row('b', 1), row('c', 0), row('d', 1)]);

		const keys = groups.map((group) => group.key);
		expect(new Set(keys).size).toBe(keys.length);
		expect(shape(groups)).toEqual([
			['Today', ['a']],
			['Yesterday', ['b']],
			['Today', ['c']],
			['Yesterday', ['d']]
		]);
	});

	it('files a fresh item in the day of the item it sits in front of', () => {
		const groups = groupByDay([row('a', 0), row('n1', 0), row('old', 3), row('older', 3)], ['n1']);

		expect(shape(groups)).toEqual([
			['Today', ['a']],
			[expect.stringMatching(/\d/), ['n1', 'old', 'older']]
		]);
	});

	it('keeps a run of fresh items together in the source day', () => {
		const groups = groupByDay([row('a', 0), row('p1', 0), row('p2', 0), row('src', 5), row('b', 5)], ['p1', 'p2']);

		expect(groups.map((group) => group.items.map((item) => item.id))).toEqual([['a'], ['p1', 'p2', 'src', 'b']]);
	});

	it('leaves a fresh item in its own day when nothing follows it', () => {
		expect(shape(groupByDay([row('a', 1), row('n1', 0)], ['n1']))).toEqual([
			['Yesterday', ['a']],
			['Today', ['n1']]
		]);
	});

	it('does not change the order of the items', () => {
		const list = [row('a', 0), row('n1', 0), row('old', 3)];

		expect(groupByDay(list, ['n1']).flatMap((group) => group.items.map((item) => item.id))).toEqual([
			'a',
			'n1',
			'old'
		]);
	});

	it('is the plain grouping when nothing is fresh', () => {
		const list = [row('a', 0), row('b', 1)];

		expect(groupByDay(list, [])).toEqual(groupByDay(list));
	});
});
