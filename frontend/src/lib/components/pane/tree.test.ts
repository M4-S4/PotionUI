import { describe, it, expect } from 'vitest';
import { buildTree, flattenTree, descendantIds, filterTree, ancestorsOf } from './tree';
import type { TreeItem } from './types';

interface Item extends TreeItem {
	name: string;
}

function item(id: string, parent_id: string | null, name: string): Item {
	return { id, parent_id, name };
}

describe('buildTree', () => {
	it('treats a missing/dangling parent_id as a root', () => {
		const items = [item('a', 'does-not-exist', 'A'), item('b', null, 'B')];
		const tree = buildTree(items);
		expect(tree.map((n) => n.item.id).sort()).toEqual(['a', 'b']);
		expect(tree.every((n) => n.depth === 0)).toBe(true);
	});

	it('numbers depth per level', () => {
		const items = [
			item('root', null, 'Root'),
			item('child', 'root', 'Child'),
			item('grandchild', 'child', 'Grandchild')
		];
		const tree = buildTree(items);
		expect(tree[0].depth).toBe(0);
		expect(tree[0].children[0].depth).toBe(1);
		expect(tree[0].children[0].children[0].depth).toBe(2);
	});

	it('sorts roots and each sibling group by name by default', () => {
		const items = [
			item('b', null, 'Bravo'),
			item('a', null, 'Alpha'),
			item('c1', 'a', 'Charlie'),
			item('a1', 'a', 'Alpha-child')
		];
		const tree = buildTree(items);
		expect(tree.map((n) => n.item.id)).toEqual(['a', 'b']);
		expect(tree[0].children.map((n) => n.item.id)).toEqual(['a1', 'c1']);
	});

	it('accepts a custom compare function overriding the default name sort', () => {
		const items = [item('a', null, 'Alpha'), item('b', null, 'Bravo')];
		const tree = buildTree(items, (x, y) => y.name.localeCompare(x.name));
		expect(tree.map((n) => n.item.id)).toEqual(['b', 'a']);
	});
});

describe('flattenTree', () => {
	it('walks parents before children, depth-first', () => {
		const items = [
			item('root', null, 'Root'),
			item('child-b', 'root', 'B'),
			item('child-a', 'root', 'A'),
			item('grandchild', 'child-a', 'GC')
		];
		const tree = buildTree(items);
		const flat = flattenTree(tree);
		expect(flat.map((n) => n.item.id)).toEqual(['root', 'child-a', 'grandchild', 'child-b']);
	});
});

describe('descendantIds', () => {
	it('includes the id itself plus every transitive descendant', () => {
		const items = [
			item('root', null, 'Root'),
			item('child', 'root', 'Child'),
			item('grandchild', 'child', 'Grandchild'),
			item('sibling', null, 'Sibling')
		];
		expect(descendantIds(items, 'root')).toEqual(new Set(['root', 'child', 'grandchild']));
	});

	it('terminates instead of looping on a malformed parent_id cycle', () => {
		const items = [item('a', 'b', 'A'), item('b', 'a', 'B')];
		expect(descendantIds(items, 'a')).toEqual(new Set(['a', 'b']));
	});
});

describe('filterTree', () => {
	const items = [
		item('films', null, 'Short films'),
		item('alchemist', 'films', 'The alchemist'),
		item('opening', 'alchemist', 'Opening scene'),
		item('workshop', 'alchemist', 'The workshop'),
		item('characters', null, 'Characters'),
		item('portraits', 'characters', 'Portraits')
	];
	const ids = (nodes: ReturnType<typeof buildTree<Item>>) => flattenTree(nodes).map((n) => n.item.id);

	it('keeps a nested match with its ancestors and drops unrelated branches', () => {
		const kept = filterTree(buildTree(items), (i) => i.name.includes('Opening'));
		expect(ids(kept)).toEqual(['films', 'alchemist', 'opening']);
		expect(kept[0].children[0].children.map((n) => n.depth)).toEqual([2]);
	});

	it('hides non-matching children of a match', () => {
		const kept = filterTree(buildTree(items), (i) => i.name === 'Characters');
		expect(ids(kept)).toEqual(['characters']);
		expect(kept[0].children).toEqual([]);
	});

	it('returns nothing when no item matches', () => {
		expect(filterTree(buildTree(items), () => false)).toEqual([]);
	});
});

describe('ancestorsOf', () => {
	it('lists the parents root first', () => {
		const items = [item('a', null, 'A'), item('b', 'a', 'B'), item('c', 'b', 'C')];
		expect(ancestorsOf(items, 'c').map((i) => i.id)).toEqual(['a', 'b']);
		expect(ancestorsOf(items, 'a')).toEqual([]);
	});

	it('stops on a malformed parent_id cycle', () => {
		const items = [item('a', 'b', 'A'), item('b', 'a', 'B')];
		expect(ancestorsOf(items, 'a').map((i) => i.id)).toEqual(['b']);
	});
});
