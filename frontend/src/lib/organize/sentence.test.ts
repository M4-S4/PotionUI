import { describe, expect, it } from 'vitest';
import { awaitsTagging } from './sentence';
import type { OrganizeCatalog } from '$lib/types/organize';

const catalog = {
	facts: [
		{ key: 'tags', triggers: ['item_created', 'tags_changed'] },
		{ key: 'detected_tags', triggers: ['detected_tags_changed'] },
		{ key: 'legacy' }
	]
} as unknown as OrganizeCatalog;

describe('awaitsTagging', () => {
	it('is true when a condition uses a fact that is not known at creation', () => {
		expect(awaitsTagging([{ fact: 'tags' }, { fact: 'detected_tags' }], catalog)).toBe(true);
	});

	it('is false for facts known at creation, unknown facts, and no catalog', () => {
		expect(awaitsTagging([{ fact: 'tags' }, { fact: 'legacy' }, { fact: 'nope' }], catalog)).toBe(false);
		expect(awaitsTagging([{ fact: 'detected_tags' }], null)).toBe(false);
	});
});
