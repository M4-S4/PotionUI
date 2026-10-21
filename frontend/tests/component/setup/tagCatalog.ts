import { beforeEach } from 'vitest';

beforeEach(async () => {
	const { tagCatalog } = await import('../../../src/lib/stores/tagCatalog');
	tagCatalog.reset();
});
