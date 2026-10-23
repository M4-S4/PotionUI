import { beforeEach } from 'vitest';

beforeEach(async () => {
	const { page } = await import('../stubs/appStores');
	page.update((current) => ({ ...current, url: new URL('http://localhost/generate') }));
});
