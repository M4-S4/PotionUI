import { afterEach } from 'vitest';

afterEach(async () => {
	const { destroyMounted } = await import('../helpers/mount');
	destroyMounted();
});
