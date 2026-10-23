import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
	clearSchemaCache,
	createLatestRequestGuard,
	getCachedSchema,
	invalidateCachedSchema,
	schemaCacheKey
} from './schemaCache';

describe('schema cache', () => {
	beforeEach(clearSchemaCache);

	it('deduplicates in-flight requests by preset and mode', async () => {
		let resolve!: (value: object) => void;
		const loader = vi.fn(() => new Promise<object>((done) => (resolve = done)));

		const first = getCachedSchema('preset', 'txt2img', loader);
		const second = getCachedSchema('preset', 'txt2img', loader);
		expect(loader).toHaveBeenCalledTimes(1);

		resolve({ properties: {} });
		expect(await first).toBe(await second);
	});

	it('keeps modes separate and supports explicit invalidation', async () => {
		const loader = vi.fn(async () => ({ call: loader.mock.calls.length }));
		await getCachedSchema('preset', 'txt2img', loader);
		await getCachedSchema('preset', 'img2img', loader);
		invalidateCachedSchema('preset', 'txt2img');
		await getCachedSchema('preset', 'txt2img', loader);
		expect(loader).toHaveBeenCalledTimes(3);
	});

	it('does not cache failures', async () => {
		const loader = vi
			.fn<() => Promise<object>>()
			.mockRejectedValueOnce(new Error('network'))
			.mockResolvedValueOnce({ properties: {} });

		await expect(getCachedSchema('preset', 'txt2img', loader)).rejects.toThrow('network');
		await expect(getCachedSchema('preset', 'txt2img', loader)).resolves.toEqual({ properties: {} });
		expect(loader).toHaveBeenCalledTimes(2);
	});
});

describe('latest request guard', () => {
	it('marks older responses as stale', () => {
		const guard = createLatestRequestGuard();
		const first = guard.next();
		const second = guard.next();

		expect(guard.isCurrent(first)).toBe(false);
		expect(guard.isCurrent(second)).toBe(true);
		guard.invalidate();
		expect(guard.isCurrent(second)).toBe(false);
	});
});

describe('schemaCacheKey', () => {
	it('separates preset, mode and form name with NUL', () => {
		expect(schemaCacheKey('p', 'm', 'f')).toBe('p\u0000m\u0000f');
		expect(schemaCacheKey('p', 'm')).toBe('p\u0000m\u0000');
	});

	it('keeps the source file free of raw NUL bytes', () => {
		const source = readFileSync(new URL('./schemaCache.ts', import.meta.url));
		expect(source.includes(0)).toBe(false);
	});
});
