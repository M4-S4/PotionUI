import type { AxiosInstance } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createGenerationsApi } from './generations';

const client = { get: vi.fn() } as unknown as AxiosInstance;
const generations = createGenerationsApi(client);

describe('getGenerationById', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		vi.mocked(client.get).mockResolvedValue({ data: { success: true } });
	});

	it('requests the id as a single path segment', async () => {
		await generations.getGenerationById('01ABC');

		expect(client.get).toHaveBeenCalledWith('/api/generations/history/01ABC');
	});

	it('percent-encodes an id that tries to leave the path', async () => {
		await generations.getGenerationById('../../x?a=b#c', true);

		expect(client.get).toHaveBeenCalledWith(
			'/api/generations/history/..%2F..%2Fx%3Fa%3Db%23c?include_tags=true'
		);
	});
});
