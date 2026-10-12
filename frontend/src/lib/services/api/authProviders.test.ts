import { describe, it, expect, vi } from 'vitest';
import type { AxiosInstance } from 'axios';
import { createAuthProvidersApi } from './authProviders';

function clientReturning(data: unknown) {
	const post = vi.fn(async () => ({ data }));
	return { client: { post } as unknown as AxiosInstance, post };
}

describe('exchangeExternalLoginCode', () => {
	it('sends the stay-signed-in choice with the handoff code', async () => {
		const { client, post } = clientReturning({ access_token: 'jwt', token_type: 'bearer' });

		await createAuthProvidersApi(client).exchangeExternalLoginCode('code-1', true);

		expect(post).toHaveBeenCalledWith('/api/auth/external/exchange', { code: 'code-1', remember_me: true });
	});

	it('defaults to a session that is not remembered', async () => {
		const { client, post } = clientReturning({ access_token: 'jwt', token_type: 'bearer' });

		await createAuthProvidersApi(client).exchangeExternalLoginCode('code-1');

		expect(post).toHaveBeenCalledWith('/api/auth/external/exchange', { code: 'code-1', remember_me: false });
	});
});
