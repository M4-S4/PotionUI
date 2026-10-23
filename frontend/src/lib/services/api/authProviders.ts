import type { AxiosInstance } from 'axios';

export interface LoginProvider {
	id: string;
	label: string;
	start_path: string;
	auto_redirect?: boolean;
}

export interface ExternalLoginToken {
	access_token: string;
	token_type: string;
}

export function createAuthProvidersApi(client: AxiosInstance) {
	return {
		async getLoginProviders(): Promise<LoginProvider[]> {
			const response = await client.get('/api/auth/providers');
			return response.data;
		},

		async exchangeExternalLoginCode(code: string, rememberMe: boolean = false): Promise<ExternalLoginToken> {
			const response = await client.post('/api/auth/external/exchange', { code, remember_me: rememberMe });
			return response.data;
		}
	};
}
