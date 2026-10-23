import type { LoginProvider } from '$lib/services/api/index';

export const AUTO_REDIRECT_GUARD_KEY = 'sso_auto_redirect_at';
export const AUTO_REDIRECT_GUARD_MS = 30_000;

const SUPPRESSING_PARAMS = ['local', 'error', 'expired', 'signedout', 'add', 'dup', 'as'];

export function suppressesAutoRedirect(params: URLSearchParams): boolean {
	return SUPPRESSING_PARAMS.some((name) => params.has(name));
}

export function pickAutoRedirectProvider(providers: LoginProvider[]): LoginProvider | null {
	return providers.find((provider) => provider.auto_redirect === true) ?? null;
}

export function shouldAutoRedirect(input: {
	providers: LoginProvider[];
	params: URLSearchParams;
	lastAttemptAt: number | null;
	now: number;
}): LoginProvider | null {
	const provider = pickAutoRedirectProvider(input.providers);
	if (!provider) return null;
	if (suppressesAutoRedirect(input.params)) return null;
	const { lastAttemptAt, now } = input;
	if (lastAttemptAt !== null && now - lastAttemptAt >= 0 && now - lastAttemptAt < AUTO_REDIRECT_GUARD_MS) {
		return null;
	}
	return provider;
}

export function readAutoRedirectAttempt(): number | null {
	try {
		const raw = sessionStorage.getItem(AUTO_REDIRECT_GUARD_KEY);
		const value = raw === null ? NaN : Number(raw);
		return Number.isFinite(value) ? value : null;
	} catch {
		return null;
	}
}

export function markAutoRedirectAttempt(now: number): void {
	try {
		sessionStorage.setItem(AUTO_REDIRECT_GUARD_KEY, String(now));
	} catch {}
}
