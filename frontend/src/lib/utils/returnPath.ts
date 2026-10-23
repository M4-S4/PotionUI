export const RETURN_PATH_KEY = 'post_login_path';
export const RETURN_PARAM = 'next';
export const DEFAULT_RETURN_PATH = '/';

const MAX_LENGTH = 2048;
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;
const DOT_SEGMENT = /(^|\/)(\.|%2e){1,2}(\/|$)/i;
const AUTH_PAGES = ['/login', '/register', '/setup/claim'];

function isSafeShape(value: string): boolean {
	if (!value.startsWith('/')) return false;
	if (value.startsWith('//')) return false;
	if (value.includes('\\')) return false;
	if (CONTROL_CHARS.test(value)) return false;
	return true;
}

function hasDotSegment(value: string): boolean {
	return DOT_SEGMENT.test(value.split(/[?#]/, 1)[0]);
}

function isAuthPage(value: string): boolean {
	const pathname = value.split(/[?#]/, 1)[0].toLowerCase();
	return AUTH_PAGES.some((page) => pathname === page || pathname.startsWith(`${page}/`));
}

export function sanitizeReturnPath(raw: unknown): string | null {
	if (typeof raw !== 'string') return null;
	const value = raw.trim();
	if (!value || value.length > MAX_LENGTH) return null;
	if (!isSafeShape(value)) return null;

	let decoded: string;
	try {
		decoded = decodeURIComponent(value);
	} catch {
		return null;
	}
	if (!isSafeShape(decoded)) return null;
	if (hasDotSegment(value) || hasDotSegment(decoded)) return null;

	let parsed: URL;
	try {
		parsed = new URL(value, 'http://return.invalid');
	} catch {
		return null;
	}
	if (parsed.origin !== 'http://return.invalid') return null;

	const normalized = `${parsed.pathname}${parsed.search}${parsed.hash}`;
	if (!isSafeShape(normalized)) return null;
	if (isAuthPage(normalized)) return null;
	return normalized;
}

export function returnPathFromUrl(url: URL): string | null {
	return sanitizeReturnPath(url.searchParams.get(RETURN_PARAM));
}

export function loginUrlWithReturn(url: URL): string {
	const target = sanitizeReturnPath(`${url.pathname}${url.search}${url.hash}`);
	if (!target || target === DEFAULT_RETURN_PATH) return '/login';
	return `/login?${RETURN_PARAM}=${encodeURIComponent(target)}`;
}

export function saveReturnPath(path: string | null): void {
	try {
		const safe = sanitizeReturnPath(path);
		if (safe) sessionStorage.setItem(RETURN_PATH_KEY, safe);
		else sessionStorage.removeItem(RETURN_PATH_KEY);
	} catch {}
}

export function consumeReturnPath(): string {
	try {
		const stored = sessionStorage.getItem(RETURN_PATH_KEY);
		sessionStorage.removeItem(RETURN_PATH_KEY);
		return sanitizeReturnPath(stored) ?? DEFAULT_RETURN_PATH;
	} catch {
		return DEFAULT_RETURN_PATH;
	}
}
