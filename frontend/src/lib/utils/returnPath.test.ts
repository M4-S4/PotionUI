import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
	consumeReturnPath,
	loginUrlWithReturn,
	returnPathFromUrl,
	sanitizeReturnPath,
	saveReturnPath
} from './returnPath';

describe('sanitizeReturnPath', () => {
	it('keeps a plain app path with query and hash', () => {
		expect(sanitizeReturnPath('/library?tab=models#top')).toBe('/library?tab=models#top');
	});

	it.each([
		'//evil.example',
		'/\\evil.example',
		'\\\\evil.example',
		'https://evil.example/x',
		'http://evil.example',
		'javascript:alert(1)',
		'evil.example',
		'',
		'/%2f%2fevil.example',
		'/%5cevil.example',
		'/a%0d%0ab',
		'/a\nb',
		'/\tfoo',
		'/%zz',
		'/.//evil.com',
		'/..//evil.com',
		'/a/..//evil.com',
		'/%2e//evil.com',
		'/%2e%2e//evil.com',
		'/%2E%2E%2F%2Fevil.com',
		'/%252e//evil.com',
		'/./x',
		'/a/../x',
		'/a/.'
	])('rejects %j', (raw) => {
		expect(sanitizeReturnPath(raw)).toBeNull();
	});

	it('keeps dots that are not whole segments', () => {
		expect(sanitizeReturnPath('/history?generation=abc')).toBe('/history?generation=abc');
		expect(sanitizeReturnPath('/a.b/..c/d...')).toBe('/a.b/..c/d...');
		expect(sanitizeReturnPath('/x?next=/../y')).toBe('/x?next=/../y');
	});

	it('rejects non-strings', () => {
		expect(sanitizeReturnPath(null)).toBeNull();
		expect(sanitizeReturnPath(undefined)).toBeNull();
		expect(sanitizeReturnPath(42)).toBeNull();
	});

	it('rejects overlong values', () => {
		expect(sanitizeReturnPath(`/${'a'.repeat(3000)}`)).toBeNull();
	});

	it('rejects auth pages so a return cannot loop', () => {
		expect(sanitizeReturnPath('/login')).toBeNull();
		expect(sanitizeReturnPath('/login?next=/x')).toBeNull();
		expect(sanitizeReturnPath('/LOGIN/callback?code=1')).toBeNull();
		expect(sanitizeReturnPath('/register')).toBeNull();
	});
});

describe('loginUrlWithReturn', () => {
	it('encodes the current path as next', () => {
		const url = new URL('http://localhost/history?id=7');
		expect(loginUrlWithReturn(url)).toBe('/login?next=%2Fhistory%3Fid%3D7');
	});

	it('is the bare login page for the root', () => {
		expect(loginUrlWithReturn(new URL('http://localhost/'))).toBe('/login');
	});

	it('is the bare login page from an auth page', () => {
		expect(loginUrlWithReturn(new URL('http://localhost/register'))).toBe('/login');
	});
});

describe('returnPathFromUrl', () => {
	it('reads a safe next', () => {
		expect(returnPathFromUrl(new URL('http://localhost/login?next=%2Fhistory%3Fid%3D7'))).toBe(
			'/history?id=7'
		);
	});

	it('drops an unsafe next', () => {
		expect(returnPathFromUrl(new URL('http://localhost/login?next=%2F%2Fevil.example'))).toBeNull();
		expect(returnPathFromUrl(new URL('http://localhost/login?next=https%3A%2F%2Fevil'))).toBeNull();
	});
});

describe('save and consume', () => {
	beforeEach(() => {
		const store = new Map<string, string>();
		vi.stubGlobal('sessionStorage', {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => void store.set(key, value),
			removeItem: (key: string) => void store.delete(key)
		});
	});

	it('round-trips once', () => {
		saveReturnPath('/history?id=7');
		expect(consumeReturnPath()).toBe('/history?id=7');
		expect(consumeReturnPath()).toBe('/');
	});

	it('never stores an unsafe path', () => {
		saveReturnPath('//evil.example');
		expect(consumeReturnPath()).toBe('/');
	});

	it('re-validates a tampered stored value on use', () => {
		sessionStorage.setItem('post_login_path', '//evil.example');
		expect(consumeReturnPath()).toBe('/');
	});

	it('saving null clears a stale value', () => {
		saveReturnPath('/a');
		saveReturnPath(null);
		expect(consumeReturnPath()).toBe('/');
	});
});
