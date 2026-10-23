import { describe, expect, it } from 'vitest';
import { generationIdFromUrl, hrefWithGeneration } from './deepLink';
import { loginUrlWithReturn, returnPathFromUrl } from './returnPath';

const at = (path: string) => new URL(path, 'http://localhost');

describe('generationIdFromUrl', () => {
	it('reads the generation param', () => {
		expect(generationIdFromUrl(at('/history?generation=01ABC'))).toBe('01ABC');
	});

	it('is null when the param is absent or blank', () => {
		expect(generationIdFromUrl(at('/history'))).toBeNull();
		expect(generationIdFromUrl(at('/history?generation='))).toBeNull();
		expect(generationIdFromUrl(at('/history?generation=%20'))).toBeNull();
	});
});

describe('generationIdFromUrl unsafe ids', () => {
	it.each(['../../x', 'a/b', 'a\\b', 'a%3Fb', 'a%23b', '..', 'x..y'])('rejects %s', (id) => {
		expect(generationIdFromUrl(at(`/history?generation=${id}`))).toBeNull();
	});
});

describe('hrefWithGeneration', () => {
	it('sets the param and keeps the other query params and hash', () => {
		expect(hrefWithGeneration(at('/history?sort=largest#top'), '01ABC')).toBe(
			'/history?sort=largest&generation=01ABC#top'
		);
	});

	it('replaces an existing id', () => {
		expect(hrefWithGeneration(at('/history?generation=old'), 'new')).toBe('/history?generation=new');
	});

	it('removes the param and leaves a bare path when nothing else remains', () => {
		expect(hrefWithGeneration(at('/history?generation=old'), null)).toBe('/history');
		expect(hrefWithGeneration(at('/history?sort=largest&generation=old'), null)).toBe('/history?sort=largest');
	});

	it('percent-encodes ids that need it', () => {
		expect(hrefWithGeneration(at('/history'), 'a b/c')).toBe('/history?generation=a+b%2Fc');
	});
});

describe('login return path', () => {
	it('keeps the generation param through the login redirect', () => {
		const login = loginUrlWithReturn(at('/history?generation=01ABC'));
		const next = returnPathFromUrl(new URL(login, 'http://localhost'));

		expect(next).toBe('/history?generation=01ABC');
		expect(generationIdFromUrl(new URL(next ?? '', 'http://localhost'))).toBe('01ABC');
	});
});
