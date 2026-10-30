import { describe, expect, it } from 'vitest';
import { hrefWithLibraryItem, libraryItemIdFromUrl } from './deepLink';
import { removePendingFresh, setPendingFresh, takePendingFresh } from './freshHandoff';

describe('library item deep link', () => {
	it('reads the item id', () => {
		expect(libraryItemIdFromUrl(new URL('http://x/library?item=abc'))).toBe('abc');
	});

	it('ignores a missing, empty or unsafe id', () => {
		expect(libraryItemIdFromUrl(new URL('http://x/library'))).toBeNull();
		expect(libraryItemIdFromUrl(new URL('http://x/library?item='))).toBeNull();
		expect(libraryItemIdFromUrl(new URL('http://x/library?item=..%2Fa'))).toBeNull();
		expect(libraryItemIdFromUrl(new URL('http://x/library?item=a%2Fb'))).toBeNull();
	});

	it('writes and clears the param, keeping other params and the hash', () => {
		const url = new URL('http://x/library?media=image#top');
		expect(hrefWithLibraryItem(url, 'abc')).toBe('/library?media=image&item=abc#top');
		expect(hrefWithLibraryItem(new URL('http://x/library?item=abc&media=image'), null)).toBe('/library?media=image');
		expect(hrefWithLibraryItem(new URL('http://x/library?item=abc'), null)).toBe('/library');
	});
});

describe('fresh handoff', () => {
	it('drops one id when its item was opened', () => {
		setPendingFresh(['a', 'b']);
		removePendingFresh('a');
		expect(takePendingFresh()).toEqual(['b']);
	});

	it('hands the ids over once', () => {
		setPendingFresh(['a', 'b']);
		expect(takePendingFresh()).toEqual(['a', 'b']);
		expect(takePendingFresh()).toEqual([]);
	});
});
