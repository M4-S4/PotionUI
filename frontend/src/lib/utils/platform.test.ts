import { afterEach, describe, expect, it, vi } from 'vitest';
import { isMacPlatform, primaryModifier } from './platform';

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('isMacPlatform', () => {
	it('recognises Apple desktops and tablets from any of the three signals', () => {
		expect(isMacPlatform({ platform: 'MacIntel' })).toBe(true);
		expect(isMacPlatform({ userAgentData: { platform: 'macOS' } })).toBe(true);
		expect(isMacPlatform({ userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' })).toBe(true);
		expect(isMacPlatform({ userAgent: 'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)' })).toBe(true);
		expect(isMacPlatform({ platform: 'iPhone' })).toBe(true);
	});

	it('prefers the high-entropy platform over the legacy ones', () => {
		expect(isMacPlatform({ userAgentData: { platform: 'Windows' }, platform: 'MacIntel' })).toBe(false);
		expect(isMacPlatform({ userAgentData: { platform: '' }, platform: 'MacIntel' })).toBe(true);
	});

	it('is false on Windows, Linux, Android and when nothing is reported', () => {
		expect(isMacPlatform({ platform: 'Win32' })).toBe(false);
		expect(isMacPlatform({ platform: 'Linux x86_64' })).toBe(false);
		expect(isMacPlatform({ userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8)' })).toBe(false);
		expect(isMacPlatform({ userAgent: 'Mozilla/5.0 (X11; CrOS x86_64 14541.0.0)' })).toBe(false);
		expect(isMacPlatform({})).toBe(false);
	});

	it('reads the live navigator when no source is given', () => {
		vi.stubGlobal('navigator', { platform: 'MacIntel' });
		expect(isMacPlatform()).toBe(true);
		vi.stubGlobal('navigator', { platform: 'Win32' });
		expect(isMacPlatform()).toBe(false);
	});

	it('is false without a navigator', () => {
		vi.stubGlobal('navigator', undefined);
		expect(isMacPlatform()).toBe(false);
	});
});

describe('primaryModifier', () => {
	it('is the command glyph on Apple platforms and Ctrl everywhere else', () => {
		expect(primaryModifier({ platform: 'MacIntel' })).toBe('⌘');
		expect(primaryModifier({ platform: 'Win32' })).toBe('Ctrl');
		expect(primaryModifier({})).toBe('Ctrl');
	});

	it('follows the live navigator', () => {
		vi.stubGlobal('navigator', { platform: 'MacIntel' });
		expect(primaryModifier()).toBe('⌘');
	});
});
