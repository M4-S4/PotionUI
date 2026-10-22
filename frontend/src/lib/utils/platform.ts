export interface PlatformSource {
	userAgentData?: { platform?: string };
	platform?: string;
	userAgent?: string;
}

const APPLE_PLATFORM = /mac|iphone|ipad|ipod/i;

function currentSource(): PlatformSource {
	return typeof navigator === 'undefined' ? {} : navigator;
}

export function isMacPlatform(source: PlatformSource = currentSource()): boolean {
	const hint = source.userAgentData?.platform || source.platform || source.userAgent || '';
	return APPLE_PLATFORM.test(hint);
}

export function primaryModifier(source: PlatformSource = currentSource()): string {
	return isMacPlatform(source) ? '⌘' : 'Ctrl';
}
