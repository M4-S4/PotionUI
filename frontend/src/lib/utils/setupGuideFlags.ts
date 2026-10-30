import { browser } from '$app/environment';

export const WELCOME_SEEN_KEY = 'potionui:setup:welcomeSeen';
export const COMPUTER_SEEN_KEY = 'potionui:setup:computerSeen';
export const MODELS_LOCATION_DONE_KEY = 'potionui:setup:modelsLocationDismissed';

export function readGuideFlag(key: string): boolean {
	if (!browser) return false;
	try {
		return localStorage.getItem(key) === '1';
	} catch {
		return false;
	}
}

export function writeGuideFlag(key: string, value: boolean): void {
	if (!browser) return;
	try {
		if (value) {
			localStorage.setItem(key, '1');
		} else {
			localStorage.removeItem(key);
		}
	} catch {
		return;
	}
}
