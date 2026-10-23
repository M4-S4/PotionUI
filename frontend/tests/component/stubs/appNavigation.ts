import { page } from './appStores';

export const goto = async (href: string | URL) => {
	page.update((current) => ({ ...current, url: new URL(href, 'http://localhost') }));
};
export const invalidate = async () => {};
export const invalidateAll = async () => {};
export const preloadData = async () => {};
export const preloadCode = async () => {};
export const afterNavigate = () => {};
export const beforeNavigate = () => {};
export const pushState = () => {};
export const replaceState = () => {};
