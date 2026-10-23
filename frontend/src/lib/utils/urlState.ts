import { goto } from '$app/navigation';

export interface UrlNavOptions {
	replace?: boolean;
	keepFocus?: boolean;
	noScroll?: boolean;
}

export interface UrlState<O> {
	go(overrides: O, options?: UrlNavOptions): Promise<void>;
	replace(overrides: O): Promise<void>;
	debounced(overrides: O, onFire?: () => void): void;
	cancel(): void;
}

export function hrefWithQuery(pathname: string, params: URLSearchParams): string {
	const query = params.toString();
	return query ? `${pathname}?${query}` : pathname;
}

export function createUrlState<O>(config: { href: (overrides: O) => string; delay?: number }): UrlState<O> {
	const delay = config.delay ?? 250;
	let timer: ReturnType<typeof setTimeout> | undefined;

	function go(overrides: O, options: UrlNavOptions = {}): Promise<void> {
		const gotoOptions: Parameters<typeof goto>[1] = {};
		if (options.replace) gotoOptions.replaceState = true;
		if (options.keepFocus) gotoOptions.keepFocus = true;
		if (options.noScroll) gotoOptions.noScroll = true;
		return goto(config.href(overrides), gotoOptions);
	}

	function replace(overrides: O): Promise<void> {
		return go(overrides, { replace: true, keepFocus: true, noScroll: true });
	}

	function cancel() {
		clearTimeout(timer);
		timer = undefined;
	}

	function debounced(overrides: O, onFire?: () => void) {
		clearTimeout(timer);
		timer = setTimeout(() => {
			timer = undefined;
			onFire?.();
			void replace(overrides);
		}, delay);
	}

	return { go, replace, debounced, cancel };
}
