export function keyboardFocus(el: HTMLElement): void {
	if (!el.hasAttribute('tabindex') && !/^(button|a|input|select|textarea)$/i.test(el.tagName)) {
		el.setAttribute('tabindex', '-1');
	}
	const matches = el.matches.bind(el);
	el.matches = (selector: string) =>
		selector === ':focus-visible' ? document.activeElement === el : matches(selector);
	el.focus();
}

export function pointerFocus(el: HTMLElement): void {
	if (!el.hasAttribute('tabindex') && !/^(button|a|input|select|textarea)$/i.test(el.tagName)) {
		el.setAttribute('tabindex', '-1');
	}
	el.focus();
}
