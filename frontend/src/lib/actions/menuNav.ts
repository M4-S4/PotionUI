const DEFAULT_ITEMS = '[role="menuitem"]:not(:disabled), button:not(:disabled)';

function isRendered(item: HTMLElement, root: HTMLElement): boolean {
	for (let current: HTMLElement | null = item; current; current = current.parentElement) {
		if (getComputedStyle(current).display === 'none') return false;
		if (current === root) break;
	}
	return true;
}

export function menuItems(node: HTMLElement, selector: string = DEFAULT_ITEMS): HTMLElement[] {
	return Array.from(node.querySelectorAll<HTMLElement>(selector)).filter((item) => isRendered(item, node));
}

export function focusFirstItem(node: HTMLElement, selector: string = DEFAULT_ITEMS): void {
	menuItems(node, selector)[0]?.focus();
}

export default function menuNav(node: HTMLElement, selector: string = DEFAULT_ITEMS) {
	let itemSelector = selector;

	function onKeyDown(event: KeyboardEvent) {
		const items = menuItems(node, itemSelector);
		if (items.length === 0) return;
		const index = items.indexOf(document.activeElement as HTMLElement);
		let next: number;
		if (event.key === 'ArrowDown') next = (index + 1) % items.length;
		else if (event.key === 'ArrowUp') next = (index - 1 + items.length) % items.length;
		else if (event.key === 'Home') next = 0;
		else if (event.key === 'End') next = items.length - 1;
		else return;
		event.preventDefault();
		event.stopPropagation();
		items[next].focus();
	}

	node.addEventListener('keydown', onKeyDown);
	return {
		update(next: string = DEFAULT_ITEMS) {
			itemSelector = next;
		},
		destroy() {
			node.removeEventListener('keydown', onKeyDown);
		}
	};
}
