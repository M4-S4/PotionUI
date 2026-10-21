import { mount, unmount } from 'svelte';
import TooltipBubble from '$lib/components/TooltipBubble.svelte';

export interface TooltipOptions {
	text?: string | null;
	position?: 'top' | 'bottom' | 'left' | 'right';
	delay?: number;
	kbd?: string;
	truncateOnly?: boolean;
}

export type TooltipParam = string | null | undefined | TooltipOptions;

const OVERFLOW_SELECTOR = '.truncate, [class*="line-clamp"], [class*="text-ellipsis"]';

let sequence = 0;

function normalize(param: TooltipParam): TooltipOptions {
	return typeof param === 'string' || param == null ? { text: param } : param;
}

export function isOverflowing(node: HTMLElement): boolean {
	const cut = (el: Element) => el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight;
	return cut(node) || Array.from(node.querySelectorAll(OVERFLOW_SELECTOR)).some(cut);
}

export default function tooltip(node: HTMLElement, param: TooltipParam) {
	let options = normalize(param);
	let timer: ReturnType<typeof setTimeout> | undefined;
	let bubble: Record<string, unknown> | undefined;
	let pending = false;
	let previousDescribedBy: string | null = null;
	const id = `tooltip-${++sequence}`;

	function hide() {
		clearTimeout(timer);
		timer = undefined;
		pending = false;
		if (!bubble) return;
		unmount(bubble);
		bubble = undefined;
		if (previousDescribedBy === null) node.removeAttribute('aria-describedby');
		else node.setAttribute('aria-describedby', previousDescribedBy);
	}

	function mountBubble() {
		if (!pending) return;
		pending = false;
		if (bubble || !options.text || !node.isConnected) return;
		previousDescribedBy = node.getAttribute('aria-describedby');
		node.setAttribute('aria-describedby', id);
		bubble = mount(TooltipBubble, {
			target: document.body,
			props: {
				text: options.text,
				anchor: node,
				position: options.position,
				kbd: options.kbd,
				id
			}
		});
	}

	function reveal() {
		if (bubble || pending || !options.text) return;
		if (options.truncateOnly && !isOverflowing(node)) return;
		pending = true;
		queueMicrotask(mountBubble);
	}

	function schedule(delay: number) {
		clearTimeout(timer);
		if (delay <= 0) reveal();
		else timer = setTimeout(reveal, delay);
	}

	const onEnter = () => schedule(options.delay ?? 200);
	const onFocus = () => schedule(0);
	const onKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') hide();
	};

	node.addEventListener('mouseenter', onEnter);
	node.addEventListener('mouseleave', hide);
	node.addEventListener('focusin', onFocus);
	node.addEventListener('focusout', hide);
	node.addEventListener('keydown', onKeydown);

	return {
		update(next: TooltipParam) {
			const wasVisible = bubble !== undefined;
			hide();
			options = normalize(next);
			if (wasVisible) reveal();
		},
		destroy() {
			hide();
			node.removeEventListener('mouseenter', onEnter);
			node.removeEventListener('mouseleave', hide);
			node.removeEventListener('focusin', onFocus);
			node.removeEventListener('focusout', hide);
			node.removeEventListener('keydown', onKeydown);
		}
	};
}
