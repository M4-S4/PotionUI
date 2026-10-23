import { mount, unmount } from 'svelte';
import TooltipBubble from '$lib/components/TooltipBubble.svelte';

const attribute = 'data-tooltip';
const delay = 200;

let sequence = 0;

export default function delegatedTooltip(node: HTMLElement) {
	const id = `delegated-tooltip-${++sequence}`;
	let timer: ReturnType<typeof setTimeout> | undefined;
	let bubble: Record<string, unknown> | undefined;
	let current: HTMLElement | null = null;
	let previousDescribedBy: string | null = null;

	function hide() {
		clearTimeout(timer);
		timer = undefined;
		const anchor = current;
		current = null;
		if (!bubble || !anchor) return;
		unmount(bubble);
		bubble = undefined;
		if (previousDescribedBy === null) anchor.removeAttribute('aria-describedby');
		else anchor.setAttribute('aria-describedby', previousDescribedBy);
	}

	function show(anchor: HTMLElement) {
		const text = anchor.getAttribute(attribute);
		if (!text || !anchor.isConnected) return;
		previousDescribedBy = anchor.getAttribute('aria-describedby');
		anchor.setAttribute('aria-describedby', id);
		bubble = mount(TooltipBubble, { target: document.body, props: { text, anchor, id } });
	}

	function resolve(event: Event): HTMLElement | null {
		const target = event.target;
		if (!(target instanceof Element)) return null;
		const match = target.closest<HTMLElement>(`[${attribute}]`);
		return match && node.contains(match) ? match : null;
	}

	function schedule(anchor: HTMLElement, wait: number) {
		if (anchor === current) return;
		hide();
		current = anchor;
		if (wait <= 0) show(anchor);
		else
			timer = setTimeout(() => {
				if (current === anchor) show(anchor);
			}, wait);
	}

	const onOver = (e: MouseEvent) => {
		const anchor = resolve(e);
		if (anchor) schedule(anchor, delay);
	};
	const onOut = (e: MouseEvent) => {
		const anchor = resolve(e);
		if (!anchor) return;
		const next = e.relatedTarget;
		if (next instanceof Node && anchor.contains(next)) return;
		hide();
	};
	const onFocusIn = (e: FocusEvent) => {
		const anchor = resolve(e);
		if (anchor) schedule(anchor, 0);
	};
	const onFocusOut = (e: FocusEvent) => {
		if (resolve(e)) hide();
	};
	const onKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') hide();
	};

	const observer = new MutationObserver(() => {
		if (current && !current.isConnected) hide();
	});
	observer.observe(node, { childList: true, subtree: true });

	node.addEventListener('mouseover', onOver);
	node.addEventListener('mouseout', onOut);
	node.addEventListener('focusin', onFocusIn);
	node.addEventListener('focusout', onFocusOut);
	node.addEventListener('keydown', onKeydown);

	return {
		destroy() {
			hide();
			observer.disconnect();
			node.removeEventListener('mouseover', onOver);
			node.removeEventListener('mouseout', onOut);
			node.removeEventListener('focusin', onFocusIn);
			node.removeEventListener('focusout', onFocusOut);
			node.removeEventListener('keydown', onKeydown);
		}
	};
}
