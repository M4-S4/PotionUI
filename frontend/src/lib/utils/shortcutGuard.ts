import { hasOpenDismissLayer, topDismissLayer } from '$lib/actions/layerStack';
import { isTypingInField } from '$lib/utils/editableTarget';

export function plainLetter(event: KeyboardEvent): string | null {
	if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return null;
	if (event.key.length !== 1 || !/[a-z]/i.test(event.key)) return null;
	return event.key.toLowerCase();
}

export function windowOwnsKeys(event: KeyboardEvent, root: Element | null | undefined): boolean {
	if (!root || isTypingInField(event)) return false;
	const top = topDismissLayer();
	return !!top && top.node.contains(root);
}

export function pageOwnsKeys(event: KeyboardEvent): boolean {
	return !isTypingInField(event) && !hasOpenDismissLayer();
}
