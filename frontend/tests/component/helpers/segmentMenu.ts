import { settle } from './settle';

export async function openSegmentMenuLabels(target: HTMLElement, triggerLabel: string): Promise<string[]> {
	target.querySelector<HTMLButtonElement>(`button[aria-label="${triggerLabel}"]`)!.click();
	await settle();
	return Array.from(document.querySelectorAll('[role="menuitem"]')).map(
		(item) => item.querySelector('span:last-child')?.textContent?.trim() ?? ''
	);
}
