import { attachPointerDrag, tryCapturePointer, tryReleasePointer } from '../timelineCore';

export function laneFractionFromClientX(lane: Element | undefined, clientX: number): number | null {
	if (!lane) return null;
	const rect = lane.getBoundingClientRect();
	if (rect.width <= 0) return null;
	return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
}

export function isOverSelector(e: Event, selector: string): boolean {
	return (e.target as HTMLElement).closest(selector) != null;
}

export function insertFraction(e: MouseEvent, lane: Element | undefined, markSelector: string): number | null {
	if (isOverSelector(e, markSelector)) return null;
	return laneFractionFromClientX(lane, e.clientX);
}

export interface LaneDragHandlers {
	onMove: (fraction: number) => void;
	onCommit: () => void;
	onCancel: () => void;
}

export function beginLaneDrag(e: PointerEvent, lane: Element | undefined, handlers: LaneDragHandlers): boolean {
	e.stopPropagation();
	e.preventDefault();
	if (!lane) return false;
	const target = e.currentTarget as Element;
	const pointerId = e.pointerId;
	tryCapturePointer(target, pointerId);
	attachPointerDrag(
		(move) => {
			const fraction = laneFractionFromClientX(lane, move.clientX);
			if (fraction != null) handlers.onMove(fraction);
		},
		() => {
			tryReleasePointer(target, pointerId);
			handlers.onCommit();
		},
		() => {
			tryReleasePointer(target, pointerId);
			handlers.onCancel();
		}
	);
	return true;
}
