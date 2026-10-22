export const CHIP_POPOVER_WIDTH = 420;
export const CHIP_POPOVER_SHEET_BREAKPOINT = 767;
export const CHIP_POPOVER_VIEWPORT_GUTTER = 8;

export function isChipPopoverSheet(viewportWidth: number): boolean {
	return viewportWidth <= CHIP_POPOVER_SHEET_BREAKPOINT;
}
