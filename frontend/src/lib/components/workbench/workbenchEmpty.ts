export type WorkbenchEmptyLayout = 'pane' | 'fade' | 'window';

export const WORKBENCH_PANE_PADDING = 16;

export type WorkbenchEmptyState = 'empty' | 'failed' | 'suppressed' | 'blocked';

export interface WorkbenchEmptyClasses {
	root: string;
	stage: string;
	dots: string;
	copy: string;
	sized: boolean;
	reserve: boolean;
	slider: boolean;
}

export const WORKBENCH_EMPTY_LAYOUTS: Record<WorkbenchEmptyLayout, WorkbenchEmptyClasses> = {
	pane: {
		root: 'group/workbench [&:has(~*)]:min-h-[var(--workbench-stage)]',
		stage: 'absolute inset-x-0 top-[var(--workbench-stage-top,0px)] bottom-[var(--dock-height)] pointer-events-none',
		dots: 'absolute inset-0 dot-grid',
		copy: 'pointer-events-auto absolute inset-x-0 top-0 flex h-full flex-col items-center [justify-content:safe_center] overflow-y-auto px-8 text-center group-[:has(~*)]/workbench:h-[min(100%,var(--workbench-region))]',
		sized: false,
		reserve: true,
		slider: false
	},
	fade: {
		root: 'relative',
		stage: 'relative',
		dots: 'absolute -inset-x-4 -top-4 bottom-0 dot-grid dot-grid-fade pointer-events-none',
		copy: 'absolute inset-x-0 top-0 bottom-1/4 flex flex-col items-center [justify-content:safe_center] overflow-y-auto px-8 text-center',
		sized: true,
		reserve: false,
		slider: true
	},
	window: {
		root: 'relative',
		stage: 'relative',
		dots: 'absolute -inset-4 rounded-b-xl dot-grid pointer-events-none',
		copy: 'absolute inset-0 flex flex-col items-center [justify-content:safe_center] overflow-y-auto px-8 text-center',
		sized: true,
		reserve: false,
		slider: true
	}
};

export interface WorkbenchEmptyInput {
	hasDisplayMedia: boolean;
	status: string | null | undefined;
	previewSuppressed: boolean;
	contentBlocked: unknown;
}

export function workbenchEmptyState(input: WorkbenchEmptyInput): WorkbenchEmptyState | null {
	if (input.hasDisplayMedia) return null;
	if (input.status === 'failed') return 'failed';
	if (input.previewSuppressed) return 'suppressed';
	if (input.contentBlocked) return 'blocked';
	return 'empty';
}
