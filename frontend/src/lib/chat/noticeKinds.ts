export interface NoticeAppearance {
	icon: string;
	tone: string;
}

export const NOTICE_APPEARANCE = {
	retrying: { icon: 'clock', tone: 'text-warning' },
	degraded_retry: { icon: 'refresh', tone: 'text-warning' },
	reasoning_dropped: { icon: 'info', tone: 'text-info' },
	tool_choice_downgraded: { icon: 'adjustments-horizontal', tone: 'text-info' },
	structured_fallback: { icon: 'braces', tone: 'text-warning' },
	structured_repaired: { icon: 'check-circle', tone: 'text-info' },
	approval_expired: { icon: 'hourglass', tone: 'text-fg-subtle' }
} as const satisfies Record<string, NoticeAppearance>;

export type NoticeKind = keyof typeof NOTICE_APPEARANCE;

const FALLBACK: NoticeAppearance = { icon: 'info', tone: 'text-fg-subtle' };

export function noticeAppearance(kind: string): NoticeAppearance {
	return Object.prototype.hasOwnProperty.call(NOTICE_APPEARANCE, kind)
		? (NOTICE_APPEARANCE as Record<string, NoticeAppearance>)[kind]
		: FALLBACK;
}
