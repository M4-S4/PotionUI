import { describe, expect, it } from 'vitest';
import { getIconPath } from '$lib/utils/IconLibrary';
import { NOTICE_APPEARANCE, noticeAppearance } from './noticeKinds';

describe('noticeAppearance', () => {
	it('knows every kind the harness raises', () => {
		expect(Object.keys(NOTICE_APPEARANCE).sort()).toEqual([
			'approval_expired',
			'degraded_retry',
			'reasoning_dropped',
			'retrying',
			'structured_fallback',
			'structured_repaired',
			'tool_choice_downgraded'
		]);
	});

	it('uses an icon the icon library has for every kind', () => {
		for (const [kind, look] of Object.entries(NOTICE_APPEARANCE)) {
			expect(getIconPath(look.icon), kind).toBeTruthy();
		}
		expect(getIconPath(noticeAppearance('something_new').icon)).toBeTruthy();
	});

	it('gives a degradation a warning tone and an explanation an info tone', () => {
		expect(noticeAppearance('degraded_retry').tone).toBe('text-warning');
		expect(noticeAppearance('structured_fallback').tone).toBe('text-warning');
		expect(noticeAppearance('tool_choice_downgraded').tone).toBe('text-info');
		expect(noticeAppearance('structured_repaired').tone).toBe('text-info');
	});

	it('falls back to a quiet generic look for a kind it does not know', () => {
		expect(noticeAppearance('something_new')).toEqual({ icon: 'info', tone: 'text-fg-subtle' });
	});

	it('does not mistake an inherited property for a kind', () => {
		expect(noticeAppearance('constructor')).toEqual({ icon: 'info', tone: 'text-fg-subtle' });
	});
});
