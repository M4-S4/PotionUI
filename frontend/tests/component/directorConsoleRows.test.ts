// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import type { ConsoleShot, ConsoleJoin } from '../../src/lib/components/video-director/console/consoleModel';

const { default: ShotRow } = await import('../../src/lib/components/video-director/console/ShotRow.svelte');
const { default: JoinConnector } = await import('../../src/lib/components/video-director/console/JoinConnector.svelte');
const { createClassComponent } = await import('svelte/legacy');

function shot(overrides: Partial<ConsoleShot> = {}): ConsoleShot {
	return {
		id: 'shot-1',
		index: 0,
		number: '01',
		title: 'Artisan at work',
		durationSeconds: 3.5,
		startSeconds: 0,
		frames: 84,
		capFrames: 84,
		newFrames: null,
		fps: 24,
		fpsLocked: false,
		timingQualified: true,
		thumb: { url: null, source: 'slate' },
		badge: 'independent',
		hasIcLora: false,
		icLoraCount: 0,
		run: null,
		tabs: [{ id: 'selection', label: 'Selection' }],
		canRemove: true,
		canDuplicate: true,
		...overrides
	};
}

function join(overrides: Partial<ConsoleJoin> = {}): ConsoleJoin {
	return {
		afterShotId: 'shot-1',
		beforeShotId: 'shot-1',
		kind: 'cut',
		label: 'Hard cut',
		sentence: 'Starts fresh — nothing shared.',
		control: { kind: 'chip', text: 'Independent' },
		overlapFrames: null,
		...overrides
	};
}

function mount<T extends Record<string, unknown>>(component: unknown, props: T) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({ component: component as never, target, props });
	return {
		target,
		destroy: () => {
			instance.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mount> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
});

describe('ShotRow', () => {
	it('renders the shot number, an empty-thumb placeholder, and toggles checked on click', () => {
		let toggled: string | null = null;
		mounted = mount(ShotRow, {
			shot: shot(),
			checked: false,
			onToggleChecked: (id: string) => (toggled = id),
			onActivate: () => {}
		});

		expect(mounted.target.textContent).toContain('01');
		expect(mounted.target.textContent).toContain('Artisan at work');
		// no thumb url -> the dashed empty-slate variant, not a background-image well
		const thumb = mounted.target.querySelector('[role="button"][aria-label="Expand Artisan at work"]') as HTMLElement;
		expect(thumb.style.backgroundImage).toBe('');

		const checkbox = mounted.target.querySelector('button[aria-label="Select for generation"]') as HTMLButtonElement;
		expect(checkbox).toBeTruthy();
		checkbox.click();
		expect(toggled).toBe('shot-1');
	});

	it('renders a thumb background-image when the shot has one', () => {
		mounted = mount(ShotRow, {
			shot: shot({ thumb: { url: 'blob:abc', source: 'keyframe' } }),
			checked: false,
			onToggleChecked: () => {},
			onActivate: () => {}
		});
		const thumb = mounted.target.querySelector('[role="button"] > .bg-cover') as HTMLElement;
		expect(thumb.style.backgroundImage).toContain('blob:abc');
	});

	it.each([
		['independent', 'Independent'],
		['needs-previous', 'Needs previous shot'],
		['input-ready', 'Input ready'],
		['stale', 'Stale dependency'],
		['continuous', 'Part of continuous render']
	] as const)('renders the %s dependency badge as %s', (badge, label) => {
		mounted = mount(ShotRow, { shot: shot({ badge }), checked: false, onToggleChecked: () => {}, onActivate: () => {} });
		expect(mounted.target.textContent).toContain(label);
	});

	it('shows the Selected chip only once checked, and a checked checkbox renders the checkmark icon', () => {
		mounted = mount(ShotRow, { shot: shot(), checked: true, onToggleChecked: () => {}, onActivate: () => {} });
		expect(mounted.target.textContent).toContain('Selected');
		const checkbox = mounted.target.querySelector('button[aria-label="Selected for generation"]') as HTMLButtonElement;
		expect(checkbox.querySelector('svg')).toBeTruthy();
	});

	// DIR-06 rework: a Wan shot with no known timing profile shows its raw
	// duration with a "requested" qualifier instead of a falsely-precise
	// number; every qualified shot (every family besides Wan, and Wan once
	// the profile is known) renders no such marker.
	it('renders a "requested" qualifier when timingQualified is false, and none when true', () => {
		mounted = mount(ShotRow, { shot: shot({ timingQualified: false }), checked: false, onToggleChecked: () => {}, onActivate: () => {} });
		expect(mounted.target.textContent).toContain('requested');

		mounted.destroy();
		mounted = mount(ShotRow, { shot: shot({ timingQualified: true }), checked: false, onToggleChecked: () => {}, onActivate: () => {} });
		expect(mounted.target.textContent).not.toContain('requested');
	});
});

describe('JoinConnector', () => {
	it('renders a chip control read-only, with no toggle buttons', () => {
		mounted = mount(JoinConnector, { join: join(), onSetJoin: () => {} });
		expect(mounted.target.textContent).toContain('Independent');
		expect(mounted.target.querySelector('button')).toBeNull();
	});

	it('unfolds join settings on a continuing join, clamps overlap to the cap and emits the new value', async () => {
		let received: number | null = null;
		mounted = mount(JoinConnector, {
			join: join({ kind: 'continue', overlapFrames: 8, control: { kind: 'toggle', value: 'continue' } }),
			onSetJoin: () => {},
			onSetOverlap: (frames: number) => (received = frames),
			maxOverlapFrames: 34
		});
		expect(mounted.target.querySelector('input[aria-label="Overlap frames"]')).toBeNull();
		(mounted.target.querySelector('button[aria-label="Show join settings"]') as HTMLElement).click();
		await tick();
		const input = mounted.target.querySelector('input[aria-label="Overlap frames"]') as HTMLInputElement;
		expect(input.value).toBe('8');
		input.value = '99';
		input.dispatchEvent(new Event('change', { bubbles: true }));
		expect(received).toBe(34);
	});

	it('renders no join settings disclosure on a fresh cut', () => {
		mounted = mount(JoinConnector, {
			join: join({ kind: 'cut', overlapFrames: null, control: { kind: 'toggle', value: 'cut' } }),
			onSetJoin: () => {}
		});
		expect(mounted.target.querySelector('button[aria-label="Show join settings"]')).toBeNull();
	});

	it('emits the toggled value when the inactive side of a Continue/Fresh cut toggle is clicked', () => {
		let call: [string, string] | null = null;
		mounted = mount(JoinConnector, {
			join: join({
				afterShotId: 'shot-2',
				kind: 'native',
				label: 'Native continuation',
				control: { kind: 'toggle', value: 'continue' }
			}),
			onSetJoin: (afterShotId: string, value: 'continue' | 'cut') => (call = [afterShotId, value])
		});

		const buttons = Array.from(mounted.target.querySelectorAll('button'));
		const freshCut = buttons.find((b) => b.textContent?.includes('Fresh cut'));
		expect(freshCut).toBeTruthy();
		freshCut!.click();
		expect(call).toEqual(['shot-2', 'cut']);

		// bite-check: the still-active "Continue" side must be a no-op-looking
		// click too, not silently doing nothing structurally different
		const continueBtn = buttons.find((b) => b.textContent?.includes('Continue'));
		continueBtn!.click();
		expect(call).toEqual(['shot-2', 'continue']);
	});

	it('renders the missing-predecessor warning block with its two actions (W3: spans the contiguous run back to the nearest fresh cut)', () => {
		let generated: string[] | null = null;
		let converted: string | null = null;
		mounted = mount(JoinConnector, {
			join: join({
				afterShotId: 'shot-0',
				beforeShotId: 'shot-1',
				kind: 'missing',
				label: 'Missing predecessor',
				sentence: 'Shot 01 has no output yet.',
				control: { kind: 'missing', spanShotIds: ['shot-0', 'shot-1'] }
			}),
			onSetJoin: () => {},
			onGeneratePreviousAndThis: (spanShotIds: string[]) => (generated = spanShotIds),
			onConvertToFreshCut: (id: string) => (converted = id)
		});
		expect(mounted.target.textContent).toContain('Missing predecessor');
		const buttons = Array.from(mounted.target.querySelectorAll('button'));
		buttons.find((b) => b.textContent?.includes('Generate previous'))!.click();
		buttons.find((b) => b.textContent?.includes('Convert to fresh cut'))!.click();
		expect(generated).toEqual(['shot-0', 'shot-1']);
		expect(converted).toBe('shot-0');
	});
});
