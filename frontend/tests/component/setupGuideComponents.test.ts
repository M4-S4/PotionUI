import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { machine, recipe, QWEN, SDXL, ZIMAGE } from './setupGuideFixtures';

const { default: ChoiceCard } = await import('../../src/lib/components/ui/ChoiceCard.svelte');
const { default: SetupStepper } = await import('../../src/lib/components/setup/SetupStepper.svelte');
const { default: StarterCard } = await import('../../src/lib/components/setup/StarterCard.svelte');
const { default: StarterStep } = await import('../../src/routes/setup/components/StarterStep.svelte');
const { default: MachineSummary } = await import('../../src/lib/components/setup/MachineSummary.svelte');
const { default: StepFacts } = await import('../../src/lib/components/setup/StepFacts.svelte');

let mounted: Record<string, any> | null = null;

function render(component: any, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	mounted = mount(component, { target, props });
	flushSync();
	return target;
}

afterEach(() => {
	if (mounted) unmount(mounted);
	mounted = null;
	document.body.innerHTML = '';
});

describe('ChoiceCard', () => {
	function group(onSelect: (i: number) => void, checkedIndex = 0) {
		const target = document.createElement('div');
		target.setAttribute('role', 'radiogroup');
		document.body.appendChild(target);
		const cards = ['One', 'Two', 'Three'].map((title, i) => {
			const holder = document.createElement('div');
			target.appendChild(holder);
			return mount(ChoiceCard, {
				target: holder,
				props: { checked: i === checkedIndex, title, onselect: () => onSelect(i) }
			});
		});
		flushSync();
		return { target, cards };
	}

	it('renders a radio with checked state, roving tabindex and a badge', () => {
		const target = render(ChoiceCard, {
			checked: true,
			title: 'Start fresh',
			badge: 'Recommended',
			description: 'Own folder',
			onselect: () => {}
		});
		const radio = target.querySelector('[role="radio"]') as HTMLElement;
		expect(radio.getAttribute('aria-checked')).toBe('true');
		expect(radio.tabIndex).toBe(0);
		expect(target.textContent).toContain('Recommended');
		expect(target.textContent).toContain('Own folder');
	});

	it('is out of the tab order when not the active choice', () => {
		const target = render(ChoiceCard, { checked: false, title: 'X', onselect: () => {} });
		expect((target.querySelector('[role="radio"]') as HTMLElement).tabIndex).toBe(-1);
	});

	it('selects on click', () => {
		const onselect = vi.fn();
		const target = render(ChoiceCard, { checked: false, title: 'X', onselect });
		(target.querySelector('[role="radio"]') as HTMLElement).click();
		expect(onselect).toHaveBeenCalledTimes(1);
	});

	it('moves selection with arrow keys, wrapping around', () => {
		const picked: number[] = [];
		const { target, cards } = group((i) => picked.push(i));
		const radios = Array.from(target.querySelectorAll<HTMLElement>('[role="radio"]'));
		radios[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
		radios[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
		radios[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
		expect(picked).toEqual([1, 0, 2]);
		cards.forEach((c) => unmount(c));
	});
});

describe('SetupStepper', () => {
	it('marks the current step and makes earlier steps clickable', () => {
		const onnavigate = vi.fn();
		const target = render(SetupStepper, { current: 'starter', ownerExists: true, onnavigate });
		const current = target.querySelector('[aria-current="step"]') as HTMLElement;
		expect(current.textContent).toContain('Starter');
		const back = target.querySelector('[aria-label="Back to Models"]') as HTMLButtonElement;
		expect(back.disabled).toBe(false);
		back.click();
		expect(onnavigate).toHaveBeenCalledWith('models');
	});

	it('locks Welcome and Account once the owner exists and future steps always', () => {
		const target = render(SetupStepper, { current: 'starter', ownerExists: true, onnavigate: () => {} });
		expect(target.querySelector('[aria-label="Back to Welcome"]')).toBeNull();
		expect(target.querySelector('[aria-label="Back to Account"]')).toBeNull();
		const buttons = Array.from(target.querySelectorAll<HTMLButtonElement>('ol button'));
		const install = buttons.find((b) => b.textContent?.includes('Install'))!;
		expect(install.disabled).toBe(true);
	});

	it('lets the focus shell go back to Welcome before an owner exists', () => {
		const target = render(SetupStepper, { current: 'account', ownerExists: false, onnavigate: () => {} });
		expect(target.querySelector('[aria-label="Back to Welcome"]')).not.toBeNull();
	});

	it('shows the compact step line for narrow screens', () => {
		const target = render(SetupStepper, { current: 'computer', ownerExists: true });
		expect(target.textContent).toContain('Step 3 of 7');
	});
});

describe('StarterCard', () => {
	it('shows size, source host, graphics memory and fit against the card', () => {
		const target = render(StarterCard, {
			recipe: ZIMAGE,
			machine: machine(24),
			checked: true,
			tabbable: true,
			recommended: true,
			onselect: () => {}
		});
		expect(target.querySelector('[data-spec-download]')?.textContent?.trim()).toBe('19.27 GB');
		expect(target.querySelector('[data-spec-source]')?.textContent?.trim()).toBe('huggingface.co');
		expect(target.querySelector('[data-spec-vram]')?.textContent?.trim()).toBe('12 to 16 GB suggested');
		expect(target.querySelector('[data-spec-fit]')?.textContent?.trim()).toBe('Fits your card');
		expect(target.textContent).toContain('Recommended for your card');
	});

	it('warns when the card falls short', () => {
		const target = render(StarterCard, {
			recipe: QWEN,
			machine: machine(8),
			checked: false,
			tabbable: false,
			onselect: () => {}
		});
		expect(target.querySelector('[data-spec-fit]')?.textContent?.trim()).toBe('Wants 16 GB, you have 8 GB');
		expect(target.textContent).not.toContain('Recommended for your card');
	});
});

describe('StarterStep', () => {
	const baseProps = {
		loading: false,
		error: '',
		modelsPath: '/data/models',
		starting: false,
		startError: '',
		onStart: () => {},
		onSkip: () => {},
		onRetry: () => {}
	};

	it('preselects the recommended starter first and states its size on the button', () => {
		const target = render(StarterStep, {
			...baseProps,
			recipes: [QWEN, SDXL, ZIMAGE, recipe({ id: 'klein', name: 'Klein', first_run: false })],
			machine: machine(24)
		});
		const radios = Array.from(target.querySelectorAll<HTMLElement>('[role="radio"]'));
		expect(radios.map((r) => r.textContent?.includes('Z-Image Starter'))).toEqual([true, false, false]);
		expect(radios[0].getAttribute('aria-checked')).toBe('true');
		expect(target.textContent).not.toContain('Klein');
		expect(target.textContent).toContain('Review 19.27 GB download');
	});

	it('recommends SDXL on an 8 GB card and follows a different pick', () => {
		const onStart = vi.fn();
		const target = render(StarterStep, {
			...baseProps,
			onStart,
			recipes: [QWEN, SDXL, ZIMAGE],
			machine: machine(8)
		});
		const radios = Array.from(target.querySelectorAll<HTMLElement>('[role="radio"]'));
		expect(radios[0].textContent).toContain('SDXL Starter');
		expect(radios[0].getAttribute('aria-checked')).toBe('true');
		radios[1].click();
		flushSync();
		const review = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Review'))!;
		review.click();
		expect(onStart).toHaveBeenCalledWith(ZIMAGE);
	});

	it('preselects nothing and disables review without a GPU', () => {
		const target = render(StarterStep, {
			...baseProps,
			recipes: [QWEN, SDXL, ZIMAGE],
			machine: machine(null)
		});
		expect(target.querySelector('[role="radio"][aria-checked="true"]')).toBeNull();
		const review = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Review'))!;
		expect(review.disabled).toBe(true);
		expect(target.textContent).toContain('No NVIDIA graphics card was found.');
		expect(target.textContent).toContain('Skip for now');
	});

	it('shows an error with a retry', () => {
		const onRetry = vi.fn();
		const target = render(StarterStep, {
			...baseProps,
			onRetry,
			recipes: null,
			error: 'boom',
			machine: null
		});
		expect(target.textContent).toContain('boom');
		Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Try again'))!.click();
		expect(onRetry).toHaveBeenCalled();
	});
});

describe('MachineSummary', () => {
	it('renders detected values and a verdict', () => {
		const target = render(MachineSummary, { machine: machine(31.4), recipes: [QWEN, SDXL, ZIMAGE] });
		expect(target.textContent).toContain('NVIDIA GeForce RTX');
		expect(target.textContent).toContain('31 GB');
		expect(target.textContent).toContain('64 GB');
		expect(target.textContent).toContain('412 GB');
		expect(target.querySelector('[data-verdict="good"]')).not.toBeNull();
	});

	it('explains a missing GPU', () => {
		const target = render(MachineSummary, { machine: machine(null), recipes: [] });
		expect(target.textContent).toContain('No NVIDIA graphics card found');
		expect(target.querySelector('[data-verdict="no-gpu"]')).not.toBeNull();
	});

	it('degrades gracefully when nothing could be detected', () => {
		const target = render(MachineSummary, { machine: null, recipes: [] });
		expect(target.textContent?.match(/Couldn't detect/g)?.length).toBe(3);
		expect(target.querySelector('[data-verdict="unknown"]')).not.toBeNull();
	});

	it('copes with partial data', () => {
		const target = render(MachineSummary, {
			machine: machine(16, { ram_total_bytes: null, models_disk: null }),
			recipes: []
		});
		expect(target.textContent?.match(/Couldn't detect/g)?.length).toBe(2);
	});
});

describe('StepFacts', () => {
	it('renders plain and monospaced parts', () => {
		const target = render(StepFacts, { facts: [{ label: 'Stored in', body: [{ mono: 'storage/db.sqlite' }, ' inside it.'] }] });
		expect(target.textContent).toContain('Stored in');
		expect(target.querySelector('.font-mono.tabular-nums')?.textContent).toBe('storage/db.sqlite');
	});

	it('offers line breaks only after path separators', () => {
		const target = render(StepFacts, { facts: [{ label: 'Saved to', body: [{ mono: '/workspace/potionui/models' }] }] });
		const mono = target.querySelector('.font-mono.tabular-nums')!;
		expect(mono.textContent).toBe('/workspace/potionui/models');
		expect(mono.querySelectorAll('wbr')).toHaveLength(3);
		expect(mono.className).not.toContain('anywhere');
		const windows = render(StepFacts, { facts: [{ label: 'Saved to', body: [{ mono: 'C:\\models\\flux' }] }] });
		expect(windows.querySelectorAll('wbr')).toHaveLength(2);
	});
});
