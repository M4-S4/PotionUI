// @vitest-environment jsdom
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { tick } from 'svelte';

vi.mock('../../src/lib/utils/chipParser', () => ({
	hydrateSegments: async (segments: unknown[]) => segments
}));

const { default: SegmentedPromptEditor } = await import(
	'../../src/lib/components/SegmentedPromptEditor.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

const NEED: Record<string, number> = {
	'row||': 640,
	'row|styles variables|': 440,
	'row|styles variables library|': 320,
	'row|styles variables library|variables': 240,
	'row|styles variables library|variables styles': 170,
	'row|variables|': 440,
	'row|variables library|': 320,
	'row|variables library|variables': 240
};

let headerWidth = 1000;
let resizeCallbacks: Array<() => void> = [];

class TestResizeObserver {
	constructor(private cb: (entries: unknown[]) => void) {}
	observe(el: Element) {
		if (el.classList.contains('composer-toolbar')) resizeCallbacks.push(() => this.cb([]));
	}
	unobserve() {}
	disconnect() {}
}

function stage(el: HTMLElement) {
	return `${el.dataset.wrap ? 'wrap' : 'row'}|${el.dataset.icons ?? ''}|${el.dataset.folded ?? ''}`;
}

beforeEach(() => {
	headerWidth = 1000;
	resizeCallbacks = [];
	vi.stubGlobal('ResizeObserver', TestResizeObserver);
	Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
		configurable: true,
		get(this: HTMLElement) {
			return this.classList.contains('composer-toolbar') ? headerWidth : 0;
		}
	});
	Object.defineProperty(HTMLElement.prototype, 'scrollWidth', {
		configurable: true,
		get(this: HTMLElement) {
			return this.classList.contains('composer-toolbar') ? Math.max(NEED[stage(this)] ?? 0, 0) : 0;
		}
	});
});

afterEach(() => {
	document.body.innerHTML = '';
	vi.unstubAllGlobals();
	delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
	delete (HTMLElement.prototype as unknown as Record<string, unknown>).scrollWidth;
});

function mount(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({
		component: SegmentedPromptEditor as never,
		target,
		props: {
			segments: [{ id: 'a', content: 'a lighthouse keeper', type: 'content', chips: {}, enabled: true }],
			onOpenStyles: () => {},
			onOpenVariableManager: () => {},
			variables: { mood: { type: 'choice', options: ['calm'] } },
			...props
		}
	});
	return target;
}

async function settle() {
	await tick();
	await Promise.resolve();
	await tick();
	await Promise.resolve();
	await tick();
}

async function resizeTo(width: number) {
	headerWidth = width;
	resizeCallbacks.forEach((cb) => cb());
	await settle();
}

const main = (target: HTMLElement) => target.querySelector('header.composer-toolbar:not(.negative-header)') as HTMLElement;

async function libraryItems(target: HTMLElement, scope = 'header.composer-toolbar:not(.negative-header)') {
	(target.querySelector(`${scope} button[aria-label="Library"]`) as HTMLButtonElement).click();
	await settle();
	return Array.from(document.body.querySelectorAll('[role="menu"] [role="menuitem"]')).map((el) =>
		(el.querySelector('span span')?.textContent ?? el.textContent ?? '').trim()
	);
}

describe('prompt header ladder', () => {
	it('shows every action with its label when the pane is wide', async () => {
		const target = mount();
		await settle();

		expect(stage(main(target))).toBe('row||');
		expect(Array.from(main(target).querySelectorAll('[data-toolbar-action]')).map((el) => (el as HTMLElement).dataset.toolbarAction)).toEqual([
			'styles',
			'variables',
			'library'
		]);
		expect(await libraryItems(target)).not.toContain('Variables');
	});

	it('goes icon-only first, then drops the Library label, then folds Variables and Styles into the menu', async () => {
		const target = mount();
		await settle();

		await resizeTo(500);
		expect(stage(main(target))).toBe('row|styles variables|');
		await resizeTo(380);
		expect(stage(main(target))).toBe('row|styles variables library|');
		await resizeTo(260);
		expect(stage(main(target))).toBe('row|styles variables library|variables');
		await resizeTo(200);
		expect(stage(main(target))).toBe('row|styles variables library|variables styles');
	});

	it('restores the labels when the pane widens again', async () => {
		const target = mount();
		await settle();

		await resizeTo(180);
		expect(main(target).dataset.folded).toBe('variables styles');
		await resizeTo(1000);
		expect(stage(main(target))).toBe('row||');
	});

	it('leaves no action unreachable at any width', async () => {
		const target = mount();
		await settle();

		for (const width of [900, 700, 500, 380, 260, 200, 120]) {
			await resizeTo(width);
			const folded = (main(target).dataset.folded ?? '').split(' ').filter(Boolean);
			const items = await libraryItems(target);
			for (const key of folded) expect(items).toContain(key === 'styles' ? 'Styles' : 'Variables');
			const shown = ['styles', 'variables'].filter((key) => !folded.includes(key));
			for (const key of shown) {
				expect(main(target).querySelector(`[data-header-action="${key}"]`)).toBeTruthy();
			}
			(target.querySelector('header.composer-toolbar:not(.negative-header) button[aria-label="Library"]') as HTMLButtonElement).click();
			await settle();
		}
	});

	it('leads the menu with the folded actions, Styles before Variables, carrying their state', async () => {
		const target = mount({ appliedStyleName: 'Cinematic' });
		await settle();
		await resizeTo(180);

		const items = await libraryItems(target);
		expect(items.slice(0, 2)).toEqual(['Styles', 'Variables']);
		const states = Array.from(document.body.querySelectorAll('[role="menu"] [role="menuitem"] .text-signal')).map((el) => el.textContent);
		expect(states).toEqual(['Cinematic', '1']);
	});

	it('flags the Library button when a folded action has state, and only then', async () => {
		const dot = (target: HTMLElement) => main(target).querySelector('[data-testid="library-state-dot"]');

		const stateful = mount({ appliedStyleName: 'Cinematic' });
		await settle();
		expect(dot(stateful)).toBeNull();
		await resizeTo(180);
		expect(dot(stateful)).toBeTruthy();
		await resizeTo(1000);
		expect(dot(stateful)).toBeNull();

		const calm = mount({ variables: {} });
		await settle();
		await resizeTo(180);
		expect(dot(calm)).toBeNull();
	});

	it('runs a folded action from the menu', async () => {
		const onOpenVariableManager = vi.fn();
		const onOpenStyles = vi.fn();
		const target = mount({ onOpenVariableManager, onOpenStyles });
		await settle();
		await resizeTo(180);

		await libraryItems(target);
		const row = (label: string) =>
			Array.from(document.body.querySelectorAll('[role="menu"] [role="menuitem"]')).find((el) =>
				(el.textContent || '').trim().startsWith(label)
			) as HTMLButtonElement;
		row('Variables').click();
		expect(onOpenVariableManager).toHaveBeenCalledTimes(1);
		await settle();
		await libraryItems(target);
		row('Styles').click();
		expect(onOpenStyles).toHaveBeenCalledTimes(1);
	});

	it('only folds what the call site renders', async () => {
		const target = mount({ onOpenStyles: null });
		await settle();
		await resizeTo(180);

		expect(stage(main(target))).toBe('row|variables library|variables');
		expect((await libraryItems(target)).slice(0, 1)).toEqual(['Variables']);
	});

	it('starts at icons in compact mode', async () => {
		const target = mount({ compact: true });
		await settle();

		expect(stage(main(target))).toBe('row|styles variables|');
	});

	it('wraps the actions under the tab strip before touching them when tabs are present', async () => {
		const { createRawSnippet } = await import('svelte');
		const titleExtra = createRawSnippet(() => ({
			render: () => '<div class="prompt-tabs" data-header-lead><div data-header-strip></div></div>'
		}));
		NEED['row||'] = 640;
		NEED['wrap||'] = 500;
		NEED['wrap|styles variables|'] = 400;
		const target = mount({ titleExtra });
		await settle();

		await resizeTo(560);
		expect(stage(main(target))).toBe('wrap||');
		await resizeTo(450);
		expect(stage(main(target))).toBe('wrap|styles variables|');
		delete NEED['wrap||'];
		delete NEED['wrap|styles variables|'];
	});

	it('never folds anything out of the negative header', async () => {
		const target = mount({ negativeSegments: [{ id: 'n', content: 'blurry', type: 'content', chips: {}, enabled: true }] });
		await settle();
		await resizeTo(120);

		const negative = target.querySelector('header.negative-header') as HTMLElement;
		expect(negative.dataset.folded).toBe('');
		expect(negative.querySelector('button[aria-label="Library"]')).toBeTruthy();
		expect(await libraryItems(target, 'header.negative-header')).toEqual([
			'Prompts',
			'Segments',
			'Templates',
			'Save as Prompt',
			'Save as Segment Template',
			'Copy prompt'
		]);
	});
});
