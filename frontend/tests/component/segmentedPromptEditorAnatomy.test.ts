// @vitest-environment jsdom
//
// Two structural claims the rework makes that only a mount can check: the
// negative region is now the same anatomy as Prompt (header, cards, add row —
// no collapsed one-line summary), and the resolved panel appears only where the
// call site asks for it. `showPreview={false}` is load-bearing: the generate
// sidebar and Video Director both pass it, and a panel appearing there would be
// a visible change at a mount site nobody asked about.
import { describe, it, expect, afterEach, vi } from 'vitest';

vi.mock('../../src/lib/utils/chipParser', () => ({
	hydrateSegments: async (segments: unknown[]) => segments
}));

const { default: SegmentedPromptEditor } = await import(
	'../../src/lib/components/SegmentedPromptEditor.svelte'
);
const { createClassComponent } = await import('svelte/legacy');
const { tick, createRawSnippet } = await import('svelte');
const { settle } = await import('./helpers/settle');
const { isNegativeInert } = await import('../../src/lib/generation/negativeApplied');

function segment(id: string, content: string, partial: Record<string, unknown> = {}) {
	return { id, content, type: 'content', chips: {}, enabled: true, ...partial };
}

const tabStrip = createRawSnippet(() => ({
	render: () => '<div class="prompt-tabs" data-header-lead><div data-header-strip role="tablist"></div></div>'
}));

function mount(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: SegmentedPromptEditor as never,
		target,
		props: { segments: [segment('a', 'a lighthouse keeper')], ...props }
	});

	const buttons = () => Array.from(target.querySelectorAll('button'));
	return {
		target,
		component,
		buttons,
		byText: (text: string) =>
			buttons().find((b) => (b.textContent || '').trim() === text) as HTMLButtonElement | undefined,
		text: () => target.textContent || '',
		lists: () =>
			Array.from(target.querySelectorAll('[role="list"]')).map(
				(el) => el.getAttribute('aria-label') || ''
			)
	};
}

afterEach(() => {
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

describe('the negative region', () => {
	it('repeats the Prompt anatomy — its own header, list and add row', () => {
		const editor = mount({ negativeSegments: [segment('n1', 'blurry')] });

		expect(editor.text()).toContain('Negative');
		expect(editor.lists()).toEqual(expect.arrayContaining(['Positive segments', 'Negative segments']));
		// One add row per region, both reachable without expanding anything.
		expect(editor.buttons().filter((b) => (b.textContent || '').includes('Add segment'))).toHaveLength(2);
	});

	it('gives its header the title and the Library menu, with no segment count', () => {
		const editor = mount({ negativeSegments: [segment('n1', 'blurry'), segment('n2', 'watermark')] });

		const header = editor.target.querySelector('header.negative-header') as HTMLElement;
		expect(header.querySelector('.composer-title')!.textContent).toBe('Negative');
		expect(header.querySelector('button[aria-label="Library"]')).toBeTruthy();
		expect(editor.target.querySelector('.composer-count')).toBeNull();
		expect(editor.text()).not.toMatch(/\d+ segments?/);
	});

	it('hides the whole negative section while guidance makes it inert', () => {
		const inert = mount({ negativeSegments: [segment('n1', 'blurry')], negativeInert: true });
		expect(inert.lists()).not.toContain('Negative segments');
		expect(inert.target.querySelector('.negative-composer')).toBeNull();

		const live = mount({ negativeSegments: [segment('n1', 'blurry')] });
		expect(live.lists()).toContain('Negative segments');
	});

	it("hides and shows per the preset's applies_when declaration", async () => {
		const declarations = {
			modes: {
				txt2img: {
					default: {
						logic: 'OR' as const,
						conditions: [
							{ field: 'cfg', greater_than: 1 },
							{ field: 'nag_enabled', equals: true }
						]
					}
				}
			}
		};
		const inertFor = (formData: Record<string, unknown>, decl: unknown = declarations) =>
			isNegativeInert(decl as never, 'txt2img', null, formData);

		const editor = mount({
			negativeSegments: [segment('n1', 'blurry')],
			negativeInert: inertFor({ cfg: 1, nag_enabled: false })
		});
		expect(editor.lists()).not.toContain('Negative segments');
		const setInert = (negativeInert: boolean) =>
			(editor.component as unknown as { $set(props: Record<string, unknown>): void }).$set({ negativeInert });

		setInert(inertFor({ cfg: 4, nag_enabled: false }));
		await tick();
		expect(editor.lists()).toContain('Negative segments');

		setInert(inertFor({ cfg: 1, nag_enabled: false }));
		await tick();
		expect(editor.lists()).not.toContain('Negative segments');

		setInert(inertFor({ cfg: 1, nag_enabled: true }));
		await tick();
		expect(editor.lists()).toContain('Negative segments');

		setInert(inertFor({ cfg: 1 }, null));
		await tick();
		expect(editor.lists()).toContain('Negative segments');
	});

	it('is absent entirely when the call site pairs no negative list', () => {
		const editor = mount();
		expect(editor.lists()).not.toContain('Negative segments');
	});
});

describe('adding segments', () => {
	// e2e (fe74) reported "Add segment" clicked twice still leaving 1 listitem
	// in the list. Reproduced here to rule the component in or out: two clicks
	// append two rows, both inside the same role="list" container the mock's
	// anatomy renders (not a sibling of it) — confirms the real failure was
	// downstream (a focus-restoration race after the Details modal closes,
	// stealing a keystroke into a global shortcut — see fe70/fe74's own
	// `setSegmentMeta` fix), not this component's add/commit path.
	it('appends a row per click, every row inside the same list container', async () => {
		const editor = mount();
		const addBtn = () => editor.buttons().find((b) => (b.textContent || '').includes('Add segment'));
		// Each click's own reactive update needs to settle before the DOM (and
		// the button reference it re-renders) reflects it — a real click, with
		// its own dispatch + paint cycle, always clears that; two `.click()`
		// calls back-to-back in the same tick do not, which is a test-harness
		// gap, not the double-click bug e2e chased (that traced to a focus race
		// after the Details modal, not to this add/commit path — see the
		// describe block's own comment).
		addBtn()?.click();
		await Promise.resolve();
		addBtn()?.click();
		await Promise.resolve();

		const list = editor.target.querySelector('div[role="list"][aria-label="Positive segments"]');
		expect(list).not.toBeNull();
		expect(list?.querySelectorAll('[role="listitem"]')).toHaveLength(3);
		expect(editor.target.querySelectorAll('[role="listitem"]')).toHaveLength(3);
	});
});

describe('embedded (Video Director\'s stage beat)', () => {
	// StageBeat.svelte already gives this editor its own card (`.editor` in
	// its own stylesheet) — without `embedded`, the mock's `.composer` shell
	// nests a second, redundant container around the same segments.
	it('drops the composer card and toolbar, keeping the list and add row', () => {
		const editor = mount({ embedded: true, label: 'Prompt' });

		expect(editor.target.querySelector('.composer')).toBeNull();
		expect(editor.target.querySelector('.composer-toolbar')).toBeNull();
		expect(editor.target.querySelector('div[role="list"][aria-label="Prompt"]')).toBeTruthy();
		expect(editor.buttons().some((b) => (b.textContent || '').includes('Add segment'))).toBe(true);
	});

	it('renders the ordinary composer card when not embedded', () => {
		const editor = mount({ embedded: false, label: 'Prompt' });
		expect(editor.target.querySelector('.composer')).toBeTruthy();
		expect(editor.target.querySelector('.composer-toolbar')).toBeTruthy();
	});
});

describe('the Styles toolbar button', () => {
	it('is absent when the call site has no styles to offer', () => {
		const editor = mount();
		expect(editor.byText('Styles')).toBeUndefined();
	});

	it('appears first in the main header, before Variables and Library, when the call site sets onOpenStyles', () => {
		const editor = mount({ onOpenStyles: () => {}, onOpenVariableManager: () => {} });
		const names = Array.from(editor.target.querySelectorAll('header.composer-toolbar:not(.negative-header) [data-toolbar-action] button')).map(
			(button) => button.getAttribute('aria-label')
		);
		expect(names).toEqual(['Styles', 'Variables', 'Library']);
	});

	it('shows the applied style name instead of the bare label', () => {
		const editor = mount({ onOpenStyles: () => {}, appliedStyleName: '80s OVA Sci-Fi' });
		expect(editor.byText('Style: 80s OVA Sci-Fi')).toBeTruthy();
	});
});

describe('the Library menu', () => {
	const items = () =>
		Array.from(document.body.querySelectorAll('[role="menu"] [role="menuitem"]')).map(
			(el) => el.querySelector('span span')?.textContent?.trim() ?? (el.textContent || '').trim()
		);

	async function openLibrary(editor: ReturnType<typeof mount>, scope = 'header:not(.negative-header)') {
		(editor.target.querySelector(`${scope} button[aria-label="Library"]`) as HTMLButtonElement).click();
		await settle();
	}

	it('replaces the row of Prompts, Segments and Templates buttons with one menu', () => {
		const editor = mount();
		expect(editor.byText('Prompts')).toBeUndefined();
		expect(editor.byText('Segments')).toBeUndefined();
		expect(editor.byText('Templates')).toBeUndefined();
		expect(editor.target.querySelector('button[aria-label="Library"]')!.getAttribute('aria-haspopup')).toBe('menu');
		expect(editor.target.querySelector('button[aria-label="More prompt actions"]')).toBeNull();
	});

	it('lists Insert then Save, in that order', async () => {
		const editor = mount();
		await openLibrary(editor);

		expect(items()).toEqual(['Prompts', 'Segments', 'Templates', 'Save as Prompt', 'Save as Segment Template', 'Copy prompt']);
		const groups = Array.from(document.body.querySelectorAll('[role="menu"] [role="group"] > div[aria-hidden="true"]')).map(
			(el) => el.textContent
		);
		expect(groups).toEqual(['Insert', 'Save']);
	});

	it('keeps Apply Prompt out of the menu', async () => {
		const editor = mount();
		await openLibrary(editor);
		expect(items()).toContain('Save as Prompt');
		expect(items()).not.toContain('Apply Prompt');
	});

	it('hides Save as Segment Template and Copy prompt while the prompt is empty', async () => {
		const editor = mount({ segments: [segment('a', '')] });
		await openLibrary(editor);

		expect(items()).toEqual(['Prompts', 'Segments', 'Templates', 'Save as Prompt']);
	});

	it('has its own menu on the negative header, labelled for it', async () => {
		const editor = mount({ negativeSegments: [segment('n1', 'blurry')] });
		await openLibrary(editor, 'header.negative-header');

		expect(items()).toEqual(['Prompts', 'Segments', 'Templates', 'Save as Prompt', 'Save as Segment Template', 'Copy prompt']);
		expect(document.body.querySelector('[role="menu"]')!.getAttribute('aria-label')).toBe('Library (negative)');
	});

	it('offers only Copy under a plain more-actions button when the call site has no library', async () => {
		const editor = mount({ showLibraryActions: false });
		(editor.target.querySelector('button[aria-label="More prompt actions"]') as HTMLButtonElement).click();
		await settle();

		expect(editor.target.querySelector('button[aria-label="Library"]')).toBeNull();
		expect(items()).toEqual(['Copy prompt']);
	});

	it('drops the menu button altogether when there is nothing to put in it', () => {
		const editor = mount({ showLibraryActions: false, segments: [segment('a', '')] });
		expect(editor.target.querySelector('button[aria-label="Library"]')).toBeNull();
		expect(editor.target.querySelector('button[aria-label="More prompt actions"]')).toBeNull();
	});

	it('writes no title attribute on any header control', () => {
		const editor = mount({ onOpenStyles: () => {}, onOpenVariableManager: () => {}, negativeSegments: [segment('n1', 'blurry')] });
		expect(editor.target.querySelectorAll('header.composer-toolbar [title]')).toHaveLength(0);
	});
});

describe('the header title', () => {
	it('is a plain Prompt title with no tab strip when the call site injects none', () => {
		const editor = mount();
		const header = editor.target.querySelector('header.composer-toolbar') as HTMLElement;

		expect(header.querySelector('.composer-title')!.textContent).toBe('Prompt');
		expect(header.classList.contains('has-tabs')).toBe(false);
		expect(editor.target.querySelector('[data-header-strip]')).toBeNull();
	});

	it('uses the label the call site passes', () => {
		const editor = mount({ label: 'Template slots' });
		expect(editor.target.querySelector('header.composer-toolbar .composer-title')!.textContent).toBe('Template slots');
	});

	it('hands the title over to an injected tab strip and wraps the header around it', () => {
		const editor = mount({ titleExtra: tabStrip });
		const header = editor.target.querySelector('header.composer-toolbar') as HTMLElement;

		expect(header.classList.contains('has-tabs')).toBe(true);
		expect(header.querySelector('.composer-title')).toBeNull();
		expect(header.querySelector('[data-header-strip]')).toBeTruthy();
		expect(header.querySelector('button[aria-label="Library"]')).toBeTruthy();
	});
});

describe('the resolved panel', () => {
	it('is rendered when the call site asks for a preview', () => {
		const editor = mount({ showPreview: true });
		expect(editor.text()).toContain('What the model receives');
	});

	it('stays out of the way when the call site opts out', () => {
		const editor = mount({ showPreview: false });
		expect(editor.text()).not.toContain('What the model receives');
	});

	it('counts what the model receives, leaving disabled segments out', () => {
		const editor = mount({
			showPreview: true,
			segments: [segment('a', 'a forest'), segment('b', 'harsh noon sun', { enabled: false })]
		});

		expect(editor.text()).toContain(`${'a forest'.length} chars`);
		// The disabled card still shows its own text — the panel is what must not.
		expect(editor.text()).toContain('harsh noon sun');
		const panel = editor.target.querySelector('.resolved-body');
		expect(panel?.textContent).toBe('a forest');
	});

});
