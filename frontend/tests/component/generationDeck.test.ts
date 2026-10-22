import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { flushSync } from 'svelte';
import type { GenerationState } from '$lib/types/tabs';
import { STATUS_LINES, STATUS_LINE_MS, STATUS_FADE_MS } from '$lib/components/generation-panel/statusLines';
import { playfulStatusLines } from '$lib/stores/playfulStatusLines';

const { default: GenerationPanel } = await import('../../src/lib/components/GenerationPanel.svelte');
const { createClassComponent } = await import('svelte/legacy');

const preset = { id: 'flux2', name: 'FLUX.2', cover: null, category: 'image' };

function generation(overrides: Partial<GenerationState> = {}): GenerationState {
	return {
		isGenerating: false,
		currentGeneration: null,
		currentProgress: null,
		pipeTimers: {},
		startedAt: null,
		totalTime: null,
		lastDurationMs: null,
		batchImages: [],
		batchVideos: [],
		batchAudios: [],
		artifacts: [],
		workbenchIndex: 0,
		workbenchTotal: 0,
		queue: [],
		submittedPromptTemplate: null,
		...overrides
	};
}

function running(progress: number | null, overrides: Partial<GenerationState> = {}) {
	return generation({
		isGenerating: true,
		currentGeneration: { id: 'run-1', status: 'running' },
		currentProgress: { step: 'Sampling', progress, current_step: 'Sampling' },
		startedAt: Date.now(),
		...overrides
	});
}

type Mounted = { target: HTMLElement; set: (props: Record<string, unknown>) => void; destroy: () => void };
let mounted: Mounted | undefined;
const realMatchMedia = window.matchMedia;

function mount(props: Record<string, unknown> = {}): Mounted {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationPanel as never,
		target,
		props: { generation: generation(), isGenerating: false, canGenerate: true, tabId: '', preset, modeLabel: 'Text to image', ...props }
	});
	flushSync();
	return {
		target,
		set: (next) => {
			component.$set(next);
			flushSync();
		},
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

const q = <T extends Element = HTMLElement>(selector: string) => mounted!.target.querySelector<T>(selector);
const title = () => q('[data-deck-title]')!.textContent!.trim();
const literal = () => q('[data-deck-literal]')!.textContent!.trim();
const live = () => q('[data-deck-announcer]')!.textContent!.trim();
const detail = () => q('.deck-detail-text')!.textContent!.replace(/\s+/g, ' ').trim();
const thumb = () => q<HTMLElement>('.deck-thumb')!;

function preferReducedMotion() {
	window.matchMedia = ((query: string) => ({
		matches: query.includes('prefers-reduced-motion'),
		media: query,
		onchange: null,
		addListener() {},
		removeListener() {},
		addEventListener() {},
		removeEventListener() {},
		dispatchEvent: () => false
	})) as typeof window.matchMedia;
}

beforeEach(() => {
	playfulStatusLines.set(true);
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	window.matchMedia = realMatchMedia;
	vi.useRealTimers();
});

describe('generation deck progress', () => {
	it('has exactly one progress bar, hidden at rest and determinate while running', () => {
		mounted = mount();
		const bars = mounted.target.querySelectorAll('[role="progressbar"]');
		expect(bars).toHaveLength(1);
		expect((bars[0] as HTMLElement).hidden).toBe(true);
		mounted.set({ generation: running(0.64), isGenerating: true });
		const bar = q<HTMLElement>('[role="progressbar"]')!;
		expect(bar.hidden).toBe(false);
		expect(bar.getAttribute('aria-label')).toBe('Generation progress');
		expect(bar.getAttribute('aria-valuenow')).toBe('64');
		expect(mounted.target.querySelectorAll('[role="progressbar"]')).toHaveLength(1);
	});

	it('is indeterminate while the model loads', () => {
		mounted = mount({ generation: running(null), isGenerating: true });
		expect(q('[role="progressbar"]')!.hasAttribute('aria-valuenow')).toBe(false);
		expect(q('.generation-panel')!.getAttribute('data-state')).toBe('loading');
	});

	it('keeps nothing progress-like around the play control', () => {
		mounted = mount({ generation: running(0.4), isGenerating: true });
		const run = q('.run-cluster')!;
		expect(run.querySelectorAll('[role="progressbar"], circle[stroke-dasharray], .mark-ring, [class*="progress"], [class*="ring"]')).toHaveLength(0);
		const mark = q<HTMLButtonElement>('button.generate-button')!;
		expect(mark.textContent!.trim()).toBe('');
		expect(mark.getAttribute('aria-label')).toBe('Cancel generation');
		expect(mark.classList.contains('is-cancel')).toBe(true);
		expect(mark.querySelector('.mark-orbit')).not.toBeNull();
	});

	it('names the idle play control Generate and keeps it icon-only', () => {
		mounted = mount();
		const mark = q<HTMLButtonElement>('button.generate-button')!;
		expect(mark.getAttribute('aria-label')).toBe('Generate');
		expect(mark.textContent!.trim()).toBe('');
		expect(mark.disabled).toBe(false);
	});
});

describe('generation deck status', () => {
	it('shows Done after a completed run', () => {
		mounted = mount({
			generation: generation({ currentGeneration: { id: 'run-1', status: 'completed' }, batchImages: [{ url: '/a.png' }], lastDurationMs: 18400 })
		});
		expect(title()).toBe('Done');
		expect(q('.deck-eyebrow')!.textContent).toBe('New result');
		expect(detail()).toBe('FLUX.2 · Text to image');
	});

	it('shows Failed with the error and a Details action', () => {
		mounted = mount({ generation: generation({ currentGeneration: { id: 'run-1', status: 'failed', message: 'Out of GPU memory' } }) });
		expect(title()).toBe('Failed');
		expect(detail()).toBe('Out of GPU memory');
		expect(q<HTMLButtonElement>('.inline-action')!.textContent!.trim()).toBe('Details');
	});

	it("shows Can't generate yet with the reason", () => {
		mounted = mount({ canGenerate: false, disabledReason: 'Select a preset to generate' });
		expect(title()).toBe("Can't generate yet");
		expect(detail()).toBe('Select a preset to generate');
		expect(q<HTMLButtonElement>('button.generate-button')!.disabled).toBe(true);
	});

	it('offers Switch to tab while another tab owns the worker', () => {
		const onSwitchToGeneratingTab = vi.fn();
		mounted = mount({ generatingTabName: 'Short film', isActiveTabGenerating: false, onSwitchToGeneratingTab });
		expect(title()).toBe('Generating in Short film');
		q<HTMLButtonElement>('.inline-action')!.click();
		expect(onSwitchToGeneratingTab).toHaveBeenCalledOnce();
	});

	it('says Ready to brew on a fresh tab and keeps a literal title for screen readers', () => {
		mounted = mount();
		expect(title()).toBe('Ready to brew');
		expect(literal()).toBe('Ready to generate');
		expect(q('[data-deck-title]')!.getAttribute('aria-hidden')).toBe('true');
		expect(q('.deck-eyebrow')!.textContent).toBe('No results yet');
		expect(detail()).toBe('FLUX.2 · Text to image');
	});

	it('keeps the percentage and the step in the secondary line', () => {
		mounted = mount({ generation: running(0.64), isGenerating: true });
		expect(detail()).toBe('64% · Sampling');
		mounted.set({ generation: running(0.64, { currentProgress: { step: 'Sampling', progress: 0.64, current_step: 'Sampling <<PROGRESS:18/28>>' } }) });
		expect(detail()).toBe('64% · Sampling · Step 18 / 28');
	});
});

describe('generation deck copy and dot', () => {
	const result = { batchImages: [{ url: '/a.png' }], lastDurationMs: 18400 };
	const finished = () => generation({ currentGeneration: { id: 'run-1', status: 'completed' }, ...result });
	const failed = () => generation({ currentGeneration: { id: 'run-1', status: 'failed', message: 'Out of GPU memory' }, ...result });
	const cancelled = () =>
		generation({ currentGeneration: { id: 'run-1', status: 'cancelled' }, cancelNotice: 'Generation cancelled', ...result });
	const idle = () => generation(result);
	const dot = () => q('.deck-dot')!;
	const tone = () => [...dot().classList].filter((name) => name.startsWith('is-'));
	const readout = () => q('.deck-timing .cell-value')!.textContent!.trim();

	it.each([
		['done', finished(), 'New result'],
		['ready with a last result', idle(), 'Last result'],
		['failed', failed(), 'Last result'],
		['cancelled', cancelled(), 'Last result']
	])('keeps the duration out of the eyebrow and the meta when %s, and in the readout', (_name, gen, eyebrow) => {
		mounted = mount({ generation: gen });
		expect(q('.deck-eyebrow')!.textContent).toBe(eyebrow);
		expect(q('.deck-copy')!.textContent).not.toContain('18.4s');
		expect(readout()).toBe('18.4s');
	});

	it.each([
		['done', finished()],
		['ready with a last result', idle()]
	])('shows the model and the mode as the meta when %s', (_name, gen) => {
		mounted = mount({ generation: gen });
		expect(detail()).toBe('FLUX.2 · Text to image');
	});

	it('shows the progress meta while running', () => {
		mounted = mount({ generation: running(0.64, { lastDurationMs: 18400 }), isGenerating: true });
		expect(detail()).toBe('64% · Sampling');
		expect(q('.deck-eyebrow')!.textContent).toBe('Now generating');
	});

	it('hangs the dot inside the title line, ahead of the title, so the three copy lines start together', () => {
		mounted = mount();
		expect([...q('.deck-copy')!.children].map((child) => child.classList[0])).toEqual(['deck-eyebrow', 'deck-title-line', 'deck-detail']);
		expect(q('.deck-title-line')!.firstElementChild).toBe(dot());
		expect(dot().nextElementSibling).toBe(q('[data-deck-title]'));
		expect(dot().getAttribute('aria-hidden')).toBe('true');
	});

	it.each([
		['done', {}, finished(), 'is-success'],
		['ready on a fresh tab', {}, generation(), 'is-muted'],
		['ready with a last result', {}, idle(), 'is-muted'],
		['cancelled', {}, cancelled(), 'is-muted'],
		['running', { isGenerating: true }, running(0.5), 'is-signal'],
		['loading the model', { isGenerating: true }, running(null), 'is-signal'],
		['failed', {}, failed(), 'is-danger'],
		['blocked', { canGenerate: false, disabledReason: 'Select a preset to generate' }, generation(), 'is-warning'],
		['owned by another tab', { generatingTabName: 'Short film', isActiveTabGenerating: false }, generation(), 'is-warning']
	])('paints the dot for %s', (_name, extra, gen, expected) => {
		mounted = mount({ generation: gen, ...extra });
		expect(tone()).toEqual([expected]);
	});
});

describe('generation deck status lines', () => {
	it('rotates potion lines while running without touching the live region', () => {
		vi.useFakeTimers();
		mounted = mount();
		mounted.set({ generation: running(0.4), isGenerating: true });
		const seen = [title()];
		const announced = live();
		expect(announced).toBe('Generation started');
		expect(literal()).toBe('Generating');
		for (let i = 0; i < 5; i++) {
			vi.advanceTimersByTime(i === 0 ? STATUS_LINE_MS + STATUS_FADE_MS : STATUS_LINE_MS);
			flushSync();
			seen.push(title());
			expect(live()).toBe(announced);
		}
		expect(seen.every((line) => STATUS_LINES.running.includes(line))).toBe(true);
		seen.slice(1).forEach((line, index) => expect(line).not.toBe(seen[index]));
	});

	it('uses the loading set while the model loads and switches to a finishing line at 80%', () => {
		vi.useFakeTimers();
		mounted = mount({ generation: running(null), isGenerating: true });
		expect(STATUS_LINES.loading).toContain(title());
		mounted.set({ generation: running(0.85) });
		vi.advanceTimersByTime(STATUS_FADE_MS);
		flushSync();
		expect(STATUS_LINES.finishing).toContain(title());
	});

	it('keeps one line for the whole run under reduced motion', () => {
		vi.useFakeTimers();
		preferReducedMotion();
		mounted = mount({ generation: running(0.4), isGenerating: true });
		const first = title();
		expect(STATUS_LINES.running).toContain(first);
		vi.advanceTimersByTime(STATUS_LINE_MS * 6);
		mounted.set({ generation: running(0.9) });
		vi.advanceTimersByTime(STATUS_LINE_MS);
		flushSync();
		expect(title()).toBe(first);
	});

	it('shows the plain title when the playful lines are turned off', () => {
		mounted = mount({ generation: running(0.4), isGenerating: true });
		expect(STATUS_LINES.running).toContain(title());
		playfulStatusLines.set(false);
		flushSync();
		expect(title()).toBe('Generating');
	});
});

describe('generation deck live region', () => {
	it('announces start, coarse progress and completion only', () => {
		mounted = mount();
		expect(live()).toBe('');
		const said: string[] = [];
		const note = () => {
			if (live() && live() !== said.at(-1)) said.push(live());
		};
		for (const progress of [0.1, 0.3, 0.35, 0.6, 0.8, 1]) {
			mounted.set({ generation: running(progress), isGenerating: true });
			note();
		}
		mounted.set({ generation: generation({ currentGeneration: { id: 'run-1', status: 'completed' }, batchImages: [{ url: '/a.png' }] }), isGenerating: false });
		note();
		expect(said).toEqual(['Generation started', '25% generated', '50% generated', '75% generated', 'Generation complete']);
	});

	it('announces a cancelled run', () => {
		mounted = mount({ generation: running(0.5), isGenerating: true });
		mounted.set({
			generation: generation({ currentGeneration: { id: 'run-1', status: 'cancelled' }, cancelNotice: 'Cancelled', cancelledGenerationId: 'run-1' }),
			isGenerating: false
		});
		expect(live()).toBe('Generation cancelled');
	});

	it('announces a failure with its message', () => {
		mounted = mount({ generation: running(0.5), isGenerating: true });
		mounted.set({ generation: generation({ currentGeneration: { id: 'run-1', status: 'failed', message: 'Out of GPU memory' } }), isGenerating: false });
		expect(live()).toBe('Generation failed: Out of GPU memory');
	});
});

describe('generation deck with a floating workbench', () => {
	const queueButton = () => q<HTMLButtonElement>('button[aria-label="Generation queue"]')!;
	const queuePopover = () => document.querySelector('[role="dialog"][aria-label="Generation queue"]');

	function hoverTooltip(button: HTMLElement) {
		button.parentElement!.dispatchEvent(new MouseEvent('mouseenter'));
		vi.advanceTimersByTime(150);
		flushSync();
		const text = document.querySelector('[role="tooltip"]')?.textContent?.trim() ?? '';
		button.parentElement!.dispatchEvent(new MouseEvent('mouseleave'));
		flushSync();
		return text;
	}

	it('makes Queue unavailable the same way as the drawers', () => {
		vi.useFakeTimers();
		mounted = mount({ workbenchFloating: true, generation: generation({ queue: [{ generation_id: 'q-1', queue_position: 1, status: 'pending' }] as never }) });
		const settings = q<HTMLButtonElement>('button[aria-label="Generation settings"]')!;
		expect(queueButton().disabled).toBe(true);
		expect(hoverTooltip(queueButton())).toBe('Unavailable while the workbench is floating');
		expect(hoverTooltip(queueButton())).toBe(hoverTooltip(settings));
		queueButton().click();
		flushSync();
		expect(queuePopover()).toBeNull();
		q<HTMLButtonElement>('[data-deck-more]')!.click();
		flushSync();
		const row = [...document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')].find((item) => item.textContent?.includes('Queue'));
		expect(row?.disabled).toBe(true);
	});

	it('closes an open queue popover when the workbench starts floating', () => {
		mounted = mount();
		expect(queueButton().disabled).toBe(false);
		queueButton().click();
		flushSync();
		expect(queuePopover()).not.toBeNull();
		mounted.set({ workbenchFloating: true });
		expect(queuePopover()).toBeNull();
	});
});

describe('generation deck halo', () => {
	const halo = () => [...mounted!.target.querySelectorAll('.deck-ambient img')].map((img) => img.getAttribute('src'));

	it('glows behind an image result and behind the live preview', () => {
		mounted = mount({ generation: generation({ batchImages: [{ url: '/r.png' }] }) });
		expect(halo()).toEqual(['/r.png']);
		mounted.destroy();
		mounted = mount({ generation: running(0.4, { currentGeneration: { id: 'run-1', status: 'running', current_image: '/p.png' } }), isGenerating: true });
		expect(halo()).toEqual(['/p.png']);
	});

	it.each([
		['a content-flagged result', generation({ batchImages: [{ url: '/r.png', content_flagged: true }] })],
		['a video result', generation({ batchVideos: [{ url: '/v.mp4', duration: 5 }] })]
	])('shows no halo for %s', (_name, result) => {
		mounted = mount({ generation: result });
		expect(mounted.target.querySelector('.deck-ambient')).not.toBeNull();
		expect(halo()).toEqual([]);
	});
});

describe('generation deck teardown', () => {
	it('leaves no timers behind when destroyed mid-run and mid-fade', () => {
		vi.useFakeTimers();
		mounted = mount({ generation: running(0.4), isGenerating: true });
		vi.advanceTimersByTime(STATUS_LINE_MS + 100);
		flushSync();
		expect(q('[data-deck-title]')!.classList.contains('is-fading')).toBe(true);
		expect(vi.getTimerCount()).toBeGreaterThan(0);
		mounted.destroy();
		mounted = undefined;
		expect(vi.getTimerCount()).toBe(0);
	});
});

describe('generation deck thumbnail', () => {
	function noUnsourcedImages() {
		expect(document.querySelectorAll('img:not([src]), img[src=""]')).toHaveLength(0);
	}

	it('shows the latest image with a count for several outputs', () => {
		mounted = mount({ generation: generation({ batchImages: [{ url: '/a.png' }, { url: '/b.png' }, { url: '/c.png' }, { url: '/d.png' }] }) });
		expect(thumb().dataset.thumb).toBe('image');
		expect(thumb().querySelector('img')!.getAttribute('src')).toBe('/d.png');
		expect(thumb().querySelector('.is-count')!.textContent).toBe('×4');
		noUnsourcedImages();
	});

	it('shows a video frame with a play glyph and its duration', () => {
		mounted = mount({ generation: generation({ batchVideos: [{ url: '/v.mp4', duration: 5 }] }) });
		expect(thumb().dataset.thumb).toBe('video');
		expect(thumb().querySelector('video')!.getAttribute('src')).toBe('/v.mp4');
		expect(thumb().querySelector('.deck-thumb-glyph svg path')).not.toBeNull();
		expect(thumb().querySelector('.is-duration')!.textContent).toBe('0:05');
		noUnsourcedImages();
	});

	it('shows a waveform tile with the duration for audio', () => {
		mounted = mount({ generation: generation({ batchAudios: [{ url: '/a.wav', duration: 94 }] }) });
		expect(thumb().dataset.thumb).toBe('audio');
		expect(thumb().querySelector('.deck-thumb-wave')).not.toBeNull();
		expect(thumb().querySelector('.is-duration')!.textContent).toBe('1:34');
		expect(thumb().querySelector('img')).toBeNull();
	});

	it('shows the 3D icon for meshes and the type icon for other outputs', () => {
		mounted = mount({ generation: generation({ batchMeshes: [{ url: '/m.glb' }] }) });
		expect(thumb().dataset.thumb).toBe('mesh');
		mounted.set({ generation: generation({ pluginOutputs: { caption_text: { msg: {}, pluginId: 'p', asset: 'a', generationId: 'g' } } }) });
		expect(thumb().dataset.thumb).toBe('text');
	});

	it('shows the preset cover with no result, else the family icon', () => {
		mounted = mount({ preset: { ...preset, cover: 'cover.webp' } });
		expect(thumb().dataset.thumb).toBe('cover');
		mounted.set({ preset });
		expect(thumb().dataset.thumb).toBe('family');
		expect(thumb().querySelector('img, video')).toBeNull();
	});

	it('shows the live preview while an image family runs and the target icon otherwise', () => {
		mounted = mount({
			generation: running(0.4, { currentGeneration: { id: 'run-1', status: 'running', current_image: '/preview.png' } }),
			isGenerating: true
		});
		expect(thumb().dataset.thumb).toBe('live');
		expect(thumb().querySelector('img')!.getAttribute('src')).toBe('/preview.png');
		mounted.set({ generation: running(0.4), preset: { ...preset, category: 'video' } });
		expect(thumb().dataset.thumb).toBe('video-pending');
		expect(thumb().querySelector('img, video')).toBeNull();
	});

	it('falls back to the type icon when the image fails to load', () => {
		mounted = mount({ generation: generation({ batchImages: [{ url: '/missing.png' }] }) });
		thumb().querySelector('img')!.dispatchEvent(new Event('error'));
		flushSync();
		expect(thumb().dataset.thumb).toBe('image-missing');
		expect(thumb().querySelector('img')).toBeNull();
	});
});
