import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';

const { default: AmbientGlow } = await import('$lib/components/AmbientGlow.svelte');
const { createClassComponent } = await import('svelte/legacy');

let target: HTMLElement | undefined;
let component: ReturnType<typeof createClassComponent> | undefined;

function mount(props: Record<string, unknown>) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: AmbientGlow as never, target, props });
	flushSync();
}

function swap(src: string) {
	component!.$set({ src });
	flushSync();
}

async function settle() {
	for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
	flushSync();
}

const media = () => [...target!.querySelectorAll('.ambient-media')];
const sources = () => media().map((el) => el.getAttribute('src'));
const fadeDurations = (animate: ReturnType<typeof vi.spyOn>) =>
	animate.mock.calls.map(([, options]) => (typeof options === 'object' && options ? options.duration : options)).filter((duration) => typeof duration === 'number' && duration > 0);

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	target = undefined;
	vi.restoreAllMocks();
});

describe('AmbientGlow', () => {
	it('renders a video for video media', () => {
		mount({ src: '/clip.mp4', kind: 'video' });
		expect(media()).toHaveLength(1);
		expect(media()[0].tagName).toBe('VIDEO');
		expect(sources()).toEqual(['/clip.mp4']);
	});

	it('renders no media without a source', () => {
		mount({ src: null });
		expect(target!.querySelector('.ambient')).not.toBeNull();
		expect(media()).toHaveLength(0);
	});

	it('keeps one element and swaps its source in place when unkeyed', () => {
		mount({ src: '/a.png', keyed: false });
		const first = media()[0];
		swap('/b.png');
		expect(media()).toEqual([first]);
		expect(first.getAttribute('src')).toBe('/b.png');
	});

	it('cross-fades a keyed swap and settles on exactly one element', async () => {
		const animate = vi.spyOn(Element.prototype, 'animate');
		mount({ src: '/a.png', crossfade: 400 });
		swap('/b.png');
		expect(sources().sort()).toEqual(['/a.png', '/b.png']);
		await settle();
		expect(fadeDurations(animate)).toContain(400);
		expect(sources()).toEqual(['/b.png']);
	});

	it('swaps without a fade under reduced motion', async () => {
		const animate = vi.spyOn(Element.prototype, 'animate');
		mount({ src: '/a.png', crossfade: 400, reducedMotion: true });
		swap('/b.png');
		await settle();
		expect(fadeDurations(animate)).toEqual([]);
		expect(sources()).toEqual(['/b.png']);
	});
});
