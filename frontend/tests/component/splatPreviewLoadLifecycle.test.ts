import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

interface FakeViewer {
	load: ReturnType<typeof vi.fn>;
	clear: ReturnType<typeof vi.fn>;
	resetView: ReturnType<typeof vi.fn>;
	setAutoRotate: ReturnType<typeof vi.fn>;
	dispose: ReturnType<typeof vi.fn>;
	pending: Array<{
		url: string;
		format: string | null;
		signal: AbortSignal;
		resolve: (value: { splatCount: number }) => void;
		reject: (reason: unknown) => void;
	}>;
}

const hoisted = vi.hoisted(() => ({
	created: [] as unknown[],
	createOptions: [] as Array<{ container: HTMLElement; onContextLost?: () => void }>,
	failCreate: null as unknown
}));

vi.mock('$lib/components/workbench/renderers/splat/splatViewer', () => ({
	createSplatViewer: (options: { container: HTMLElement; onContextLost?: () => void }) => {
		if (hoisted.failCreate) throw hoisted.failCreate;
		hoisted.createOptions.push(options);
		const viewer: FakeViewer = {
			pending: [],
			load: vi.fn(
				(url: string, format: string | null, signal: AbortSignal) =>
					new Promise<{ splatCount: number }>((resolve, reject) => {
						viewer.pending.push({ url, format, signal, resolve, reject });
					})
			),
			clear: vi.fn(),
			resetView: vi.fn(),
			setAutoRotate: vi.fn(),
			dispose: vi.fn()
		};
		hoisted.created.push(viewer);
		return viewer;
	}
}));

const { createClassComponent } = await import('svelte/legacy');
const { default: SplatPreview } = await import(
	'$lib/components/workbench/renderers/SplatPreview.svelte'
);
const { SplatLoadError } = await import(
	'$lib/components/workbench/renderers/splat/splatSupport'
);

async function settle() {
	for (let i = 0; i < 8; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

function viewers(): FakeViewer[] {
	return hoisted.created as FakeViewer[];
}

function mount(file: Record<string, unknown> | null) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: SplatPreview as never,
		target,
		props: { file }
	});
	return {
		target,
		component: component as { $set: (props: Record<string, unknown>) => void; $destroy: () => void },
		alertEl: () => target.querySelector('[role="alert"]'),
		spinnerEl: () => target.querySelector('[role="status"].animate-spin'),
		resetButton: () => target.querySelector<HTMLButtonElement>('button[aria-label="Reset view"]'),
		rotateButton: () =>
			target.querySelector<HTMLButtonElement>('button[aria-label="Toggle auto-rotate"]'),
		viewerBox: () => target.querySelector<HTMLElement>('.splat-viewer'),
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mount> | undefined;

beforeEach(() => {
	hoisted.created.length = 0;
	hoisted.createOptions.length = 0;
	hoisted.failCreate = null;
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	vi.useRealTimers();
});

describe('SplatPreview loading lifecycle', () => {
	it('shows the empty state and never starts a viewer when the file has no resolvable url', async () => {
		mounted = mount({});
		await settle();

		expect(mounted.target.textContent).toContain('No scene');
		expect(mounted.spinnerEl()).toBeNull();
		expect(viewers()).toHaveLength(0);
	});

	it('starts one viewer in its container and asks it to load the url with the format from the name', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.spz' });
		await settle();

		expect(viewers()).toHaveLength(1);
		expect(hoisted.createOptions[0].container).toBe(mounted.viewerBox());
		const [request] = viewers()[0].pending;
		expect(request.url).toBe('/api/media/generations/g/0.spz');
		expect(request.format).toBe('spz');
		expect(request.signal.aborted).toBe(false);
		expect(mounted.spinnerEl()).not.toBeNull();
		expect(mounted.target.textContent).toContain('Loading scene');
		expect(mounted.viewerBox()?.getAttribute('style')).toContain('opacity: 0');
		expect(mounted.resetButton()).toBeNull();
	});

	it('prefers the format the server sent over the one in the name', async () => {
		mounted = mount({ url: '/api/media/generations/g/scene', splat_format: 'ply' });
		await settle();

		expect(viewers()[0].pending[0].format).toBe('ply');
	});

	it('passes a null format when nothing says what the container is, leaving the sniffing to the viewer', async () => {
		mounted = mount({ url: '/api/media/generations/g/scene' });
		await settle();

		expect(viewers()[0].pending[0].format).toBeNull();
	});

	it('becomes ready, reveals the viewer and its controls, and reports the splat count it loaded', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply', splat_name: '0.ply' });
		await settle();

		viewers()[0].pending[0].resolve({ splatCount: 250000 });
		await settle();

		expect(mounted.spinnerEl()).toBeNull();
		expect(mounted.viewerBox()?.getAttribute('style')).toContain('opacity: 1');
		expect(mounted.resetButton()).not.toBeNull();
		expect(mounted.target.textContent).toContain(`${(250000).toLocaleString()} splats`);
		expect(mounted.target.textContent).toContain('0.ply');
	});

	it('shows a readable error and no spinner when the file cannot be fetched', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();

		viewers()[0].pending[0].reject(new SplatLoadError('not-found', 404));
		await settle();

		expect(mounted.alertEl()?.textContent).toContain('could not be found');
		expect(mounted.spinnerEl()).toBeNull();
		expect(mounted.resetButton()).toBeNull();
	});

	it('shows a readable error when the viewer rejects the file as unreadable', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();

		viewers()[0].pending[0].reject(new SplatLoadError('decode'));
		await settle();

		expect(mounted.alertEl()?.textContent).toContain('may be corrupted');
	});

	it('re-enters loading when the file swaps, aborts the first request and ignores its late answer', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		const viewer = viewers()[0];
		viewer.pending[0].resolve({ splatCount: 10 });
		await settle();
		expect(mounted.resetButton()).not.toBeNull();

		mounted.component.$set({ file: { url: '/api/media/generations/g/1.ply' } });
		await settle();

		expect(viewers()).toHaveLength(1);
		expect(viewer.load).toHaveBeenCalledTimes(2);
		expect(viewer.pending[1].url).toBe('/api/media/generations/g/1.ply');
		expect(mounted.spinnerEl()).not.toBeNull();

		mounted.component.$set({ file: { url: '/api/media/generations/g/2.ply' } });
		await settle();
		expect(viewer.pending[1].signal.aborted).toBe(true);

		viewer.pending[1].resolve({ splatCount: 999 });
		await settle();
		expect(mounted.spinnerEl()).not.toBeNull();
		expect(mounted.target.textContent).not.toContain('999');

		viewer.pending[2].resolve({ splatCount: 3 });
		await settle();
		expect(mounted.spinnerEl()).toBeNull();
		expect(mounted.target.textContent).toContain('3 splats');
	});

	it('recovers from an error when the next file loads', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		viewers()[0].pending[0].reject(new SplatLoadError('network'));
		await settle();
		expect(mounted.alertEl()).not.toBeNull();

		mounted.component.$set({ file: { url: '/api/media/generations/g/1.ply' } });
		await settle();
		expect(mounted.alertEl()).toBeNull();
		viewers()[0].pending[1].resolve({ splatCount: 5 });
		await settle();

		expect(mounted.resetButton()).not.toBeNull();
	});

	it('clears the scene and falls back to the empty state when the url goes away', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		viewers()[0].pending[0].resolve({ splatCount: 10 });
		await settle();

		mounted.component.$set({ file: {} });
		await settle();

		expect(viewers()[0].clear).toHaveBeenCalled();
		expect(mounted.target.textContent).toContain('No scene');
		expect(mounted.resetButton()).toBeNull();
	});

	it('gives up with a timeout message when the load never ends, and aborts it', async () => {
		vi.useFakeTimers();
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(0);
		const [request] = viewers()[0].pending;

		await vi.advanceTimersByTimeAsync(60001);
		expect(request.signal.aborted).toBe(true);
		request.reject(new DOMException('Aborted', 'AbortError'));
		await vi.advanceTimersByTimeAsync(0);

		expect(mounted.alertEl()?.textContent).toContain('took too long');
	});

	it('forwards the toolbar buttons to the viewer', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		const viewer = viewers()[0];
		viewer.pending[0].resolve({ splatCount: 10 });
		await settle();

		mounted.resetButton()!.click();
		expect(viewer.resetView).toHaveBeenCalledTimes(1);

		mounted.rotateButton()!.click();
		expect(viewer.setAutoRotate).toHaveBeenLastCalledWith(true);
		mounted.rotateButton()!.click();
		expect(viewer.setAutoRotate).toHaveBeenLastCalledWith(false);
	});

	it('shows the WebGL message when the viewer cannot be created', async () => {
		hoisted.failCreate = new SplatLoadError('webgl');
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();

		expect(mounted.alertEl()?.textContent).toContain('WebGL');
		expect(mounted.spinnerEl()).toBeNull();
	});

	it('turns a lost graphics context into an error', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		viewers()[0].pending[0].resolve({ splatCount: 10 });
		await settle();

		hoisted.createOptions[0].onContextLost?.();
		await settle();

		expect(mounted.alertEl()?.textContent).toContain('graphics context was lost');
	});

	it('disposes the viewer and aborts the request on destroy', async () => {
		mounted = mount({ url: '/api/media/generations/g/0.ply' });
		await settle();
		const viewer = viewers()[0];
		const [request] = viewer.pending;

		mounted.destroy();
		mounted = undefined;

		expect(viewer.dispose).toHaveBeenCalledTimes(1);
		expect(request.signal.aborted).toBe(true);
	});
});
