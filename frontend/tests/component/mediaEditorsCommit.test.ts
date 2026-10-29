// @vitest-environment jsdom
//
// Covers the hop the pure modules cannot: that the editors actually send the
// operations the geometry produced, to the right endpoint, in the mode the
// button the user pressed means. `toCropOperation` staying green proves nothing
// if the footer wires "Replace original" to the same call as "Save as new".
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// The crop stage measures itself with `bind:clientWidth`, which Svelte 5
// implements with a ResizeObserver jsdom does not provide.
class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

const startMediaToolRun = vi.fn();
const getMediaToolRun = vi.fn();
const cancelMediaToolRun = vi.fn();
const listUploads = vi.fn();
const listGenerationMedia = vi.fn();
const copyGenerationFileToLibrary = vi.fn();
const uploadMedia = vi.fn();
const deleteLibraryItem = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	startMediaToolRun: (...args: unknown[]) => startMediaToolRun(...args),
	getMediaToolRun: (...args: unknown[]) => getMediaToolRun(...args),
	cancelMediaToolRun: (...args: unknown[]) => cancelMediaToolRun(...args),
	listMediaTools: vi.fn(),
	previewMediaTool: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({
	api: {
		listUploads: (...args: unknown[]) => listUploads(...args),
		listGenerationMedia: (...args: unknown[]) => listGenerationMedia(...args),
		copyGenerationFileToLibrary: (...args: unknown[]) => copyGenerationFileToLibrary(...args),
		uploadMedia: (...args: unknown[]) => uploadMedia(...args),
		deleteLibraryItem: (...args: unknown[]) => deleteLibraryItem(...args)
	}
}));

const { default: MediaEditors } = await import('$lib/media/editors/MediaEditors.svelte');
const { mediaToolRuns } = await import('$lib/media-tools/runs');
const { createClassComponent } = await import('svelte/legacy');
const { tick } = await import('svelte');

function mount(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: MediaEditors as any, target, props });
	// Every editor renders through EditorShell -> BaseModal, which portals its
	// dialog straight onto document.body (src/lib/actions/portal.ts) so a
	// `position: fixed` overlay stays viewport-relative rather than being
	// captured by a transformed ancestor. Query from body, not `target`, or
	// the modal's own content is invisible to every helper below.
	return document.body;
}

function buttonByText(target: HTMLElement, text: string): HTMLButtonElement | undefined {
	return Array.from(target.querySelectorAll('button')).find(
		(button) => (button.textContent || '').trim() === text
	);
}

async function settle() {
	for (let i = 0; i < 5; i += 1) {
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

function completedRun(items: unknown[], replaced = false) {
	return { run_id: 'run-1', status: 'completed', progress: 1, label: '', items, replaced, error: null };
}

const imageSource = {
	url: '/api/media/uploads/portrait.png',
	kind: 'image' as const,
	fileName: 'portrait.png',
	itemId: 'row-1',
	storedPath: 'uploads/portrait.png',
	width: 1024,
	height: 1536
};

const clipSource = {
	url: '/api/media/uploads/clip.mp4',
	kind: 'video' as const,
	fileName: 'clip.mp4',
	itemId: 'row-2',
	storedPath: 'uploads/clip.mp4',
	width: 1280,
	height: 720,
	durationSeconds: 8.4,
	fps: 24
};

const audioSource = {
	url: '/api/media/uploads/track.mp3',
	kind: 'audio' as const,
	fileName: 'track.mp3',
	itemId: 'row-3',
	storedPath: 'uploads/track.mp3',
	durationSeconds: 65
};

afterEach(() => {
	mediaToolRuns.destroy();
});

beforeEach(() => {
	document.body.innerHTML = '';
	startMediaToolRun.mockReset();
	getMediaToolRun.mockReset();
	cancelMediaToolRun.mockReset();
	listUploads.mockReset();
	listGenerationMedia.mockReset();
	copyGenerationFileToLibrary.mockReset();
	uploadMedia.mockReset();
	deleteLibraryItem.mockReset();
	deleteLibraryItem.mockResolvedValue({ success: true });
	listGenerationMedia.mockResolvedValue({ success: false });
	copyGenerationFileToLibrary.mockResolvedValue({ success: false });
	uploadMedia.mockResolvedValue({
		success: true,
		data: { path: 'uploads/mask-1.png', relative_path: 'uploads/mask-1.png', filename: 'mask-1.png', size: 2, url: '/api/media/uploads/mask-1.png' }
	});
	startMediaToolRun.mockResolvedValue({ run_id: 'run-1' });
	getMediaToolRun.mockResolvedValue(
		completedRun([
			{ id: 'row-1', filename: 'edited.png', media_type: 'image', url: '/api/media/uploads/edited.png' }
		])
	);
	listUploads.mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } });
});

describe('crop editor', () => {
	it('sends the pixel rectangle the chosen aspect produced, in the source size', async () => {
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		// 1:1 on a 1024×1536 source is the widest square that fits, centred.
		buttonByText(target, '1:1')!.click();
		await settle();

		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'crop', params: { x: 0, y: 256, width: 1024, height: 1024 } }],
			inputs: [{ item_id: 'row-1' }],
			output: { mode: 'new' }
		});
	});

	it('wires Replace original to the replace mode, not to the same call', async () => {
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Replace original')!.click();
		await settle();
		expect(startMediaToolRun).not.toHaveBeenCalled();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun.mock.calls[0][0].output).toEqual({ mode: 'replace' });
	});

	it('turns rotate, flip and crop into one run of ordered steps', async () => {
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		const orientation = (label: string) =>
			Array.from(target.querySelectorAll('button')).find((b) => b.getAttribute('aria-label') === label)!;
		orientation('Rotate right').click();
		await settle();
		orientation('Flip horizontal').click();
		await settle();
		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledTimes(1);
		const { steps } = startMediaToolRun.mock.calls[0][0];
		expect(steps.map((step: { tool: string }) => step.tool)).toEqual(['rotate', 'flip', 'crop']);
		expect(steps[0].params).toEqual({ degrees: 90 });
		expect(steps[1].params).toEqual({ axis: 'horizontal' });
		expect(Object.keys(steps[2].params)).toEqual(['x', 'y', 'width', 'height']);
	});

	it('adds a resize step when a longest side is chosen', async () => {
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		buttonByText(target, '1024')?.click();
		buttonByText(target, '768')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		const { steps } = startMediaToolRun.mock.calls[0][0];
		expect(steps).toEqual([{ tool: 'resize', params: { height: 768 } }]);
	});

	it('sends new when Replace was picked while the file was still being looked up and turned out to be a generation', async () => {
		let resolveMedia: (value: unknown) => void = () => undefined;
		listGenerationMedia.mockReturnValue(new Promise((resolve) => (resolveMedia = resolve)));
		const generated = { ...imageSource, itemId: null, storedPath: 'outputs/2026-08-13/01KABC/0.png' };
		const target = mount({ request: { kind: 'crop', source: generated, itemIndex: null }, onClose: vi.fn(), onResult: vi.fn() });
		await settle();

		expect(buttonByText(target, 'Replace original')).toBeDefined();
		buttonByText(target, 'Replace original')!.click();
		await settle();
		resolveMedia({ success: true, data: { media: [{ id: 'file-7', filename: '0.png' }] } });
		await settle();
		expect(buttonByText(target, 'Replace original')).toBeUndefined();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun.mock.calls[0][0].inputs).toEqual([{ generation_file_id: 'file-7' }]);
		expect(startMediaToolRun.mock.calls[0][0].output).toEqual({ mode: 'new' });
	});

	it('refuses to send an untouched plan, which the server rejects as empty', async () => {
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		expect(target.textContent).toContain('Nothing to apply yet.');
		expect(buttonByText(target, 'Apply')!.disabled).toBe(true);
		expect(startMediaToolRun).not.toHaveBeenCalled();
	});

	it('hands the result back and closes', async () => {
		const onResult = vi.fn();
		const onClose = vi.fn();
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: 3 },
			onClose,
			onResult
		});
		await settle();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(onResult).toHaveBeenCalledWith(
			{ type: 'item', item: expect.objectContaining({ filename: 'edited.png' }), replaced: false },
			expect.objectContaining({ itemIndex: 3 })
		);
		expect(onClose).toHaveBeenCalled();
	});

	it('surfaces the server refusal instead of "status code 400"', async () => {
		startMediaToolRun.mockRejectedValue({
			message: 'Request failed with status code 400',
			response: {
				data: { detail: { error: 'invalid_params', message: 'The crop rectangle does not fit inside 1024x1536' } }
			}
		});
		const onClose = vi.fn();
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose,
			onResult: vi.fn()
		});
		await settle();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(target.textContent).toContain('The crop rectangle does not fit inside 1024x1536');
		expect(target.textContent).not.toContain('status code');
		expect(onClose).not.toHaveBeenCalled();
	});
});

describe('trim editor', () => {
	function nudgeOut(target: HTMLElement) {
		const handle = Array.from(target.querySelectorAll('button')).find(
			(button) => button.getAttribute('aria-label') === 'Trim out point'
		);
		handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
	}

	it('sends one trim operation, clamped to the clip', async () => {
		const target = mount({
			request: { kind: 'trim', source: clipSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		nudgeOut(target);
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledTimes(1);
		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'trim', params: { start: 0, end: 8.3 } }],
			inputs: [{ item_id: 'row-2' }],
			output: { mode: 'new' }
		});
	});

	it('refuses a selection that is still the whole clip', async () => {
		const target = mount({
			request: { kind: 'trim', source: clipSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		expect(target.textContent).toContain('Move a handle to choose what to keep.');
		expect(buttonByText(target, 'Apply')!.disabled).toBe(true);
	});

	it('explains itself rather than offering a save when nothing can be prepared', async () => {
		listGenerationMedia.mockResolvedValue({ success: true, data: { media: [] } });

		const target = mount({
			request: {
				kind: 'trim',
				source: {
					...clipSource,
					itemId: null,
					storedPath: 'outputs/2026-08-13/01KABC/0.mp4'
				},
				itemIndex: null
			},
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		expect(target.textContent).toContain('not in your library');
		expect(buttonByText(target, 'Apply')!.disabled).toBe(true);
		expect(startMediaToolRun).not.toHaveBeenCalled();
	});

	it('resolves the library row from the stored path when the host did not supply one', async () => {
		listUploads.mockResolvedValue({
			success: true,
			data: {
				uploads: [
					{ id: 'other', filename: 'not-it.mp4' },
					{ id: 'row-77', filename: 'clip.mp4' }
				],
				total: 2,
				limit: 100,
				offset: 0
			}
		});
		const target = mount({
			request: { kind: 'trim', source: { ...clipSource, itemId: null }, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		nudgeOut(target);
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(listUploads).toHaveBeenCalledWith(expect.objectContaining({ mediaType: 'video' }));
		expect(startMediaToolRun.mock.calls[0][0].inputs).toEqual([{ item_id: 'row-77' }]);
	});
});

describe('generated media', () => {
	const generatedPath = 'outputs/2026-08-13/01KABC/0.png';
	const cases = [
		{
			kind: 'crop' as const,
			source: { ...imageSource, itemId: null, storedPath: generatedPath },
			filename: '0.png',
			prepare: async (target: HTMLElement) => {
				buttonByText(target, '1:1')!.click();
				await settle();
			},
			save: 'Apply'
		},
		{
			kind: 'trim' as const,
			source: { ...clipSource, itemId: null, storedPath: 'outputs/2026-08-13/01KABC/0.mp4' },
			filename: '0.mp4',
			prepare: async (target: HTMLElement) => {
				const handle = Array.from(target.querySelectorAll('button')).find(
					(button) => button.getAttribute('aria-label') === 'Trim out point'
				);
				handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
				await settle();
			},
			save: 'Apply'
		},
		{
			kind: 'frame' as const,
			source: { ...clipSource, itemId: null, storedPath: 'outputs/2026-08-13/01KABC/0.mp4' },
			filename: '0.mp4',
			prepare: async () => {},
			save: 'Save frame'
		}
	];

	describe.each(cases)('$kind', ({ kind, source, filename, prepare, save }) => {
		function open(onClose = vi.fn(), onResult = vi.fn()) {
			listGenerationMedia.mockResolvedValue({
				success: true,
				data: { media: [{ id: 'file-7', filename }] }
			});
			return mount({ request: { kind, source, itemIndex: null }, onClose, onResult });
		}

		it('adds nothing to the library when opened and cancelled', async () => {
			const onClose = vi.fn();
			const target = open(onClose);
			await settle();

			expect(target.textContent).toContain('Saved as a new Library item. The generation stays as it is.');
			buttonByText(target, 'Cancel')!.click();
			await settle();

			expect(onClose).toHaveBeenCalled();
			expect(startMediaToolRun).not.toHaveBeenCalled();
			expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
			expect(uploadMedia).not.toHaveBeenCalled();
		});

		it('runs the edit on the generation file directly, as a new item, with no draft copy', async () => {
			const target = open();
			await settle();
			await prepare(target);

			buttonByText(target, save)!.click();
			await settle();

			expect(startMediaToolRun).toHaveBeenCalledTimes(1);
			const body = startMediaToolRun.mock.calls[0][0];
			expect(body.inputs).toEqual([{ generation_file_id: 'file-7' }]);
			expect(body.output).toEqual({ mode: 'new' });
			expect(body.steps[0].tool).toBe(kind === 'crop' ? 'crop' : kind === 'trim' ? 'trim' : 'frame');
			expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
			expect(deleteLibraryItem).not.toHaveBeenCalled();
		});

		it('offers no replace for a generated file', async () => {
			const target = open();
			await settle();
			expect(buttonByText(target, 'Replace original')).toBeUndefined();
		});

		it('keeps the editor open and shows why when the edit fails', async () => {
			startMediaToolRun.mockRejectedValue(new Error('The edit failed'));
			const onClose = vi.fn();
			const target = open(onClose);
			await settle();
			await prepare(target);

			buttonByText(target, save)!.click();
			await settle();

			expect(target.textContent).toContain('The edit failed');
			expect(onClose).not.toHaveBeenCalled();
			expect(deleteLibraryItem).not.toHaveBeenCalled();
		});

		it('never copies a library item', async () => {
			const libraryItem = kind === 'crop' ? imageSource : clipSource;
			const target = mount({
				request: { kind, source: libraryItem, itemIndex: null },
				onClose: vi.fn(),
				onResult: vi.fn()
			});
			await settle();
			await prepare(target);
			buttonByText(target, save)!.click();
			await settle();

			expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
			expect(deleteLibraryItem).not.toHaveBeenCalled();
			expect(target.textContent).not.toContain('Saved as a new Library item');
		});
	});
});

describe('split editor', () => {
	beforeEach(() => {
		getMediaToolRun.mockResolvedValue(
			completedRun([
				{ id: 'row-10', filename: 'track-part-1.mp3', media_type: 'audio', url: '/api/media/uploads/track-part-1.mp3' },
				{ id: 'row-11', filename: 'track-part-2.mp3', media_type: 'audio', url: '/api/media/uploads/track-part-2.mp3' }
			])
		);
	});

	function partLengthInput(target: HTMLElement): HTMLInputElement {
		return target.querySelector('#split-part-seconds') as HTMLInputElement;
	}

	function setPartLength(target: HTMLElement, value: string) {
		const input = partLengthInput(target);
		input.value = value;
		input.dispatchEvent(new Event('input', { bubbles: true }));
	}

	it('sends the part length in seconds to the split endpoint, never the edit endpoint', async () => {
		const target = mount({
			request: { kind: 'split', source: audioSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		// The default part length is 10s on a 65s clip - a valid split.
		buttonByText(target, 'Split')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'split', params: { part_seconds: 10 } }],
			inputs: [{ item_id: 'row-3' }],
			output: { mode: 'new' }
		});
	});

	it('previews the part count, remainder clip included, recomputed as the user types', async () => {
		const target = mount({
			request: { kind: 'split', source: audioSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		// 65s at 10s a part is 6 full parts plus a 5s remainder - 7 parts, not 6.
		expect(target.textContent).toContain('7 parts (6 × 10s + 5s)');

		setPartLength(target, '13');
		await settle();

		// 65 / 13 divides evenly - no remainder clip, and no "+ 0s" in the text.
		expect(target.textContent).toContain('5 parts (5 × 13s)');
		expect(target.textContent).not.toContain('+ 0s');
	});

	it('disables the action once the part length reaches the clip length', async () => {
		const target = mount({
			request: { kind: 'split', source: audioSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		setPartLength(target, '65');
		await settle();

		expect(buttonByText(target, 'Split')!.disabled).toBe(true);
		expect(target.textContent).toContain('nothing to split');

		buttonByText(target, 'Split')!.click();
		await settle();
		expect(startMediaToolRun).not.toHaveBeenCalled();
	});

	it('hands back every part the split produced, not just one', async () => {
		const onResult = vi.fn();
		const onClose = vi.fn();
		const target = mount({
			request: { kind: 'split', source: audioSource, itemIndex: null },
			onClose,
			onResult
		});
		await settle();

		buttonByText(target, 'Split')!.click();
		await settle();

		expect(onResult).toHaveBeenCalledWith(
			{
				type: 'items',
				items: [
					expect.objectContaining({ filename: 'track-part-1.mp3' }),
					expect.objectContaining({ filename: 'track-part-2.mp3' })
				]
			},
			expect.objectContaining({ itemIndex: null })
		);
		expect(onClose).toHaveBeenCalled();
	});
});

describe('mask editor', () => {
	it('needs no library resource, so opening one looks nothing up', async () => {
		// A mask is painted over the media and stored beside it; it changes no
		// row, so it must not pay for - or be blocked by - a resource lookup.
		mount({
			request: { kind: 'mask', source: { ...imageSource, itemId: null }, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		expect(listUploads).not.toHaveBeenCalled();
		expect(listGenerationMedia).not.toHaveBeenCalled();
		expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
	});
});

describe('frame editor', () => {
	beforeEach(() => {
		getMediaToolRun.mockResolvedValue(
			completedRun([{ id: 'row-9', filename: 'still.png', media_type: 'image', url: '/api/media/uploads/still.png' }])
		);
	});

	it('runs the frame tool and never offers a replace', async () => {
		const target = mount({
			request: { kind: 'frame', source: clipSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		// A still cannot take a clip's place - every reference to the row was
		// given a video - so the choice collapses to one button.
		expect(buttonByText(target, 'Replace original')).toBeUndefined();

		buttonByText(target, 'Save frame')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'frame', params: { time: 0 } }],
			inputs: [{ item_id: 'row-2' }],
			output: { mode: 'new' }
		});
	});

	it('never asks for the duration itself, which the server treats as past the end', async () => {
		const target = mount({
			request: { kind: 'frame', source: clipSource, itemIndex: null },
			onClose: vi.fn(),
			onResult: vi.fn()
		});
		await settle();

		const rail = target.querySelector('[role="presentation"]') as HTMLElement;
		rail.dispatchEvent(
			new PointerEvent('pointerdown', { button: 0, clientX: 99999, bubbles: true })
		);
		await settle();
		buttonByText(target, 'Save frame')!.click();
		await settle();

		const requestedTime = startMediaToolRun.mock.calls[0][0].steps[0].params.time;
		expect(requestedTime).toBeLessThan(8.4);
	});
});

describe('a run in flight', () => {
	function running(progress: number) {
		return { run_id: 'run-1', status: 'running', progress, label: 'Encoding', items: [], replaced: false, error: null };
	}

	function openCrop(overrides: Record<string, unknown> = {}) {
		const onClose = vi.fn();
		const onResult = vi.fn();
		const target = mount({
			request: { kind: 'crop', source: imageSource, itemIndex: null },
			onClose,
			onResult,
			...overrides
		});
		return { target, onClose, onResult };
	}

	async function pressSave(target: HTMLElement) {
		await settle();
		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();
	}

	it('shows the progress the server reports and keeps the editor open', async () => {
		getMediaToolRun.mockResolvedValue(running(0.4));
		const { target, onClose } = openCrop();

		await pressSave(target);

		expect(target.querySelector('[data-tool-progress]')?.textContent).toContain('40%');
		expect(onClose).not.toHaveBeenCalled();
	});

	it('cancels the run from the footer and goes back to the form without a result', async () => {
		getMediaToolRun.mockResolvedValue(running(0.4));
		cancelMediaToolRun.mockResolvedValue({ run_id: 'run-1', status: 'cancelled', progress: 0.4, label: '', items: [], replaced: false, error: null });
		const { target, onClose, onResult } = openCrop();
		await pressSave(target);

		buttonByText(target, 'Cancel edit')!.click();
		await settle();

		expect(cancelMediaToolRun).toHaveBeenCalledWith('run-1');
		expect(onResult).not.toHaveBeenCalled();
		expect(onClose).not.toHaveBeenCalled();
		expect(target.querySelector('[data-tool-progress]')).toBeNull();
		expect(buttonByText(target, 'Apply')!.disabled).toBe(false);
	});

	it('hands the result over when the cancel is answered with a run that had already completed', async () => {
		getMediaToolRun.mockResolvedValue(running(0.95));
		cancelMediaToolRun.mockResolvedValue({
			run_id: 'run-1',
			status: 'completed',
			progress: 1,
			label: '',
			items: [{ id: 'row-1', filename: 'late.png', media_type: 'image', url: '/late.png' }],
			replaced: true,
			error: null
		});
		const { target, onClose, onResult } = openCrop();
		await pressSave(target);

		buttonByText(target, 'Cancel edit')!.click();
		await settle();

		expect(onResult).toHaveBeenCalledWith(
			{ type: 'item', item: expect.objectContaining({ filename: 'late.png' }), replaced: true },
			expect.anything()
		);
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('shows the reason a run failed on the server and stays open', async () => {
		getMediaToolRun.mockResolvedValue({
			...running(0.2),
			status: 'failed',
			error: { code: 'ffmpeg_failed', message: 'The encoder stopped' }
		});
		const { target, onClose, onResult } = openCrop();

		await pressSave(target);

		expect(target.textContent).toContain('The encoder stopped');
		expect(onClose).not.toHaveBeenCalled();
		expect(onResult).not.toHaveBeenCalled();
	});

	it('closes at once when detached and hands the result over when the run finishes', async () => {
		let finish: (run: unknown) => void = () => undefined;
		getMediaToolRun
			.mockResolvedValueOnce(running(0.3))
			.mockImplementation(() => new Promise((resolve) => (finish = resolve)));
		const { target, onClose, onResult } = openCrop({ detach: true });

		await pressSave(target);

		expect(onClose).toHaveBeenCalledTimes(1);
		expect(onResult).not.toHaveBeenCalled();

		await new Promise((resolve) => setTimeout(resolve, 450));
		finish(completedRun([{ id: 'row-5', filename: 'late.png', media_type: 'image', url: '/late.png' }]));
		await settle();

		expect(onResult).toHaveBeenCalledWith(
			{ type: 'item', item: expect.objectContaining({ filename: 'late.png' }), replaced: false },
			expect.objectContaining({ itemIndex: null })
		);
	});

	it('keeps the thumbnails the server sent on the item it hands back', async () => {
		const item = {
			id: 'row-1',
			filename: 'a.png',
			media_type: 'image',
			url: '/a.png',
			thumbnail_small: '/t/s.webp',
			thumbnail_medium: '/t/m.webp',
			thumbnail_large: '/t/l.webp'
		};
		getMediaToolRun.mockResolvedValue(completedRun([item], false));
		const { target, onResult } = openCrop();
		await pressSave(target);

		expect(onResult).toHaveBeenCalledWith({ type: 'item', item, replaced: false }, expect.anything());
	});

	it('reports a replaced original as replaced', async () => {
		getMediaToolRun.mockResolvedValue(
			completedRun([{ id: 'row-1', filename: 'a.png', media_type: 'image', url: '/a.png' }], true)
		);
		const { target, onResult } = openCrop();
		await settle();
		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Replace original')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(onResult).toHaveBeenCalledWith(expect.objectContaining({ replaced: true }), expect.anything());
	});
});

describe('the footer every editor shares with the media tool dialog', () => {
	const sources = {
		crop: { kind: 'crop', source: imageSource, apply: 'Apply', prepare: async (target: HTMLElement) => { buttonByText(target, '1:1')!.click(); await settle(); } },
		trim: {
			kind: 'trim',
			source: clipSource,
			apply: 'Apply',
			prepare: async (target: HTMLElement) => {
				const handle = Array.from(target.querySelectorAll('button')).find((b) => b.getAttribute('aria-label') === 'Trim out point');
				handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
				await settle();
			}
		},
		resize: { kind: 'resize', source: clipSource, apply: 'Apply', prepare: async () => undefined },
		frame: { kind: 'frame', source: clipSource, apply: 'Save frame', prepare: async () => undefined },
		split: { kind: 'split', source: audioSource, apply: 'Split', prepare: async () => undefined }
	} as const;

	function pickChip(target: HTMLElement, label: string) {
		const chip = Array.from(target.querySelectorAll('button')).find((b) => (b.textContent || '').trim().startsWith(label));
		chip!.click();
	}

	function footer(target: HTMLElement) {
		return target.querySelector('[data-tool-footer]') as HTMLElement;
	}

	function seg(target: HTMLElement) {
		return footer(target).querySelector('[role="group"][aria-label="Save as"]');
	}

	function open(kind: keyof typeof sources) {
		const { source } = sources[kind];
		return mount({ request: { kind, source, itemIndex: null }, onClose: vi.fn(), onResult: vi.fn() });
	}

	it.each(['crop', 'trim', 'resize'] as const)('puts the Save as new and Replace original choice on the left of %s, with Cancel and Apply on the right', async (kind) => {
		const target = open(kind);
		await settle();

		const group = seg(target)!;
		expect(group).not.toBeNull();
		expect(Array.from(group.querySelectorAll('button')).map((b) => [b.textContent?.trim(), b.getAttribute('aria-pressed')])).toEqual([
			['Save as new', 'true'],
			['Replace original', 'false']
		]);
		const right = Array.from(footer(target).querySelectorAll('.ml-auto button')).map((b) => b.textContent?.trim());
		expect(right).toEqual(['Cancel', 'Apply']);
		expect(group.compareDocumentPosition(footer(target).querySelector('.ml-auto')!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it.each(['crop', 'trim', 'resize'] as const)('sends the mode picked on the %s footer when Apply is pressed', async (kind) => {
		const target = open(kind);
		await settle();
		await sources[kind].prepare(target);
		if (kind === 'resize') {
			pickChip(target, '540p');
			await settle();
		}

		buttonByText(target, 'Replace original')!.click();
		await settle();
		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun.mock.calls[0][0].output).toEqual({ mode: 'replace' });
	});

	it.each(['crop', 'trim', 'resize'] as const)('starts %s on Save as new', async (kind) => {
		const target = open(kind);
		await settle();
		await sources[kind].prepare(target);
		if (kind === 'resize') {
			pickChip(target, '540p');
			await settle();
		}
		buttonByText(target, 'Apply')!.click();
		await settle();
		expect(startMediaToolRun.mock.calls[0][0].output).toEqual({ mode: 'new' });
	});

	it.each(['frame', 'split'] as const)('keeps the choice out of %s, which can only make new items, and names its own action', async (kind) => {
		const target = open(kind);
		await settle();

		expect(seg(target)).toBeNull();
		expect(buttonByText(target, 'Replace original')).toBeUndefined();
		expect(buttonByText(target, sources[kind].apply)).toBeDefined();
		expect(buttonByText(target, 'Apply')).toBeUndefined();
		expect(Array.from(footer(target).querySelectorAll('.ml-auto button')).map((b) => b.textContent?.trim())).toEqual(['Cancel', sources[kind].apply]);
	});

	it.each(['crop', 'trim', 'resize', 'frame', 'split'] as const)('swaps the choice for the Library note on %s when the file is a generation', async (kind) => {
		const { source } = sources[kind];
		const generated = { ...source, itemId: null, storedPath: `outputs/2026-08-13/01KABC/0.${source.kind === 'image' ? 'png' : source.kind === 'video' ? 'mp4' : 'mp3'}` };
		listGenerationMedia.mockResolvedValue({
			success: true,
			data: { media: [{ id: 'file-7', filename: `0.${source.kind === 'image' ? 'png' : source.kind === 'video' ? 'mp4' : 'mp3'}` }] }
		});
		const target = mount({ request: { kind, source: generated, itemIndex: null }, onClose: vi.fn(), onResult: vi.fn() });
		await settle();

		expect(seg(target)).toBeNull();
		expect(footer(target).querySelector('[data-tool-note]')?.textContent).toBe('Saved as a new Library item. The generation stays as it is.');
		expect(target.textContent!.split('Saved as a new Library item. The generation stays as it is.')).toHaveLength(2);
	});

	it.each(['crop', 'trim', 'resize', 'frame', 'split'] as const)('shows the field note and no choice on %s when the editor is opened for a field', async (kind) => {
		const { source } = sources[kind];
		const target = mount({ request: { kind, source, itemIndex: null }, scope: 'field', onClose: vi.fn(), onResult: vi.fn() });
		await settle();

		const word = source.kind === 'audio' ? 'clip' : source.kind;
		expect(seg(target)).toBeNull();
		expect(footer(target).querySelector('[data-tool-note]')?.textContent).toBe(
			`The result replaces this ${word} in the field. The library copy stays as it is.`
		);
		expect(buttonByText(target, 'Replace original')).toBeUndefined();
	});

	it('names the field note after a generated file too, since the field is where the result goes', async () => {
		listGenerationMedia.mockResolvedValue({ success: true, data: { media: [{ id: 'file-7', filename: '0.png' }] } });
		const generated = { ...imageSource, itemId: null, storedPath: 'outputs/2026-08-13/01KABC/0.png' };
		const target = mount({ request: { kind: 'crop', source: generated, itemIndex: null }, scope: 'field', onClose: vi.fn(), onResult: vi.fn() });
		await settle();
		expect(footer(target).querySelector('[data-tool-note]')?.textContent).toBe(
			'The result replaces this image in the field. The library copy stays as it is.'
		);
	});

	it.each(['crop', 'trim', 'resize', 'frame', 'split'] as const)('shows the footer text of a refusal or a blocked save in the %s footer', async (kind) => {
		startMediaToolRun.mockRejectedValue(new Error('The edit failed'));
		const target = open(kind);
		await settle();
		await sources[kind].prepare(target);
		if (kind === 'resize') {
			pickChip(target, '540p');
			await settle();
		}

		buttonByText(target, sources[kind].apply)!.click();
		await settle();

		expect(footer(target).querySelector('[data-tool-failure]')?.textContent).toBe('The edit failed');
	});

	it.each(['crop', 'trim', 'resize', 'frame', 'split'] as const)('shows progress and a Cancel edit button in the %s footer while its run goes', async (kind) => {
		getMediaToolRun.mockResolvedValue({ run_id: 'run-1', status: 'running', progress: 0.25, label: 'Encoding', items: [], replaced: false, error: null });
		const target = open(kind);
		await settle();
		await sources[kind].prepare(target);
		if (kind === 'resize') {
			pickChip(target, '540p');
			await settle();
		}
		if (kind === 'split') {
			const input = target.querySelector('#split-part-seconds') as HTMLInputElement;
			input.value = '10';
			input.dispatchEvent(new Event('input', { bubbles: true }));
			await settle();
		}

		buttonByText(target, sources[kind].apply)!.click();
		await settle();

		expect(footer(target).querySelector('[data-tool-progress]')?.textContent).toContain('25%');
		expect(buttonByText(target, 'Cancel edit')).toBeDefined();
		expect(seg(target)).toBeNull();
	});
});
