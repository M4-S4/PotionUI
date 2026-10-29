// @vitest-environment jsdom
//
// The field end of the editor seam. The pure modules prove the geometry and
// `mediaLoaderEdited` proves the value shape; what neither can prove is that
// pressing a tool in the toolbar reaches an editor, that the editor reaches the
// API, and that what lands back in `onChange` is the SERVER's url rather than a
// blob handle - which dies with the document and 404s after a refresh.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CORE_TOOLS } from './fixtures/mediaToolCatalog';

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

const startMediaToolRun = vi.fn();
const getMediaToolRun = vi.fn();
const listMediaTools = vi.fn();
const listUploads = vi.fn();
const listGenerationMedia = vi.fn();
const copyGenerationFileToLibrary = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: (...args: unknown[]) => listMediaTools(...args),
	startMediaToolRun: (...args: unknown[]) => startMediaToolRun(...args),
	getMediaToolRun: (...args: unknown[]) => getMediaToolRun(...args),
	cancelMediaToolRun: vi.fn(),
	previewMediaTool: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({
	api: {
		listGenerationMedia: (...args: unknown[]) => listGenerationMedia(...args),
		copyGenerationFileToLibrary: (...args: unknown[]) => copyGenerationFileToLibrary(...args),
		getUploadInfo: vi.fn().mockResolvedValue({ success: false }),
		listUploads: (...args: unknown[]) => listUploads(...args),
		uploadMedia: vi.fn()
	}
}));

vi.mock('$lib/utils/storage', () => ({
	storage: { get: vi.fn().mockReturnValue(null), set: vi.fn(), remove: vi.fn() }
}));

const { default: MediaLoaderField } = await import(
	'$lib/components/form-fields/MediaLoaderField.svelte'
);
const { createClassComponent } = await import('svelte/legacy');
const { tick } = await import('svelte');
const { mediaToolRuns } = await import('$lib/media-tools/runs');
const { resetMediaToolCatalog } = await import('$lib/media-tools/catalog');

function mount(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: MediaLoaderField as any, target, props });
	// The shared editors render through EditorShell -> BaseModal, which portals
	// its dialog straight onto document.body (src/lib/actions/portal.ts) so a
	// `position: fixed` overlay stays viewport-relative rather than being
	// captured by a transformed ancestor. Query from body, not `target`, or
	// the modal's own content is invisible to every helper below.
	return document.body;
}

async function pickTool(target: HTMLElement, id: string) {
	target.querySelector<HTMLButtonElement>('[data-tools-trigger]')!.click();
	await tick();
	await tick();
	document.body.querySelector<HTMLButtonElement>(`[data-tools-menu] [data-tool="${id}"]`)!.click();
}

function buttonByText(target: HTMLElement, text: string): HTMLButtonElement | undefined {
	return Array.from(target.querySelectorAll('button')).find(
		(button) => (button.textContent || '').trim() === text
	);
}

async function settle() {
	for (let i = 0; i < 6; i += 1) {
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

function completedRun(item: Record<string, unknown>, replaced: boolean) {
	return { run_id: 'run-1', status: 'completed', progress: 1, label: '', items: [item], replaced, error: null };
}

const clip = {
	path: 'uploads/wan22_i2v_00042.mp4',
	relative_path: 'uploads/wan22_i2v_00042.mp4',
	url: '/api/media/uploads/wan22_i2v_00042.mp4',
	name: 'wan22_i2v_00042.mp4',
	type: 'video',
	metadata: { width: 1280, height: 720, duration_seconds: 5.2, fps: 24, size: 18874368 }
};

afterEach(() => {
	mediaToolRuns.destroy();
});

beforeEach(() => {
	document.body.innerHTML = '';
	resetMediaToolCatalog();
	listMediaTools.mockReset();
	listMediaTools.mockResolvedValue(CORE_TOOLS);
	startMediaToolRun.mockReset();
	getMediaToolRun.mockReset();
	startMediaToolRun.mockResolvedValue({ run_id: 'run-1' });
	listUploads.mockReset();
	listGenerationMedia.mockReset();
	copyGenerationFileToLibrary.mockReset();
	listGenerationMedia.mockResolvedValue({ success: false });
	copyGenerationFileToLibrary.mockResolvedValue({ success: false });
	listUploads.mockResolvedValue({
		success: true,
		data: {
			uploads: [{ id: 'row-42', filename: 'wan22_i2v_00042.mp4' }],
			total: 1,
			limit: 100,
			offset: 0
		}
	});
	getMediaToolRun.mockResolvedValue(
		completedRun(
			{
				id: 'row-42',
				filename: 'trimmed-9c.mp4',
				original_filename: 'wan22_i2v_00042.mp4',
				media_type: 'video',
				mime_type: 'video/mp4',
				url: '/api/media/uploads/trimmed-9c.mp4',
				width: 1280,
				height: 720,
				duration_seconds: 5.1,
				fps: 24,
				size: 18000000
			},
			true
		)
	);
});

describe('MediaLoaderField × the shared editors', () => {
	it('writes the edited resource back as the server url, never a blob handle', async () => {
		const onChange = vi.fn();
		const target = mount({
			name: 'clip',
			config: { title: 'Source video', accept: 'video/*' },
			value: clip,
			onChange
		});
		await settle();

		await pickTool(target, 'trim');
		await settle();

		const outHandle = Array.from(target.querySelectorAll('button')).find(
			(button) => button.getAttribute('aria-label') === 'Trim out point'
		);
		outHandle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		await settle();

		buttonByText(target, 'Replace original')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'trim', params: { start: 0, end: 5.1 } }],
			inputs: [{ item_id: 'row-42' }],
			output: { mode: 'replace' }
		});

		const written = onChange.mock.calls.at(-1)?.[1];
		expect(written).toMatchObject({
			relative_path: 'uploads/trimmed-9c.mp4',
			path: 'uploads/trimmed-9c.mp4',
			url: '/api/media/uploads/trimmed-9c.mp4',
			type: 'video'
		});
		// A replace lands on a NEW filename; a value left on the old one would
		// serve the pre-edit bytes out of the browser cache.
		expect(written.url).not.toContain('wan22_i2v_00042');
		expect(String(written.url).startsWith('blob:')).toBe(false);
	});

	it('closes the editor once the edit lands', async () => {
		const target = mount({
			name: 'clip',
			config: { title: 'Source video', accept: 'video/*' },
			value: clip,
			onChange: vi.fn()
		});
		await settle();

		await pickTool(target, 'trim');
		await settle();
		expect(buttonByText(target, 'Cancel')).toBeTruthy();

		const outHandle = Array.from(target.querySelectorAll('button')).find(
			(button) => button.getAttribute('aria-label') === 'Trim out point'
		);
		outHandle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		await settle();
		buttonByText(target, 'Save as new')!.click();
		await settle();

		expect(buttonByText(target, 'Cancel')).toBeUndefined();
	});

	it('offers the replace / save-as-new choice on a crop, same as the Library page', async () => {
		listUploads.mockResolvedValue({
			success: true,
			data: {
				uploads: [{ id: 'row-8', filename: 'portrait.png' }],
				total: 1,
				limit: 100,
				offset: 0
			}
		});
		const target = mount({
			name: 'reference_image',
			config: { title: 'Reference', accept: 'image/*' },
			value: {
				path: 'uploads/portrait.png',
				relative_path: 'uploads/portrait.png',
				url: '/api/media/uploads/portrait.png',
				name: 'portrait.png',
				type: 'image',
				metadata: { width: 1024, height: 1536, size: 2411724 }
			},
			onChange: vi.fn()
		});
		await settle();

		await pickTool(target, 'crop');
		await settle();

		// The same media must not behave differently depending on which screen
		// it was opened from.
		expect(buttonByText(target, 'Replace original')).toBeTruthy();
		expect(buttonByText(target, 'Save as new')).toBeTruthy();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Replace original')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'crop', params: { x: 0, y: 256, width: 1024, height: 1024 } }],
			inputs: [{ item_id: 'row-8' }],
			output: { mode: 'replace' }
		});
	});

	it('edits a generated image straight from storage and adds only the edited result to the library', async () => {
		listGenerationMedia.mockResolvedValue({
			success: true,
			data: { media: [{ id: 'file-3', filename: '0.png' }] }
		});
		getMediaToolRun.mockResolvedValue(
			completedRun(
				{
					id: 'new-1',
					filename: 'cropped.png',
					media_type: 'image',
					url: '/api/media/uploads/cropped.png'
				},
				false
			)
		);

		const onChange = vi.fn();
		const target = mount({
			name: 'reference_image',
			config: { title: 'Reference', accept: 'image/*' },
			value: {
				path: 'outputs/2026-08-13/01KABC/0.png',
				relative_path: 'outputs/2026-08-13/01KABC/0.png',
				url: '/api/media/generations/01KABC/0.png',
				name: '0.png',
				type: 'image',
				metadata: { width: 1024, height: 1536 }
			},
			onChange
		});
		await settle();

		await pickTool(target, 'crop');
		await settle();

		expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
		expect(target.textContent).toContain('Saves the edited copy to your Library');
		expect(buttonByText(target, 'Replace original')).toBeUndefined();

		buttonByText(target, '1:1')!.click();
		await settle();
		buttonByText(target, 'Save as new')!.click();
		await settle();

		expect(copyGenerationFileToLibrary).not.toHaveBeenCalled();
		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'crop', params: { x: 0, y: 256, width: 1024, height: 1024 } }],
			inputs: [{ generation_file_id: 'file-3' }],
			output: { mode: 'new' }
		});
		expect(onChange.mock.calls.at(-1)?.[1]).toMatchObject({
			relative_path: 'uploads/cropped.png',
			url: '/api/media/uploads/cropped.png'
		});
	});

	it('runs a catalog tool without an editor through the dialog and writes its result into the field', async () => {
		listUploads.mockResolvedValue({
			success: true,
			data: { uploads: [{ id: 'row-8', filename: 'portrait.png' }], total: 1, limit: 100, offset: 0 }
		});
		getMediaToolRun.mockResolvedValue(
			completedRun(
				{ id: 'row-8b', filename: 'turned.png', media_type: 'image', url: '/api/media/uploads/turned.png' },
				false
			)
		);
		const onChange = vi.fn();
		const target = mount({
			name: 'reference_image',
			config: { title: 'Reference', accept: 'image/*' },
			value: {
				path: 'uploads/portrait.png',
				relative_path: 'uploads/portrait.png',
				url: '/api/media/uploads/portrait.png',
				name: 'portrait.png',
				type: 'image',
				metadata: { width: 1024, height: 1536 }
			},
			onChange
		});
		await settle();

		await pickTool(target, 'rotate');
		await settle();

		expect(target.querySelector('[data-tool-dialog="rotate"]')).not.toBeNull();
		expect(target.textContent).toContain('The result replaces this image in the field. The library copy stays as it is.');
		expect(buttonByText(target, 'Replace original')).toBeUndefined();

		buttonByText(target, 'Apply')!.click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'rotate', params: {} }],
			inputs: [{ item_id: 'row-8' }],
			output: { mode: 'new' }
		});
		expect(onChange.mock.calls.at(-1)?.[1]).toMatchObject({ url: '/api/media/uploads/turned.png' });
		expect(target.querySelector('[data-tool-dialog="rotate"]')).toBeNull();
	});
});
