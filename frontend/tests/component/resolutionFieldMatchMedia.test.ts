import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount } from 'svelte';
import { writable } from 'svelte/store';
import ResolutionField from '$lib/components/form-fields/ResolutionField.svelte';
import { FORM_VALUES_CONTEXT_KEY } from '$lib/form/formValuesContext';

vi.mock('$lib/services/api/index', () => ({
	api: {
		listGenerationMedia: vi.fn(async () => ({ success: false })),
		getUploadInfo: vi.fn(async () => ({ success: false }))
	}
}));

const flush = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

let instance: ReturnType<typeof mount> | null = null;

function render(formValues: Record<string, unknown>, value: string | null = '1280x720') {
	const store = writable(formValues);
	const onChange = vi.fn();
	instance = mount(ResolutionField as never, {
		target: document.body,
		props: {
			name: 'size',
			config: { title: 'Size', options: [], match_media: 'input_video', default: '1280x720' },
			value,
			onChange
		},
		context: new Map([[FORM_VALUES_CONTEXT_KEY, store]])
	});
	return { store, onChange };
}

const trigger = () => document.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!;
const optionRows = () =>
	Array.from(document.body.querySelectorAll<HTMLButtonElement>('[role="option"]')).map((row) =>
		row.textContent!.replace(/\s+/g, ' ').trim()
	);

const portraitVideo = {
	path: '/data/uploads/clip.mp4',
	url: '/api/media/uploads/clip.mp4',
	type: 'video',
	metadata: { width: 1080, height: 1920 }
};

afterEach(() => {
	if (instance) unmount(instance);
	instance = null;
	document.body.innerHTML = '';
});

describe('ResolutionField matched to a video', () => {
	it('shows a placeholder and stays closed until a video is chosen', async () => {
		render({ input_video: null });
		await flush(10);

		expect(trigger().disabled).toBe(true);
		expect(trigger().textContent).toContain('Choose a video to see sizes');
		expect(trigger().textContent).not.toContain('1280');
	});

	it('lists sizes derived from the video and moves the default onto its shape', async () => {
		const { onChange } = render({ input_video: portraitVideo });
		await flush(20);

		expect(document.querySelector('[data-testid="media-source-size"]')?.textContent).toContain('1080×1920');
		expect(onChange).toHaveBeenCalledWith('size', '720x1280');

		trigger().click();
		await flush(20);
		const rows = optionRows();
		expect(rows[0]).toContain('Original');
		expect(rows[0]).toContain('1080×1920');
		expect(rows.some((row) => row.includes('720p') && row.includes('720×1280'))).toBe(true);
		expect(rows.some((row) => row.includes('25%') && row.includes('270×480'))).toBe(true);
		expect(rows.some((row) => row.includes('1920×1080'))).toBe(false);
	});

	it('keeps a custom size the user already chose for the same video', async () => {
		const { onChange } = render({ input_video: portraitVideo }, '500x900');
		await flush(20);

		expect(onChange).not.toHaveBeenCalled();
		expect(trigger().textContent).toContain('Custom');
	});

	it('fills the other side of a custom size from the video shape', async () => {
		render({ input_video: portraitVideo });
		await flush(20);
		trigger().click();
		await flush(20);
		Array.from(document.body.querySelectorAll('button'))
			.find((b) => b.textContent?.includes('Custom size'))!
			.click();
		await flush(20);

		const width = document.body.querySelector<HTMLInputElement>('input[aria-label="Custom width"]')!;
		width.value = '540';
		width.dispatchEvent(new Event('input', { bubbles: true }));
		await flush(10);

		expect(document.body.querySelector<HTMLInputElement>('input[aria-label="Custom height"]')!.value).toBe('960');
	});
});
