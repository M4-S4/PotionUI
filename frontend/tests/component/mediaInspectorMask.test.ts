// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mount, tick } from 'svelte';
import MediaInspector from '$lib/components/form-fields/MediaInspector.svelte';

function render(extra: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	mount(MediaInspector, {
		target,
		props: {
			kind: 'image',
			url: '/api/media/uploads/a.png',
			name: 'a.png',
			multiple: false,
			toolGroups: [],
			onTool: vi.fn(),
			onRemove: vi.fn(),
			onPreview: vi.fn(),
			...extra
		}
	});
	return target;
}

beforeEach(() => {
	document.body.innerHTML = '';
});

describe('inspector mask overlay', () => {
	it('draws the painted mask over the preview', async () => {
		const target = render({ maskUrl: '/api/media/serve?path=masks%2Fm.png', hasMask: true });
		await tick();
		const overlay = target.querySelector<HTMLElement>('[data-media-mask-overlay]');
		expect(overlay).not.toBeNull();
		expect(overlay!.getAttribute('style')).toContain('masks%2Fm.png');
	});

	it('draws nothing when there is no mask', async () => {
		const target = render({ maskUrl: null });
		await tick();
		expect(target.querySelector('[data-media-mask-overlay]')).toBeNull();
	});
});
