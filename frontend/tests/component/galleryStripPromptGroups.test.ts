// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import GalleryStrip from '$lib/components/workbench/GalleryStrip.svelte';

let cleanup: (() => void) | undefined;

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	document.body.innerHTML = '';
});

function image(promptIndex: number | undefined, promptImage: number | undefined, promptText?: string) {
	return {
		url: `/img-${promptIndex}-${promptImage}.png`,
		promptIndex,
		promptImage,
		promptText,
		label: promptIndex === undefined ? null : `${promptIndex + 1}.${promptImage}`
	};
}

function render(batchImages: ReturnType<typeof image>[], selectedIndex = 0) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(GalleryStrip, { target, props: { batchImages, selectedIndex } });
	cleanup = () => {
		unmount(instance);
		target.remove();
	};
	return target;
}

const BATCH = [
	image(0, 1, 'a lighthouse'),
	image(0, 2, 'a lighthouse'),
	image(1, 1, 'a red fox'),
	image(1, 2, 'a red fox')
];

describe('GalleryStrip prompt grouping', () => {
	it('groups a batch into one row per prompt with its text', () => {
		const target = render(BATCH);

		expect(target.textContent).toContain('4 images · 2 prompts');
		expect(target.textContent).toContain('a lighthouse');
		expect(target.textContent).toContain('a red fox');
		expect(target.querySelectorAll('img')).toHaveLength(4);
		expect(target.querySelectorAll('[data-output-label]')).toHaveLength(4);
		expect(target.textContent).toContain('2.1');
	});

	it('flattens back to the single strip with the All toggle', async () => {
		const target = render(BATCH);

		[...target.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'All')!.click();
		flushSync();
		await tick();

		expect(target.textContent).not.toContain('a lighthouse');
		expect(target.textContent).toContain('#4');
	});

	it('shows no grouping chrome for an ordinary generation', () => {
		const target = render([image(undefined, undefined), image(undefined, undefined)]);

		expect(target.textContent).not.toContain('By prompt');
		expect(target.textContent).toContain('#2');
	});
});
