// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import { createClassComponent } from 'svelte/legacy';
const modules = import.meta.glob('../../../content/plugins/marketplace/model-browser/frontend/src/components/NsfwMedia.svelte', { eager: true }) as Record<string, { default: any }>;
const NsfwMedia = Object.values(modules)[0].default;

afterEach(() => {
	document.body.innerHTML = '';
});

describe('model browser remote previews', () => {
	it('sends no referrer so hotlink-protected hosts serve the image', async () => {
		const target = document.createElement('div');
		document.body.appendChild(target);
		createClassComponent({
			component: NsfwMedia as never,
			target,
			props: { image: { id: 'm1', url: 'https://i.slow.pics/abc.png' }, alt: 'preview' }
		});
		await tick();
		const img = target.querySelector('img');
		expect(img).not.toBeNull();
		expect(img!.getAttribute('referrerpolicy')).toBe('no-referrer');
	});
});
