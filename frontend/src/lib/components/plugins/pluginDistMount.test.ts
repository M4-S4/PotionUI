// @vitest-environment jsdom
// Mounts the REAL compiled plugin dists the way PluginSlot does, so a
// toolchain/runtime interop break (Svelte 4 class API vs Svelte 5 function
// components, dual-runtime mounts) fails here instead of in the live app.
import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { stagePluginDist } from '../../../../tests/component/helpers/stagePluginDist';

const REPO_ROOT = resolve(__dirname, '../../../../..');

const IMAGE_ACTION_PROPS = {
	context: {
		generationId: 'gen-1',
		fileIndex: 0,
		filename: 'x.png',
		fileUrl: '/api/media/generations/gen-1/x.png',
		fileType: 'image'
	},
	hookName: 'image.actions',
	pluginId: 'test'
};

const DISTS = [
	{
		path: 'content/plugins/marketplace/a1111-metadata-export/frontend/dist/ExportA1111Button.js',
		props: IMAGE_ACTION_PROPS
	},
	{
		path: 'content/plugins/marketplace/image-modal/frontend/dist/ImageModalAction.js',
		props: IMAGE_ACTION_PROPS
	},
	// prompt-text-import is a `content/plugins/local/` dev repo (gitignored here,
	// not always checked out) - skip it where it isn't present.
	{
		path: 'content/plugins/local/prompt-text-import/frontend/dist/TextImportModal.js',
		props: { onClose: () => {}, onImported: () => {} }
	}
].filter((dist) => existsSync(resolve(REPO_ROOT, dist.path)));

describe.each(DISTS)('plugin dist $path', ({ path, props }) => {
	it('mounts and destroys via the class API with props', async () => {
		const mod = await import(/* @vite-ignore */ stagePluginDist(resolve(REPO_ROOT, path)));
		const Component = mod.default;
		expect(Component).toBeTypeOf('function');

		const target = document.createElement('div');
		document.body.appendChild(target);
		const instance = new Component({ target, props });
		expect(instance).toBeTruthy();
		if (typeof instance.$set === 'function') {
			instance.$set(props);
		}
		instance.$destroy();
		target.remove();
	});
});
