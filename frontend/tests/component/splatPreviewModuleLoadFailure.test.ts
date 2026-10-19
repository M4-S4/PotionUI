import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('$lib/components/workbench/renderers/splat/splatViewer', () => {
	throw new Error('module unavailable');
});

const { createClassComponent } = await import('svelte/legacy');
const { default: SplatPreview } = await import(
	'$lib/components/workbench/renderers/SplatPreview.svelte'
);

async function settle() {
	for (let i = 0; i < 8; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

let target: HTMLDivElement | undefined;
let component: { $destroy: () => void } | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	target = undefined;
});

describe('SplatPreview module load failure', () => {
	it('shows an error alert instead of a permanent spinner when the viewer module fails to import', async () => {
		target = document.createElement('div');
		document.body.appendChild(target);
		component = createClassComponent({
			component: SplatPreview as never,
			target,
			props: { file: { url: '/api/media/generations/g/0.ply' } }
		}) as never;
		await settle();

		expect(target.querySelector('[role="alert"]')?.textContent).toContain('failed to load');
		expect(target.querySelector('[role="status"].animate-spin')).toBeNull();
		expect(target.querySelector('canvas')).toBeNull();
	});
});
