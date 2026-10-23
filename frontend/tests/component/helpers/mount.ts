import { mount, unmount, flushSync, type Component } from 'svelte';
import { reactiveProps } from '../stubs/reactiveProps.svelte';

type MountOptions = { target?: HTMLElement };

const mounted = new Set<() => void>();

export function mountComponent<P extends Record<string, unknown>>(
	component: Component<any>,
	props: P = {} as P,
	options: MountOptions = {}
) {
	const target = options.target ?? document.body.appendChild(document.createElement('div'));
	const ownsTarget = !options.target;
	const live = reactiveProps<P>(props);
	let instance: ReturnType<typeof mount> | null = mount(component, { target, props: live.props });
	flushSync();

	function destroy() {
		if (!instance) return;
		unmount(instance);
		instance = null;
		if (ownsTarget) target.remove();
		mounted.delete(destroy);
	}

	mounted.add(destroy);

	return {
		target,
		rerender(next: Partial<P>) {
			live.set(next);
			flushSync();
		},
		destroy
	};
}

export function destroyMounted(): void {
	for (const destroy of [...mounted]) destroy();
}
