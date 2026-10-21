<script lang="ts">
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';

	type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

	export let isOpen: boolean = true;
	export let title: string;
	export let subtitle: string = '';
	export let size: ModalSize = 'md';
	export let confirmLabel: string = 'Confirm';
	export let cancelLabel: string | undefined = undefined;
	export let confirmVariant: 'primary' | 'danger' = 'primary';
	export let busy: boolean = false;
	export let confirmDisabled: boolean = false;
	export let summary: string | undefined = undefined;
	export let hideCancel: boolean = false;
	export let onConfirm: () => void;
	export let onCancel: () => void;
	export let mountBody: (el: HTMLElement) => void | (() => void);

	function bindBody(node: HTMLElement) {
		const cleanup = mountBody?.(node);
		return {
			destroy() {
				if (typeof cleanup === 'function') cleanup();
			}
		};
	}
</script>

<ConfirmDialog
	{isOpen}
	{title}
	{subtitle}
	{size}
	{busy}
	resetKey={busy}
	{cancelLabel}
	{confirmLabel}
	{confirmVariant}
	{confirmDisabled}
	{summary}
	{hideCancel}
	onConfirm={() => onConfirm()}
	onCancel={() => onCancel()}
>
	<div class="space-y-4 p-4 sm:p-6 min-w-0" use:bindBody></div>
</ConfirmDialog>
