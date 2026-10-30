<script lang="ts">
	let { request, onClose, onResult }: { request: any; existingMaskUrl?: string | null; onClose: () => void; onResult: (r: any, req: any) => void } = $props();
	$effect(() => {
		if (request) {
			onResult({ type: 'mask', maskPath: '/abs/storage/uploads/mask-1.png' }, request);
			onClose();
		}
	});
</script>
