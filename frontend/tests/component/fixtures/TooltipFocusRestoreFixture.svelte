<script lang="ts">
	import tooltip from '$lib/actions/tooltip';
	export let open = true;
	let trigger: HTMLButtonElement;

	function restoreFocus() {
		return {
			destroy() {
				trigger.focus();
			}
		};
	}
</script>

<button bind:this={trigger} type="button" use:tooltip={'Open dialog'}>trigger</button>
{#if open}
	<div use:restoreFocus></div>
{/if}
