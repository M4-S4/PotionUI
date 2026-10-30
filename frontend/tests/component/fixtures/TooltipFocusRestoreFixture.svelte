<script lang="ts">
	import tooltip from '$lib/actions/tooltip';
	import { keyboardFocus } from '../helpers/keyboardFocus';
	export let open = true;
	let trigger: HTMLButtonElement;

	function restoreFocus(_node: HTMLElement) {
		return {
			destroy() {
				keyboardFocus(trigger);
			}
		};
	}
</script>

<button bind:this={trigger} type="button" use:tooltip={'Open dialog'}>trigger</button>
{#if open}
	<div use:restoreFocus></div>
{/if}
