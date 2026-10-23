<script lang="ts">
	import { IconButton } from '$lib/components/ui';
	// Vision-image attach surface opened from ChatInput's "Add" button: current-
	// form thumbnails, the shared MediaLoaderField (upload/library/paste/history
	// — whatever it already offers, nothing invented here), reuse-last-image,
	// and the auto-attach-next-message checkbox. Markup ported verbatim from
	// the mock's .media-popover (chat-rework BRIEF2) — absolutely positioned
	// inside .composer-wrap (its CSS), a plain sibling of .composer, no portal.
	import dismiss from '$lib/actions/dismiss';
	import MediaLoaderField from '$lib/components/form-fields/MediaLoaderField.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import type { FormImageEntry } from '$lib/chat/formMedia';
	import Icon from '$lib/components/Icon.svelte';

	export let triggerEl: HTMLElement | undefined = undefined;
	export let onClose: () => void;

	export let formImageEntries: FormImageEntry[] = [];
	export let selectedDurablePath: string | null = null;
	export let onSelectFormImage: (entry: FormImageEntry) => void = () => {};
	export let selectedImageData: unknown = null;
	export let onMediaLoaderChange: (value: unknown) => void = () => {};
	export let lastGeneratedImage: { url: string; name: string; generatedAt?: string } | null = null;
	export let onAttachLastImage: () => void = () => {};
	export let alwaysAttachLastImage = false;
	export let onToggleAttachImage: (() => void) | undefined = undefined;

	function isSelected(entry: FormImageEntry): boolean {
		return (entry.media.relative_path || entry.media.path) === selectedDurablePath;
	}

	function pickFormImage(entry: FormImageEntry) {
		onSelectFormImage(entry);
		onClose();
	}

	function attachAgain() {
		onAttachLastImage();
		onClose();
	}

	function handleFieldChange(_fieldName: string, value: unknown) {
		onMediaLoaderChange(value);
	}
</script>

<div
	class="media-popover"
	use:dismiss={{ onDismiss: onClose, ignore: [triggerEl], skipModals: true }}
	role="dialog"
	aria-label="Attach an image"
	data-testid="chat-attach-popover"
>
	<div class="media-head">
		<span class="media-head-icon"><Icon name="chat-image" className="icon" strokeWidth={1.8} /></span>
		<div class="media-head-copy">
			<strong>Attach an image</strong>
			<span>One image can be sent with the next message</span>
		</div>
		<span class="capability-badge">VISION</span>
		<Tooltip text="Close" position="bottom">
			<IconButton icon="chat-close" label="Close image picker" size="sm" onclick={onClose} />
		</Tooltip>
	</div>

	<div class="media-body">
		{#if formImageEntries.length > 0}
			<div class="media-section-label">From current form</div>
			<div class="form-images">
				{#each formImageEntries as entry (entry.key)}
					<Tooltip text={entry.label} wrapperClass="inline-flex items-center flex-shrink-0">
						<button
							type="button"
							class="form-image"
							class:selected={isSelected(entry)}
							aria-label={entry.label}
							on:click={() => pickFormImage(entry)}
						>
							<img src={entry.url} alt={entry.label} />
						</button>
					</Tooltip>
				{/each}
			</div>
		{/if}

		<MediaLoaderField
			name="vision_image"
			value={selectedImageData}
			onChange={handleFieldChange}
			config={{ title: 'Reference image', accept: 'image/*' }}
		/>

		{#if lastGeneratedImage}
			<div class="last-image-row">
				<img src={lastGeneratedImage.url} alt="Last generated" />
				<div class="last-image-copy">
					<strong>Use last generated image</strong>
					<span
						>{lastGeneratedImage.name}{#if lastGeneratedImage.generatedAt}
							· {lastGeneratedImage.generatedAt}{/if}</span
					>
				</div>
				<button type="button" class="reuse-button" on:click={attachAgain}>Attach again</button>
			</div>
		{/if}
	</div>

	{#if onToggleAttachImage}
		<label class="auto-attach-row">
			<input type="checkbox" checked={alwaysAttachLastImage} on:change={() => onToggleAttachImage?.()} />
			<span>Auto-attach the last generated image to your next message</span>
			<small>ONCE WHEN NEW</small>
		</label>
	{/if}
</div>
