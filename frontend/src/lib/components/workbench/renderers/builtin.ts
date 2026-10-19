/**
 * Registers the core `workbench.file` renderers - the image/video/audio
 * preview branches `Workbench.svelte` used to render inline. Imported once
 * (for its side effect) by `Workbench.svelte`; plugins register additional
 * `file_type -> component` entries the same way via
 * `registerWorkbenchFileRenderer` (exposed on `window.__potionui`).
 *
 * `mesh` and `splat` have no dedicated branch in `Workbench.svelte` (unlike
 * image/video/audio) - they are resolved through the generic `workbench.file`
 * fallback, which hands the resolved component a raw `file` object instead of
 * a plain URL prop. See `MeshPreview.svelte` and `SplatPreview.svelte`.
 */
import { registerWorkbenchFileRenderer } from '$lib/registries/workbenchFileRendererRegistry';
import ImagePreview from './ImagePreview.svelte';
import VideoPreview from './VideoPreview.svelte';
import AudioPreview from './AudioPreview.svelte';
import MeshPreview from './MeshPreview.svelte';
import SplatPreview from './SplatPreview.svelte';

registerWorkbenchFileRenderer('image', { component: ImagePreview });
registerWorkbenchFileRenderer('video', { component: VideoPreview });
registerWorkbenchFileRenderer('audio', { component: AudioPreview });
registerWorkbenchFileRenderer('mesh', { component: MeshPreview });
registerWorkbenchFileRenderer('splat', { component: SplatPreview });
