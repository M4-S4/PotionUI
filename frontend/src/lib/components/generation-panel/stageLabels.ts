import { parseTemplateMarkers, type StatusMarker } from '$lib/utils/templateProcessor';

export const FALLBACK_STAGE_LABEL = 'Working…';

const STAGE_BY_FAMILY: Record<string, string> = {
	checkpoint_loader: 'Loading model',
	model_loader: 'Loading model',
	controlnet_loader: 'Loading model',
	media_loader: 'Loading media',
	from_iotype: 'Preparing',
	param_emitter: 'Preparing',
	seed_generator: 'Preparing',
	prompt_encoder: 'Reading your prompt',
	prompt_expander: 'Reading your prompt',
	dynamic_prompts_renderer: 'Reading your prompt',
	adm_guidance: 'Generating',
	sag: 'Generating',
	generator: 'Generating',
	cloud_generate: 'Generating',
	sharpness: 'Refining',
	detailer: 'Refining',
	tiled_detailer: 'Refining',
	tiled_refiner: 'Refining',
	upscaler: 'Upscaling',
	latent_upscaler: 'Upscaling',
	interpolator: 'Smoothing motion',
	matting: 'Removing background',
	mask_preprocessor: 'Preparing',
	controlnet_preprocessor: 'Preparing',
	inpaint_region_crop: 'Preparing',
	inpaint_region_restore: 'Finishing up',
	film_grain: 'Finishing up',
	media_tool: 'Finishing up',
	output_skipper: 'Finishing up',
	artifact: 'Finishing up',
	gallery: 'Saving'
};

const KNOWN_LABELS = new Set(Object.values(STAGE_BY_FAMILY).map((label) => label.toLowerCase()));

const USER_META_TYPES = new Set(['RESOLUTION', 'TIME']);

export interface StageInput {
	currentStep?: string | null;
	message?: string | null;
	pipeName?: string | null;
	pipeId?: number | null;
	isAdmin: boolean;
}

export interface StageStatus {
	label: string;
	stepCount: string | undefined;
	meta: StatusMarker[];
	technical: string | null;
}

export function stageLabel(pipeName?: string | null, pipeTitle?: string | null): string {
	const family = pipeName?.split('/', 1)[0]?.trim().toLowerCase();
	if (family && STAGE_BY_FAMILY[family]) return STAGE_BY_FAMILY[family];
	const title = pipeTitle?.trim();
	if (title && KNOWN_LABELS.has(title.toLowerCase())) return title;
	return FALLBACK_STAGE_LABEL;
}

export function resolveStageStatus(input: StageInput): StageStatus {
	const step = parseTemplateMarkers(input.currentStep || '');
	const message = parseTemplateMarkers(input.message || '');
	const markers = [...step.markers, ...message.markers];
	const pipeTitle = markers.find((marker) => marker.type === 'PIPE')?.value;
	const stepCount = markers.find((marker) => marker.type === 'PROGRESS')?.value;
	const label = stageLabel(input.pipeName, pipeTitle);

	const seen = new Set<string>();
	const meta = markers.filter((marker) => {
		if (marker.type === 'PIPE' || marker.type === 'PROGRESS') return false;
		if (!input.isAdmin && !USER_META_TYPES.has(marker.type)) return false;
		const key = `${marker.type}:${marker.value}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});

	return {
		label,
		stepCount,
		meta,
		technical: input.isAdmin ? technicalDetail(input, pipeTitle) : null
	};
}

function technicalDetail(input: StageInput, pipeTitle: string | undefined): string | null {
	const parts = [
		pipeTitle ? `Pipe: ${pipeTitle}` : null,
		input.pipeName ? `Name: ${input.pipeName}` : null,
		input.pipeId != null ? `Index: ${input.pipeId}` : null,
		input.currentStep ? `Step: ${input.currentStep}` : null,
		input.message && input.message !== input.currentStep ? `Message: ${input.message}` : null
	].filter(Boolean);
	return parts.length ? parts.join(' | ') : null;
}
