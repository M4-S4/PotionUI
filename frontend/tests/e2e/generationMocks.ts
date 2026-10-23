import type { Page } from '@playwright/test';

export async function makeModelFieldsOptional(page: Page): Promise<void> {
	const relax = (node: unknown): void => {
		if (Array.isArray(node)) return node.forEach(relax);
		if (node && typeof node === 'object') {
			const record = node as Record<string, unknown>;
			if (record.type === 'model' && 'required' in record) record.required = false;
			Object.values(record).forEach(relax);
		}
	};
	await page.route(/\/api\/presets\/[^/]+\/form(\?.*)?$/, async (route) => {
		const response = await route.fetch();
		const body = await response.json();
		relax(body);
		await route.fulfill({ response, json: body });
	});
}

export interface GenerationStatusInput {
	generationId: string;
	pipeId: number;
	pipeName: string;
	currentStep: string;
	progress: number;
	stepNum?: number;
	totalSteps?: number;
}

export function generationStatusMessage(input: GenerationStatusInput) {
	return {
		type: 'generation_status',
		generation_id: input.generationId,
		pipe_id: input.pipeId,
		pipe_name: input.pipeName,
		output_type: 'progress',
		status: 'running',
		current_step: input.currentStep,
		message: `<<PIPE:${input.pipeName}>>`,
		progress: input.progress,
		...(input.stepNum !== undefined ? { current_step_num: input.stepNum, total_steps: input.totalSteps } : {})
	};
}
