export interface ToolsStepDetail {
	offered?: string[];
	withheld?: Record<string, string>;
}

const CONTEXT_ONLY_REASON = 'unavailable';

export function toolsStepLabel(detail: ToolsStepDetail | undefined): string {
	const offered = (detail?.offered || []).length;
	if (offered === 0) return 'Tools off';
	const switchedOff = Object.values(detail?.withheld || {}).filter(
		(reason) => reason !== CONTEXT_ONLY_REASON
	).length;
	return switchedOff > 0 ? `Tools: ${offered} ready, ${switchedOff} off` : `Tools: ${offered} ready`;
}
