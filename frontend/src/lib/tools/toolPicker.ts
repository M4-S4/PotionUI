import { openMediaTool } from '$lib/media-tools/open';
import type { MediaToolItem as ToolResultItem } from '$lib/media-tools/types';
import type { MediaTool, MediaToolContext } from './tools';

export interface ToolPickHandlers {
	mount: (tool: MediaTool, context: MediaToolContext) => void;
	onResult: (items: ToolResultItem[], context: MediaToolContext) => void | Promise<void>;
}

export function pickTool(tool: MediaTool, context: MediaToolContext, handlers: ToolPickHandlers): void {
	if (!tool.catalogToolId) {
		handlers.mount(tool, context);
		return;
	}
	void openMediaTool({
		toolId: tool.catalogToolId,
		items: context.items,
		scope: context.scope,
		context,
		onResult: (items) => handlers.onResult(items, context)
	});
}
