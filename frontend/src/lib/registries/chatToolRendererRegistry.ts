import { createRegistry, CORE_OWNER, pluginOwner } from './registry';
import { resolveLazyEntry } from './lazyResolve';

export type ChatToolRendererEntry =
	| { kind: 'static'; component: any }
	| { kind: 'lazy'; pluginId: string; asset: string };

const registry = createRegistry<ChatToolRendererEntry>('chat-tool-renderer');

const resolvedCache = new Map<string, any | null>();

function registerChatToolRenderer(
	toolName: string,
	entry: { component: any } | { pluginId: string; asset: string }
): void {
	if ('component' in entry) {
		registry.register(toolName, { kind: 'static', component: entry.component }, CORE_OWNER);
	} else {
		registry.register(
			toolName,
			{ kind: 'lazy', pluginId: entry.pluginId, asset: entry.asset },
			pluginOwner(entry.pluginId)
		);
		resolvedCache.delete(toolName);
	}
}

function unregisterChatToolRenderer(toolName: string, owner?: string): void {
	registry.unregister(toolName, owner);
	resolvedCache.delete(toolName);
}

/** Resolve `toolName` to its renderer component. Returns `null` if unregistered. Publication rules in `lazyResolve.ts`. */
function resolveChatToolRenderer(toolName: string): Promise<any | null> {
	return resolveLazyEntry(registry, resolvedCache, toolName);
}

export const chatToolRendererRegistry = {
	register: registerChatToolRenderer,
	unregister: unregisterChatToolRenderer,
	resolve: resolveChatToolRenderer,
	has: (toolName: string) => registry.has(toolName),
	keys: () => registry.keys()
};
