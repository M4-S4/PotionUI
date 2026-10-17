import { lockReason } from '$lib/generation/compare/candidates';

type Node = Record<string, any>;

function collect(nodes: Node[] | undefined, into: Node[]): void {
	for (const node of nodes ?? []) {
		if (!node) continue;
		into.push(node);
		collect(node.children, into);
	}
}

export function annotateLockHints(
	schema: { properties?: Record<string, { children?: Node[] }> } | null | undefined,
	formData: Record<string, unknown>
): void {
	if (!schema?.properties) return;
	const all: Node[] = [];
	for (const root of Object.values(schema.properties)) collect(root?.children, all);
	const byName = new Map<string, Node>();
	for (const node of all) if (node.name) byName.set(node.name, node);
	for (const node of all) {
		if (!node.name || node.disabled !== true || node.visible === false) continue;
		const reason = lockReason(node, formData, byName);
		if (!reason || !reason.startsWith('Set by')) continue;
		const base = typeof node.description === 'string' ? node.description.trim() : '';
		node.description = base ? `${base} ${reason}.` : `${reason}.`;
	}
}
