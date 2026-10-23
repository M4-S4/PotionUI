<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import { api } from '$lib/services/api/index';
	import { collectionRulesHref, filledByLabel } from '$lib/organize/collectionRules';
	import type { OrganizeCollectionRuleRef, OrganizeCollectionScope } from '$lib/types/organize';
	import { MENU_ITEM_CLASS } from './sidebar';

	let {
		scope,
		collectionId
	}: {
		scope: OrganizeCollectionScope;
		collectionId: string;
	} = $props();

	let rules = $state<OrganizeCollectionRuleRef[]>([]);

	$effect(() => {
		const currentScope = scope;
		const id = collectionId;
		let cancelled = false;
		rules = [];
		api
			.getOrganizeCollectionRules(currentScope, id)
			.then((response) => {
				if (cancelled) return;
				rules = response.success && Array.isArray(response.data) ? response.data : [];
			})
			.catch(() => {
				if (!cancelled) rules = [];
			});
		return () => {
			cancelled = true;
		};
	});
</script>

{#if rules.length > 0}
	<a class="{MENU_ITEM_CLASS} text-fg" href={collectionRulesHref(scope, rules)} role="menuitem">
		<Icon name="sparkles" className="h-4 w-4 flex-shrink-0 text-fg-muted" strokeWidth={1.75} />
		{filledByLabel(rules.length)}
	</a>
{/if}
