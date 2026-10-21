import { logger, getErrorMessage } from '$lib/utils/logger';
import type { SmartView } from './types';

export interface MembershipResponse {
	success: boolean;
	data?: { added?: number; removed?: number } | null;
}

export interface CreateCollectionResponse {
	success: boolean;
	data?: { collection?: { id: string } | null } | null;
}

export interface MembershipActionsConfig {
	getSelectedIds: () => string[];
	add: (collectionId: string, ids: string[]) => Promise<MembershipResponse>;
	remove?: (collectionId: string, ids: string[]) => Promise<MembershipResponse>;
	create: (name: string) => Promise<CreateCollectionResponse>;
	reloadCollections?: () => Promise<unknown>;
	clearSelection: () => void;
	afterRemove?: () => Promise<unknown>;
	flash: (text: string) => void;
}

export interface MembershipActions {
	addToCollection: (collectionId: string) => Promise<boolean>;
	removeFromCollection: (collectionId: string | undefined) => Promise<void>;
	createAndAdd: (name: string) => Promise<boolean>;
}

export function createMembershipActions(config: MembershipActionsConfig): MembershipActions {
	const { getSelectedIds, add, remove, create, reloadCollections, clearSelection, afterRemove, flash } =
		config;

	async function addToCollection(collectionId: string): Promise<boolean> {
		const ids = getSelectedIds();
		if (ids.length === 0) return false;
		try {
			const response = await add(collectionId, ids);
			if (response.success) {
				await reloadCollections?.();
				const added = response.data?.added ?? ids.length;
				clearSelection();
				flash(`Added ${added} to collection`);
				return true;
			}
			flash('Failed to add to collection');
			return false;
		} catch (e) {
			logger.error('Add to collection failed:', getErrorMessage(e));
			flash('Failed to add to collection');
			return false;
		}
	}

	async function removeFromCollection(collectionId: string | undefined): Promise<void> {
		const ids = getSelectedIds();
		if (!remove || !collectionId || ids.length === 0) return;
		try {
			const response = await remove(collectionId, ids);
			if (response.success) {
				await reloadCollections?.();
				const removed = response.data?.removed ?? ids.length;
				clearSelection();
				flash(`Removed ${removed} from collection`);
				await afterRemove?.();
				return;
			}
			flash('Failed to remove from collection');
		} catch (e) {
			logger.error('Remove from collection failed:', getErrorMessage(e));
			flash('Failed to remove from collection');
		}
	}

	async function createAndAdd(name: string): Promise<boolean> {
		try {
			const created = await create(name);
			const collection = created.success ? created.data?.collection : undefined;
			if (!collection) {
				flash('Failed to create collection');
				return false;
			}
			return await addToCollection(collection.id);
		} catch (e) {
			logger.error('Create collection failed:', getErrorMessage(e));
			flash('Failed to create collection');
			return false;
		}
	}

	return { addToCollection, removeFromCollection, createAndAdd };
}

export interface SmartViewsConfig {
	icon: string;
	label: string;
	isAll: boolean;
	favoritesOnly: boolean;
	unsorted: boolean;
	counts?: { all?: number; favorites?: number; unsorted?: number } | null;
	onAll: () => void | Promise<void>;
	onFavorites: () => void | Promise<void>;
	onUnsorted: () => void | Promise<void>;
}

export function buildSmartViews(config: SmartViewsConfig): {
	allView: SmartView;
	favoritesView: SmartView;
	unsortedView: SmartView;
} {
	const { counts } = config;
	return {
		allView: {
			id: 'all',
			icon: config.icon,
			label: config.label,
			active: config.isAll,
			count: counts?.all,
			onSelect: config.onAll
		},
		favoritesView: {
			id: 'favorites',
			icon: 'heart',
			label: 'Favorites',
			active: config.favoritesOnly,
			count: counts?.favorites,
			onSelect: config.onFavorites
		},
		unsortedView: {
			id: 'unsorted',
			icon: 'inbox',
			label: 'Unsorted',
			active: config.unsorted,
			count: counts?.unsorted,
			onSelect: config.onUnsorted
		}
	};
}

export interface FilterSelection {
	collectionId?: string;
	favoritesOnly?: boolean;
	unsorted?: boolean;
}

export function createFilterSelection(
	setFilter: (key: any, value: any) => void,
	reload: () => Promise<unknown>
) {
	async function applySelection(patch: FilterSelection): Promise<void> {
		setFilter('collectionId', patch.collectionId);
		setFilter('favoritesOnly', !!patch.favoritesOnly);
		setFilter('unsorted', !!patch.unsorted);
		await reload();
	}

	async function setDirectOnly(value: boolean): Promise<void> {
		setFilter('directOnly', value);
		await reload();
	}

	return { applySelection, setDirectOnly };
}
