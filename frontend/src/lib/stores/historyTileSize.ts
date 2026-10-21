import { persistedStore } from './persistedStore';

export type HistoryTileSize = 'small' | 'medium' | 'large';

export const TILE_SIZE_MULTIPLIER: Record<HistoryTileSize, number> = {
	small: 0.72,
	medium: 1,
	large: 1.35
};

export const historyTileSize = persistedStore<HistoryTileSize>(
	'history-tile-size',
	(raw) => (raw === 'small' || raw === 'large' ? raw : 'medium'),
	'medium'
);
