import { persistedStore } from './persistedStore';

export const playfulStatusLines = persistedStore<boolean>('potionui-playful-status-lines', (raw) => raw !== 'false', true);
