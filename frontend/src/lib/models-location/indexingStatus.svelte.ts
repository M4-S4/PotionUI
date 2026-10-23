import { api } from '$lib/services/api';
import { logger } from '$lib/utils/logger';
import { createVisiblePoll } from '$lib/utils/visiblePoll';
import { modelRootsErrorMessage } from '$lib/model-roots/errors';
import type { APIResponse } from '$lib/types/api';
import type {
	IndexingDigestRequest,
	IndexingLevel,
	IndexingStatus
} from '$lib/services/api/models';
import { indexingDigests, indexingIsActive } from './indexingDisplay';

const POLL_INTERVAL_MS = 2000;
const STALE_AFTER_MS = 5000;
const SAVE_FAILED = 'The change could not be saved.';

type Step = () => Promise<APIResponse<unknown>>;

class IndexingStatusStore {
	status = $state<IndexingStatus | null>(null);
	loading = $state(false);
	error = $state<string | null>(null);
	busy = $state(false);
	actionError = $state<string | null>(null);
	pickRequested = $state(false);

	private subscribers = 0;
	private poll = createVisiblePoll(() => this.refresh(), POLL_INTERVAL_MS);
	private inFlight: Promise<void> | null = null;
	private fetchedAt = 0;

	subscribe(): () => void {
		this.subscribers += 1;
		if (this.subscribers === 1 || this.isStale()) {
			void this.refresh();
		} else if (indexingIsActive(this.status)) {
			this.ensurePolling();
		}
		return () => {
			this.subscribers = Math.max(0, this.subscribers - 1);
			if (this.subscribers === 0) this.stopPolling();
		};
	}

	async refresh(force = false): Promise<void> {
		if (this.inFlight && !force) return this.inFlight;
		while (this.inFlight) await this.inFlight;
		this.loading = this.status === null;
		this.inFlight = (async () => {
			try {
				const response = await api.getIndexingStatus();
				if (response.success && response.data) {
					this.fetchedAt = Date.now();
					this.applyStatus(response.data);
				} else {
					this.error = response.message ?? 'Failed to load indexing status.';
				}
			} catch (e) {
				logger.error('Failed to load indexing status:', e);
				this.error = 'Failed to load indexing status.';
			} finally {
				this.loading = false;
				this.inFlight = null;
			}
		})();
		return this.inFlight;
	}

	notifyRunStarted(status: IndexingStatus): void {
		this.fetchedAt = Date.now();
		this.applyStatus(status);
	}

	requestPick(): void {
		this.pickRequested = true;
	}

	closePick(): void {
		this.pickRequested = false;
	}

	setLevel(level: IndexingLevel): Promise<boolean> {
		return this.run([() => api.setIndexingSettings({ level })]);
	}

	pause(): Promise<boolean> {
		return this.run([() => api.pauseIndexing()]);
	}

	resume(): Promise<boolean> {
		return this.run([() => api.resumeIndexing()]);
	}

	indexAll(): Promise<boolean> {
		return this.run([
			() => api.setIndexingSettings({ policy: 'background' }),
			() => api.queueIndexingDigests({ mode: 'all' })
		]);
	}

	indexSelected(requests: IndexingDigestRequest[]): Promise<boolean> {
		const steps: Step[] = requests.map((request) => () => api.queueIndexingDigests(request));
		if (indexingDigests(this.status)?.policy === 'unset') {
			steps.push(() => api.setIndexingSettings({ policy: 'on_demand' }));
		}
		return this.run(steps);
	}

	later(): Promise<boolean> {
		return this.run([() => api.setIndexingSettings({ policy: 'on_demand' })]);
	}

	private async run(steps: Step[]): Promise<boolean> {
		this.busy = true;
		this.actionError = null;
		let succeeded = true;
		try {
			for (const step of steps) {
				const response = await step();
				if (!response.success) {
					this.actionError = response.message ?? response.error ?? SAVE_FAILED;
					succeeded = false;
					break;
				}
			}
		} catch (e) {
			logger.error('Indexing change failed:', e);
			this.actionError = modelRootsErrorMessage(e, SAVE_FAILED);
			succeeded = false;
		}
		await this.refresh(true);
		this.busy = false;
		return succeeded;
	}

	private isStale(): boolean {
		return Date.now() - this.fetchedAt > STALE_AFTER_MS;
	}

	private applyStatus(status: IndexingStatus): void {
		this.status = status;
		this.error = null;
		if (this.subscribers > 0 && indexingIsActive(status)) {
			this.ensurePolling();
		} else {
			this.stopPolling();
		}
	}

	private ensurePolling(): void {
		this.poll.start();
	}

	private stopPolling(): void {
		this.poll.stop();
	}
}

export const indexingStatusStore = new IndexingStatusStore();
