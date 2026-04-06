/**
 * State persisted by the checkpoint system.
 */
export interface CheckpointState {
  /** IDs of items that have been successfully processed. */
  completedIds: string[];
  /** Total number of items processed so far. */
  totalProcessed: number;
  /** Unix timestamp (ms) when the checkpoint was saved. */
  timestamp: number;
}

/**
 * Adapter interface for checkpoint persistence.
 *
 * Users provide their own storage backend (file system, localStorage,
 * Redis, etc.) by implementing this interface.
 */
export interface CheckpointAdapter {
  /** Save checkpoint state to the backing store. */
  save(state: CheckpointState): Promise<void>;
  /** Load the most recent checkpoint. Returns null if none exists or state is corrupted. */
  load(): Promise<CheckpointState | null>;
}
