import type { EmbeddingProvider } from '../types';
import { TokenBucketRateLimiter } from './rate-limiter';
import type { CheckpointAdapter } from './checkpoint';

// ── Batch Iterator ──────────────────────────────────────────────────────────

/**
 * Async generator that yields fixed-size slices of an array.
 */
export async function* batchIterator<T>(items: T[], batchSize: number): AsyncGenerator<T[]> {
  for (let i = 0; i < items.length; i += batchSize) {
    yield items.slice(i, i + batchSize);
  }
}

// ── Types ───────────────────────────────────────────────────────────────────

/** Progress information passed to the onProgress callback. */
export interface PipelineProgressInfo {
  /** Number of batches completed so far. */
  completed: number;
  /** Total number of batches. */
  total: number;
  /** Elapsed time in milliseconds since embed() was called. */
  elapsed: number;
}

/** Options for configuring an embedding pipeline. */
export interface PipelineOptions {
  /** Number of texts per provider call. Default: 100. */
  batchSize?: number;
  /** Maximum concurrent provider calls. Default: 1. */
  concurrency?: number;
  /** Rate limit configuration. */
  rateLimit?: { requestsPerMinute?: number; tokensPerMinute?: number };
  /** Called after each batch completes. */
  onProgress?: (info: PipelineProgressInfo) => void;
  /** Checkpoint adapter for save/resume. */
  checkpoint?: CheckpointAdapter;
  /** Save checkpoint every N batches. Default: 10. */
  checkpointInterval?: number;
}

/** An embedding pipeline that processes texts in batches with concurrency control. */
export interface EmbeddingPipeline {
  /** Embed all texts, returning one Float32Array per input. */
  embed(texts: string[], ids?: string[]): Promise<Float32Array[]>;
}

// ── Semaphore ───────────────────────────────────────────────────────────────

class Semaphore {
  private current = 0;
  private readonly queue: Array<() => void> = [];

  constructor(private readonly max: number) {}

  async acquire(): Promise<void> {
    if (this.current < this.max) {
      this.current++;
      return;
    }
    return new Promise<void>((resolve) => {
      this.queue.push(() => {
        this.current++;
        resolve();
      });
    });
  }

  release(): void {
    this.current--;
    const next = this.queue.shift();
    if (next) next();
  }
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/**
 * Creates an embedding pipeline with batching, concurrency control,
 * rate limiting, progress callbacks, and optional checkpoint/resume.
 */
export function createEmbeddingPipeline(
  provider: EmbeddingProvider,
  options: PipelineOptions = {},
): EmbeddingPipeline {
  const {
    batchSize = 100,
    concurrency = 1,
    rateLimit,
    onProgress,
    checkpoint,
    checkpointInterval = 10,
  } = options;

  const rateLimiter = rateLimit ? new TokenBucketRateLimiter(rateLimit) : null;

  return {
    async embed(texts: string[], ids?: string[]): Promise<Float32Array[]> {
      // Load checkpoint state if adapter is provided
      let completedIds: Set<string> | null = null;
      let totalProcessed = 0;
      if (checkpoint) {
        const state = await checkpoint.load();
        if (state) {
          completedIds = new Set(state.completedIds);
          totalProcessed = state.totalProcessed;
        } else {
          completedIds = new Set();
        }
      }

      const results = new Array<Float32Array>(texts.length);
      const batches: Array<{ startIdx: number; batchTexts: string[]; batchIds?: string[] }> = [];

      // Build batch list, skipping checkpointed items
      let batchStart = 0;
      for (const batch of syncBatchSlices(texts, batchSize)) {
        const startIdx = batchStart;
        const endIdx = batchStart + batch.length;
        const batchIds = ids?.slice(startIdx, endIdx);

        if (completedIds && batchIds) {
          // Check if all items in this batch are already completed
          const allDone = batchIds.every((id) => completedIds!.has(id));
          if (allDone) {
            batchStart = endIdx;
            continue;
          }
        }

        batches.push({ startIdx, batchTexts: batch, batchIds });
        batchStart = endIdx;
      }

      const totalBatches = batches.length;
      let completedBatches = 0;
      const start = performance.now();
      const sem = new Semaphore(concurrency);
      let firstError: Error | null = null;

      const batchPromises = batches.map(async ({ startIdx, batchTexts, batchIds }) => {
        if (firstError) return;

        await sem.acquire();
        try {
          if (firstError) return;

          // Rate limit
          if (rateLimiter) {
            const tokenEstimate = batchTexts.reduce((sum, t) => sum + t.length, 0);
            await rateLimiter.waitForCapacity(tokenEstimate);
          }

          const result = await provider.embed(batchTexts);

          // Store results in correct positions
          for (let i = 0; i < result.embeddings.length; i++) {
            results[startIdx + i] = result.embeddings[i];
          }

          // Update checkpoint tracking
          if (completedIds && batchIds) {
            for (const id of batchIds) {
              completedIds.add(id);
            }
            totalProcessed += batchTexts.length;
          }

          completedBatches++;

          // Save checkpoint at interval
          if (checkpoint && completedIds && completedBatches % checkpointInterval === 0) {
            await saveCheckpoint(checkpoint, completedIds, totalProcessed);
          }

          // Progress callback
          if (onProgress) {
            onProgress({
              completed: completedBatches,
              total: totalBatches,
              elapsed: performance.now() - start,
            });
          }
        } catch (err) {
          if (!firstError) firstError = err as Error;
        } finally {
          sem.release();
        }
      });

      await Promise.all(batchPromises);

      if (firstError) throw firstError;

      // Final checkpoint save
      if (checkpoint && completedIds) {
        await saveCheckpoint(checkpoint, completedIds, totalProcessed);
      }

      return results;
    },
  };
}

async function saveCheckpoint(
  checkpoint: CheckpointAdapter,
  completedIds: Set<string>,
  totalProcessed: number,
): Promise<void> {
  await checkpoint.save({
    completedIds: [...completedIds],
    totalProcessed,
    timestamp: Date.now(),
  });
}

/** Synchronous batch slicing helper (no async overhead). */
function* syncBatchSlices<T>(items: T[], batchSize: number): Generator<T[]> {
  for (let i = 0; i < items.length; i += batchSize) {
    yield items.slice(i, i + batchSize);
  }
}
