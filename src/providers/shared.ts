import type { RetryConfig, BatchConfig } from '../types';
import { EmbeddingUtilsError, ProviderError } from '../types';
import { promisePool } from '../internal/concurrency';

const DEFAULT_TIMEOUT = 30000;

/**
 * Creates an AbortSignal that fires on timeout, optionally combined with a user signal.
 * @internal
 */
export function createTimeoutSignal(
  timeoutMs: number | undefined,
  userSignal?: AbortSignal,
): AbortSignal {
  const timeout = timeoutMs ?? DEFAULT_TIMEOUT;
  const timeoutSignal = AbortSignal.timeout(timeout);

  if (!userSignal) return timeoutSignal;

  // Combine user signal and timeout signal
  return AbortSignal.any([userSignal, timeoutSignal]);
}

/**
 * Wraps a provider operation and converts timeout DOMException into ProviderError.
 * @internal
 */
export function wrapTimeoutError(
  error: unknown,
  providerName: string,
  timeoutMs?: number,
): never {
  if (error instanceof DOMException && error.name === 'TimeoutError') {
    throw new ProviderError(
      `Request timed out after ${timeoutMs ?? DEFAULT_TIMEOUT}ms`,
      providerName,
    );
  }
  throw error;
}

function isRetryable(error: unknown): boolean {
  if (!(error instanceof ProviderError)) return false;
  const status = error.statusCode;
  if (status === undefined) return false;
  return status === 429 || (status >= 500 && status < 600);
}

/**
 * Retries a function with exponential backoff on retryable errors (429, 5xx).
 * @param fn - Async function to execute
 * @param config - Retry configuration (maxRetries, baseDelay, maxDelay)
 * @param signal - Optional AbortSignal to cancel retries
 * @returns The resolved value from fn
 * @throws Rethrows the last error after all retries are exhausted
 * @example
 * const result = await retryWithBackoff(() => fetchData(), { maxRetries: 3 });
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig,
  signal?: AbortSignal,
): Promise<T> {
  const maxRetries = config.maxRetries ?? 3;
  const baseDelay = config.baseDelay ?? 1000;
  const maxDelay = config.maxDelay ?? 30000;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (!isRetryable(error) || attempt === maxRetries) {
        throw error;
      }

      if (signal?.aborted) {
        throw new EmbeddingUtilsError('Aborted');
      }

      const exponential = baseDelay * Math.pow(2, attempt);
      const jitter = exponential * Math.random() * 0.3;
      const delay = Math.min(exponential + jitter, maxDelay);

      await new Promise<void>((resolve) => setTimeout(resolve, delay));

      if (signal?.aborted) {
        throw new EmbeddingUtilsError('Aborted');
      }
    }
  }

  throw lastError;
}

/**
 * Splits inputs into batches and processes them sequentially or concurrently.
 * @param inputs - Array of input strings
 * @param batchSize - Maximum items per batch
 * @param fn - Async function that processes a batch and returns embeddings
 * @param config - Optional batch configuration: maxConcurrency, delayBetweenBatches, onProgress
 * @returns Flattened array of all embedding results
 * @example
 * const embeddings = await autoBatch(texts, 100, (batch) => embed(batch));
 * // With progress:
 * const embeddings = await autoBatch(texts, 100, fn, {
 *   onProgress: (done, total) => console.log(`${done}/${total}`),
 * });
 */
export async function autoBatch(
  inputs: string[],
  batchSize: number,
  fn: (batch: string[]) => Promise<number[][]>,
  config?: BatchConfig,
): Promise<number[][]> {
  if (inputs.length === 0) return [];

  const maxConcurrency = config?.maxConcurrency ?? 1;
  const delay = config?.delayBetweenBatches ?? 0;
  const onProgress = config?.onProgress;

  // Split inputs into batches
  const batches: string[][] = [];
  for (let i = 0; i < inputs.length; i += batchSize) {
    batches.push(inputs.slice(i, i + batchSize));
  }
  const totalBatches = batches.length;

  if (maxConcurrency > 1) {
    // Concurrent execution via promise pool
    let completed = 0;
    const tasks = batches.map((batch) => async () => {
      const result = await fn(batch);
      completed++;
      onProgress?.(completed, totalBatches);
      return result;
    });
    const results = await promisePool(tasks, maxConcurrency);
    return results.flat();
  }

  // Sequential execution with optional delay
  const results: number[][][] = [];
  for (let i = 0; i < batches.length; i++) {
    if (i > 0 && delay > 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, delay));
    }
    const batchResult = await fn(batches[i]);
    results.push(batchResult);
    onProgress?.(i + 1, totalBatches);
  }

  return results.flat();
}
