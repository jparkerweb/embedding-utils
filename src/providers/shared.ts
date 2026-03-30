import type { RetryConfig } from '../types';
import { ProviderError } from '../types';

function isRetryable(error: unknown): boolean {
  if (!(error instanceof ProviderError)) return false;
  const status = error.status;
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
        throw new Error('Aborted');
      }

      const exponential = baseDelay * Math.pow(2, attempt);
      const jitter = exponential * Math.random() * 0.3;
      const delay = Math.min(exponential + jitter, maxDelay);

      await new Promise<void>((resolve) => setTimeout(resolve, delay));

      if (signal?.aborted) {
        throw new Error('Aborted');
      }
    }
  }

  throw lastError;
}

/**
 * Splits inputs into batches and processes them sequentially.
 * @param inputs - Array of input strings
 * @param batchSize - Maximum items per batch
 * @param fn - Async function that processes a batch and returns embeddings
 * @returns Flattened array of all embedding results
 * @example
 * const embeddings = await autoBatch(texts, 100, (batch) => embed(batch));
 */
export async function autoBatch(
  inputs: string[],
  batchSize: number,
  fn: (batch: string[]) => Promise<number[][]>,
): Promise<number[][]> {
  if (inputs.length === 0) return [];

  const results: number[][][] = [];

  for (let i = 0; i < inputs.length; i += batchSize) {
    const batch = inputs.slice(i, i + batchSize);
    const batchResult = await fn(batch);
    results.push(batchResult);
  }

  return results.flat();
}
