import type { EmbeddingProvider, EmbedOptions } from '../types';
import { ValidationError } from '../types';
import { averageEmbeddings } from './average';

/**
 * Generates embeddings for multiple texts and combines them into a single
 * embedding vector in one call.
 *
 * This is a convenience wrapper that:
 * 1. Calls `provider.embed(texts)` to generate embeddings for all texts
 * 2. Applies an aggregation function (default: {@link averageEmbeddings}) to
 *    combine them into a single vector
 *
 * **When to use:** Creating a single "entity embedding" from multiple text
 * fields. For example, combine a user's bio, interests, and recent posts
 * into one "user profile" vector for recommendation matching.
 *
 * **Custom aggregation:** Pass any function that takes `number[][]` and
 * returns `number[]`. Use {@link maxPooling} to preserve strongest signals,
 * or {@link weightedAverage} with custom weights.
 *
 * @param texts - Array of text strings to embed (must be non-empty)
 * @param provider - The embedding provider to use for text-to-vector conversion
 * @param options - Optional aggregation function and embed options
 * @returns A single combined embedding vector
 * @throws {ValidationError} If the texts array is empty
 *
 * @example
 * // Default averaging
 * const embedding = await combineEmbeddings(['hello', 'world'], provider);
 *
 * @example
 * // Max pooling to preserve strongest signals
 * const embedding = await combineEmbeddings(texts, provider, {
 *   aggregate: maxPooling,
 * });
 */
export async function combineEmbeddings(
  texts: string[],
  provider: EmbeddingProvider,
  options?: {
    aggregate?: (embeddings: number[][]) => number[];
    embedOptions?: EmbedOptions;
  },
): Promise<number[]> {
  if (texts.length === 0) {
    throw new ValidationError('texts array must be non-empty');
  }

  const result = await provider.embed(texts, options?.embedOptions);
  const aggregate = options?.aggregate ?? averageEmbeddings;
  return aggregate(result.embeddings);
}
