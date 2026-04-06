import type { EmbeddingProvider, EmbedOptions, Vector } from '../types';
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
 * @param texts - Array of text strings to embed (must be non-empty)
 * @param provider - The embedding provider to use for text-to-vector conversion
 * @param options - Optional aggregation function and embed options
 * @returns A single combined embedding vector
 * @throws {ValidationError} If the texts array is empty
 */
export async function combineEmbeddings(
  texts: string[],
  provider: EmbeddingProvider,
  options?: {
    aggregate?: (embeddings: Vector[]) => Float32Array;
    embedOptions?: EmbedOptions;
  },
): Promise<Float32Array> {
  if (texts.length === 0) {
    throw new ValidationError('texts array must be non-empty');
  }

  const result = await provider.embed(texts, options?.embedOptions);
  const aggregate = options?.aggregate ?? averageEmbeddings;
  return aggregate(result.embeddings);
}
