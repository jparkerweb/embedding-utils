import type { EmbeddingProvider, EmbedOptions } from '../types';
import { ValidationError } from '../types';
import { averageEmbeddings } from './average';

/**
 * Generates embeddings for multiple texts and combines them into a single embedding.
 * @param texts - Array of text strings to embed
 * @param provider - The embedding provider to use
 * @param options - Optional aggregation function and embed options
 * @returns A single combined embedding vector
 * @throws {ValidationError} If the texts array is empty
 * @example
 * const embedding = await combineEmbeddings(['hello', 'world'], provider);
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
