// ─────────────────────────────────────────────────────────────────────────────
// Averaging & Incremental Aggregation
//
// Functions for combining multiple embeddings into a single representative
// vector. These are essential for:
// - Creating document-level embeddings from chunk embeddings
// - Computing cluster centroids
// - Incrementally updating topic embeddings as new data arrives
// - Weighted multi-field representations (title + body + metadata)
// ─────────────────────────────────────────────────────────────────────────────

import { ValidationError } from '../types';
import { validateEmbeddings } from '../internal/validation';

/**
 * Computes the element-wise average (mean) of multiple embeddings.
 *
 * This is the simplest and most commonly used aggregation method. It
 * produces a single "center of mass" vector from a set of embeddings.
 *
 * **When to use:** When all inputs carry equal weight. Common scenarios:
 * - Averaging chunk embeddings to represent an entire document
 * - Computing cluster centroids
 * - Creating a "topic embedding" from multiple training phrases
 *
 * **Alias:** The {@link centroid} function is an alias for this function,
 * provided for semantic clarity in clustering contexts.
 *
 * @param embeddings - Array of equal-dimension embedding vectors (must be non-empty)
 * @returns A single averaged embedding vector with the same dimensionality
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 *
 * @example
 * averageEmbeddings([[1, 0], [0, 1]]); // [0.5, 0.5]
 *
 * @example
 * // Average chunk embeddings to get a document-level embedding
 * const docEmbedding = averageEmbeddings(chunkEmbeddings);
 */
export function averageEmbeddings(embeddings: number[][]): number[] {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const n = embeddings.length;
  const result = new Array<number>(dim).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < dim; j++) {
      result[j] += embeddings[i][j];
    }
  }
  for (let j = 0; j < dim; j++) {
    result[j] /= n;
  }
  return result;
}

/**
 * Computes a weighted average of multiple embeddings.
 *
 * Each embedding is scaled by its corresponding weight before averaging.
 * The result is normalized by the total weight sum, not the count.
 *
 * **When to use:** When some embeddings should contribute more than others:
 * - Weight a title embedding 3x and body 1x for a "document" representation
 * - Weight recent embeddings higher in a time-decaying average
 * - Give expert-labeled training phrases more influence than auto-generated ones
 *
 * @param embeddings - Array of equal-dimension embedding vectors
 * @param weights - Weight for each embedding (must match length of embeddings).
 *                  Weights do not need to sum to 1 — they are normalized internally.
 * @returns A single weighted-average embedding vector
 * @throws {ValidationError} If arrays are empty, dimensions mismatch, or lengths differ
 *
 * @example
 * weightedAverage([[1, 0], [0, 1]], [3, 1]); // [0.75, 0.25]
 *
 * @example
 * // Weight title 3x, body 2x, metadata 1x
 * const embedding = weightedAverage([titleEmb, bodyEmb, metaEmb], [3, 2, 1]);
 */
export function weightedAverage(embeddings: number[][], weights: number[]): number[] {
  validateEmbeddings(embeddings);
  if (embeddings.length !== weights.length) {
    throw new ValidationError(
      `Embeddings length (${embeddings.length}) must match weights length (${weights.length})`,
    );
  }
  const dim = embeddings[0].length;
  const result = new Array<number>(dim).fill(0);
  let totalWeight = 0;
  for (let i = 0; i < embeddings.length; i++) {
    totalWeight += weights[i];
    for (let j = 0; j < dim; j++) {
      result[j] += embeddings[i][j] * weights[i];
    }
  }
  for (let j = 0; j < dim; j++) {
    result[j] /= totalWeight;
  }
  return result;
}

/**
 * Updates a running average with a single new embedding (Welford's method).
 *
 * This avoids accumulating large sums by computing the update as a delta:
 * `result[i] = currentAvg[i] + (newEmbedding[i] - currentAvg[i]) / (count + 1)`
 *
 * **When to use:** Streaming / real-time scenarios where you process embeddings
 * one at a time and need a running mean with only 1 vector in memory:
 * - Monitoring average embedding drift in a production pipeline
 * - Computing running centroids from a data stream (tweets, logs, sensor readings)
 *
 * **For batch updates:** Use {@link batchIncrementalAverage} instead, which
 * accepts an array of new embeddings in one call and produces numerically
 * identical results to a full re-average.
 *
 * @param currentAvg - The current average embedding (centroid of previous data)
 * @param newEmbedding - The single new embedding to incorporate
 * @param count - The number of embeddings already averaged into `currentAvg`
 *               (before this new one)
 * @returns Updated average embedding (now represents count + 1 embeddings)
 * @throws {ValidationError} If vector dimensions mismatch
 *
 * @example
 * let avg = [1, 2, 3];
 * avg = incrementalAverage(avg, [4, 5, 6], 1); // now avg of 2 embeddings
 * avg = incrementalAverage(avg, [7, 8, 9], 2); // now avg of 3 embeddings
 * // Numerically equivalent to: averageEmbeddings([[1,2,3], [4,5,6], [7,8,9]])
 */
export function incrementalAverage(
  currentAvg: number[],
  newEmbedding: number[],
  count: number,
): number[] {
  if (currentAvg.length !== newEmbedding.length) {
    throw new ValidationError(
      `Dimension mismatch: ${currentAvg.length} vs ${newEmbedding.length}`,
    );
  }
  const dim = currentAvg.length;
  const result = new Array<number>(dim);
  const divisor = count + 1;
  for (let i = 0; i < dim; i++) {
    result[i] = currentAvg[i] + (newEmbedding[i] - currentAvg[i]) / divisor;
  }
  return result;
}

/**
 * Updates a running average by incorporating multiple new embeddings at once.
 *
 * This is the batch version of {@link incrementalAverage}. Instead of incorporating
 * one embedding at a time, it folds an entire array of new embeddings into the
 * existing average in a single, numerically stable operation.
 *
 * The formula used is:
 * ```
 * result[i] = (currentAvg[i] * count + sum(newEmbeddings[*][i])) / (count + newEmbeddings.length)
 * ```
 *
 * This produces results that are mathematically equivalent to computing the full
 * average of all embeddings from scratch, but without needing to store or re-process
 * the historical data. This is critical for incremental pipelines (e.g., updating
 * topic cluster centroids as new training data arrives).
 *
 * @param currentAvg - The current average embedding (centroid of previous data)
 * @param newEmbeddings - Array of new embedding vectors to incorporate
 * @param count - The number of embeddings that were already averaged into `currentAvg`
 * @returns Updated average embedding incorporating all new embeddings
 * @throws {ValidationError} If dimensions mismatch between currentAvg and any new embedding,
 *         or if newEmbeddings is empty
 *
 * @example
 * // Existing centroid from 10 embeddings, adding 3 more
 * const updated = batchIncrementalAverage(existingCentroid, threeNewEmbeddings, 10);
 * // Equivalent to: averageEmbeddings([...all13embeddings])
 *
 * @example
 * // Incremental topic pipeline: update cluster centroid as new phrases arrive
 * const newCentroid = batchIncrementalAverage(
 *   cluster.centroid,
 *   newPhraseEmbeddings,
 *   cluster.size,
 * );
 */
export function batchIncrementalAverage(
  currentAvg: number[],
  newEmbeddings: number[][],
  count: number,
): number[] {
  if (newEmbeddings.length === 0) {
    throw new ValidationError('newEmbeddings array must be non-empty');
  }
  const dim = currentAvg.length;
  for (let i = 0; i < newEmbeddings.length; i++) {
    if (newEmbeddings[i].length !== dim) {
      throw new ValidationError(
        `Dimension mismatch: expected ${dim}, got ${newEmbeddings[i].length} at index ${i}`,
      );
    }
  }

  const totalCount = count + newEmbeddings.length;
  const result = new Array<number>(dim);

  for (let j = 0; j < dim; j++) {
    let newSum = 0;
    for (let i = 0; i < newEmbeddings.length; i++) {
      newSum += newEmbeddings[i][j];
    }
    result[j] = (currentAvg[j] * count + newSum) / totalCount;
  }

  return result;
}

/**
 * Computes the centroid (mean) of a set of embeddings. Alias for {@link averageEmbeddings}.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns The centroid embedding vector
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 * @example
 * centroid([[1, 0], [0, 1]]); // [0.5, 0.5]
 */
export function centroid(embeddings: number[][]): number[] {
  return averageEmbeddings(embeddings);
}
