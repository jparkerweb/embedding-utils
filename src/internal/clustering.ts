import type { SimilarityMetric, Vector } from '../types';
import { computeScore } from './metrics';

/**
 * Computes the centroid (element-wise mean) of a set of member vectors.
 * @internal
 */
export function computeCentroid(members: Vector[]): Float32Array {
  const dims = members[0].length;
  const centroid = new Float32Array(dims);
  for (const member of members) {
    for (let i = 0; i < dims; i++) {
      centroid[i] += member[i];
    }
  }
  for (let i = 0; i < dims; i++) {
    centroid[i] /= members.length;
  }
  return centroid;
}

/**
 * Computes the average pairwise similarity among members.
 * @internal
 */
export function computePairwiseCohesion(
  members: Vector[],
  metric: SimilarityMetric,
): number {
  if (members.length <= 1) return 1.0;
  let totalSim = 0;
  let pairs = 0;
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) {
      totalSim += computeScore(members[i], members[j], metric);
      pairs++;
    }
  }
  return totalSim / pairs;
}
