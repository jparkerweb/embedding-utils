import type { ClusteringConfig } from '../types';

/**
 * Built-in clustering configuration presets.
 *
 * These presets provide tuned combinations of similarity threshold, minimum
 * cluster size, and maximum cluster count for common use cases. Use them with
 * {@link getPreset} or spread directly into {@link clusterEmbeddings}.
 *
 * | Preset | Threshold | Min Size | Max Clusters | Use Case |
 * |--------|-----------|----------|--------------|----------|
 * | HIGH_PRECISION | 0.95 | 3 | 10 | Tight, highly cohesive groups (near-duplicate detection) |
 * | BALANCED | 0.85 | 5 | 5 | General-purpose topic discovery |
 * | PERFORMANCE | 0.75 | 10 | 3 | Fast, broad groupings on large datasets |
 * | LEGACY | -- | -- | -- | Clustering disabled: returns all embeddings in a single cluster |
 *
 * The LEGACY preset sets `similarityThreshold` to 0, `minClusterSize` to 1, and
 * `maxClusters` to 1, which effectively disables clustering and returns all
 * embeddings in a single cluster. This is useful for backwards compatibility with
 * pipelines that predated clustering support.
 */
export const CLUSTERING_PRESETS = {
  HIGH_PRECISION: {
    similarityThreshold: 0.95,
    minClusterSize: 3,
    maxClusters: 10,
  } as ClusteringConfig,

  BALANCED: {
    similarityThreshold: 0.85,
    minClusterSize: 5,
    maxClusters: 5,
  } as ClusteringConfig,

  PERFORMANCE: {
    similarityThreshold: 0.75,
    minClusterSize: 10,
    maxClusters: 3,
  } as ClusteringConfig,

  LEGACY: {
    similarityThreshold: 0,
    minClusterSize: 1,
    maxClusters: 1,
  } as ClusteringConfig,
} as const;

/**
 * Returns a copy of a named clustering preset configuration.
 *
 * This is the recommended way to use presets, since it returns a fresh copy
 * that you can safely spread or modify without affecting the original constants.
 *
 * @param name - Preset name: 'high-precision', 'balanced', 'performance', or 'legacy'
 * @returns A ClusteringConfig object for the given preset
 *
 * @example
 * // Use a preset directly
 * const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));
 *
 * @example
 * // Customize a preset
 * const config = { ...getPreset('high-precision'), maxClusters: 20 };
 *
 * @example
 * // Disable clustering (single cluster output)
 * const clusters = clusterEmbeddings(embeddings, getPreset('legacy'));
 */
export function getPreset(
  name: 'high-precision' | 'balanced' | 'performance' | 'legacy',
): ClusteringConfig {
  switch (name) {
    case 'high-precision':
      return { ...CLUSTERING_PRESETS.HIGH_PRECISION };
    case 'balanced':
      return { ...CLUSTERING_PRESETS.BALANCED };
    case 'performance':
      return { ...CLUSTERING_PRESETS.PERFORMANCE };
    case 'legacy':
      return { ...CLUSTERING_PRESETS.LEGACY };
  }
}
