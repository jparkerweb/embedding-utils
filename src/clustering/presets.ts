import type { ClusteringConfig } from '../types';

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
} as const;

/**
 * Returns a copy of a named clustering preset configuration.
 * @param name - Preset name: 'high-precision', 'balanced', or 'performance'
 * @returns A ClusteringConfig object for the given preset
 * @example
 * const config = getPreset('balanced');
 * // { similarityThreshold: 0.85, minClusterSize: 5, maxClusters: 5 }
 */
export function getPreset(
  name: 'high-precision' | 'balanced' | 'performance',
): ClusteringConfig {
  switch (name) {
    case 'high-precision':
      return { ...CLUSTERING_PRESETS.HIGH_PRECISION };
    case 'balanced':
      return { ...CLUSTERING_PRESETS.BALANCED };
    case 'performance':
      return { ...CLUSTERING_PRESETS.PERFORMANCE };
  }
}
