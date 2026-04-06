import type { Vector, Cluster } from '../types';
import { ValidationError } from '../types';
import { toFloat32 } from '../internal/vector-utils';
import { euclideanDistance, cosineDistance } from '../math/distance';
import { cosineSimilarity } from '../math/similarity';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options for the {@link hdbscan} function.
 */
export interface HDBSCANOptions {
  /**
   * Minimum number of points required to form a cluster.
   * Smaller values detect smaller clusters but may split natural groups.
   * Default: 5.
   */
  minClusterSize?: number;

  /**
   * Number of neighbors used to compute core distances. Controls how
   * conservative the clustering is — higher values smooth out noise but
   * may miss small clusters. Default: same as `minClusterSize`.
   */
  minSamples?: number;

  /**
   * Distance metric for comparing vectors.
   * - `'euclidean'` (default) — L2 distance
   * - `'cosine'` — Cosine distance (1 - cosine similarity)
   */
  metric?: 'euclidean' | 'cosine';

  /**
   * String labels corresponding to each input embedding. When provided,
   * labels are preserved in cluster and noise results. Must have the
   * same length as the embeddings array.
   */
  labels?: string[];
}

/**
 * Result returned by {@link hdbscan}.
 */
export interface HDBSCANResult {
  /** Selected flat clusters with centroids, members, and optional labels. */
  clusters: (Cluster & { indices: number[] })[];
  /** Points not assigned to any cluster. */
  noise: {
    members: Float32Array[];
    indices: number[];
    labels?: string[];
  };
  /** Cluster assignment for each input point: cluster index (0-based) or -1 for noise. */
  labels: number[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface MSTEdge {
  from: number;
  to: number;
  weight: number;
}

/** An entry in the condensed tree. */
interface CondensedTreeEntry {
  parent: number;    // condensed cluster ID
  child: number;     // point index (if childSize===1) or condensed cluster ID
  lambda: number;    // 1/distance at which this separation happens
  childSize: number; // 1 for individual point, >1 for sub-cluster
}

/** Hierarchy merge record. */
interface HierarchyMerge {
  left: number;
  right: number;
  distance: number;
  leftSize: number;
  rightSize: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5.6: Mutual reachability distance computation
// ─────────────────────────────────────────────────────────────────────────────

type DistanceFn = (a: Vector, b: Vector) => number;

function getDistanceFn(metric: 'euclidean' | 'cosine'): DistanceFn {
  return metric === 'cosine' ? cosineDistance : euclideanDistance;
}

function computeDistanceMatrix(embeddings: Vector[], distFn: DistanceFn): Float64Array {
  const n = embeddings.length;
  const matrix = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distFn(embeddings[i], embeddings[j]);
      matrix[i * n + j] = d;
      matrix[j * n + i] = d;
    }
  }
  return matrix;
}

function computeCoreDistances(distMatrix: Float64Array, n: number, minSamples: number): Float64Array {
  const coreDistances = new Float64Array(n);
  const k = Math.min(minSamples, n - 1);

  for (let i = 0; i < n; i++) {
    const distances: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i !== j) distances.push(distMatrix[i * n + j]);
    }
    distances.sort((a, b) => a - b);
    coreDistances[i] = k > 0 ? distances[k - 1] : 0;
  }
  return coreDistances;
}

function computeMutualReachabilityMatrix(
  distMatrix: Float64Array,
  coreDistances: Float64Array,
  n: number,
): Float64Array {
  const mrd = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = Math.max(coreDistances[i], coreDistances[j], distMatrix[i * n + j]);
      mrd[i * n + j] = d;
      mrd[j * n + i] = d;
    }
  }
  return mrd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5.7: Minimum spanning tree via Prim's algorithm
// ─────────────────────────────────────────────────────────────────────────────

function buildMST(mrdMatrix: Float64Array, n: number): MSTEdge[] {
  const inMST = new Uint8Array(n);
  const minWeight = new Float64Array(n).fill(Infinity);
  const minFrom = new Int32Array(n).fill(-1);
  const edges: MSTEdge[] = [];

  inMST[0] = 1;
  for (let j = 1; j < n; j++) {
    minWeight[j] = mrdMatrix[j];
    minFrom[j] = 0;
  }

  for (let step = 0; step < n - 1; step++) {
    let bestNode = -1;
    let bestWeight = Infinity;
    for (let j = 0; j < n; j++) {
      if (!inMST[j] && minWeight[j] <= bestWeight) {
        bestWeight = minWeight[j];
        bestNode = j;
      }
    }
    if (bestNode === -1) break;

    inMST[bestNode] = 1;
    edges.push({ from: minFrom[bestNode], to: bestNode, weight: bestWeight });

    for (let j = 0; j < n; j++) {
      if (!inMST[j]) {
        const w = mrdMatrix[bestNode * n + j];
        if (w < minWeight[j]) {
          minWeight[j] = w;
          minFrom[j] = bestNode;
        }
      }
    }
  }

  edges.sort((a, b) => a.weight - b.weight);
  return edges;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5.8: Condensed cluster tree extraction
// ─────────────────────────────────────────────────────────────────────────────

class UnionFind {
  parent: Int32Array;
  size: Int32Array;
  nextId: number;

  constructor(n: number) {
    // Allocate enough space for n original points + n-1 internal nodes
    this.parent = new Int32Array(2 * n);
    this.size = new Int32Array(2 * n);
    for (let i = 0; i < n; i++) {
      this.parent[i] = i;
      this.size[i] = 1;
    }
    this.nextId = n;
  }

  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(a: number, b: number): number {
    const newId = this.nextId++;
    this.parent[a] = newId;
    this.parent[b] = newId;
    this.parent[newId] = newId;
    this.size[newId] = this.size[a] + this.size[b];
    return newId;
  }
}

/**
 * Build the single-linkage hierarchy from MST edges, then condense it.
 *
 * The condensed tree collapses nodes where one child has < minClusterSize
 * points — those points "fall out" as noise, while the big child continues
 * as the same condensed cluster.
 */
function buildCondensedTree(
  edges: MSTEdge[],
  n: number,
  minClusterSize: number,
): CondensedTreeEntry[] {
  // Step 1: Build single-linkage hierarchy
  const uf = new UnionFind(n);
  const merges: HierarchyMerge[] = [];

  for (const edge of edges) {
    const rootA = uf.find(edge.from);
    const rootB = uf.find(edge.to);
    if (rootA === rootB) continue;

    const sizeA = uf.size[rootA];
    const sizeB = uf.size[rootB];
    uf.union(rootA, rootB);
    merges.push({
      left: rootA,
      right: rootB,
      distance: edge.weight,
      leftSize: sizeA,
      rightSize: sizeB,
    });
  }

  if (merges.length === 0) return [];

  // Build children map: for hierarchy node (n + i), store its merge info
  const childrenOf = new Map<number, HierarchyMerge>();
  for (let i = 0; i < merges.length; i++) {
    childrenOf.set(n + i, merges[i]);
  }

  // Helper: collect all leaf point indices under a hierarchy node
  function getSubtreePoints(nodeId: number): number[] {
    if (nodeId < n) return [nodeId];
    const merge = childrenOf.get(nodeId);
    if (!merge) return [];
    return [...getSubtreePoints(merge.left), ...getSubtreePoints(merge.right)];
  }

  // Step 2: Condense the tree top-down
  const condensedTree: CondensedTreeEntry[] = [];
  const relabel = new Map<number, number>(); // hierarchy node → condensed cluster ID
  let nextLabel = n;

  const rootNode = n + merges.length - 1;
  relabel.set(rootNode, nextLabel++);

  // Process merges in reverse order (top-down: root first, then deeper nodes)
  for (let i = merges.length - 1; i >= 0; i--) {
    const nodeId = n + i;
    const condensedId = relabel.get(nodeId);
    if (condensedId === undefined) continue; // node is in a subtree that fell out

    const { left, right, distance, leftSize, rightSize } = merges[i];
    const lambda = distance > 0 ? 1 / distance : 1e10; // cap at large value to avoid Infinity

    if (leftSize >= minClusterSize && rightSize >= minClusterSize) {
      // Genuine split — both children become new condensed clusters
      const leftLabel = nextLabel++;
      const rightLabel = nextLabel++;
      relabel.set(left, leftLabel);
      relabel.set(right, rightLabel);
      condensedTree.push({ parent: condensedId, child: leftLabel, lambda, childSize: leftSize });
      condensedTree.push({ parent: condensedId, child: rightLabel, lambda, childSize: rightSize });
    } else if (leftSize >= minClusterSize) {
      // Left survives as same condensed cluster, right falls out
      relabel.set(left, condensedId);
      for (const p of getSubtreePoints(right)) {
        condensedTree.push({ parent: condensedId, child: p, lambda, childSize: 1 });
      }
    } else if (rightSize >= minClusterSize) {
      // Right survives as same condensed cluster, left falls out
      relabel.set(right, condensedId);
      for (const p of getSubtreePoints(left)) {
        condensedTree.push({ parent: condensedId, child: p, lambda, childSize: 1 });
      }
    } else {
      // Both too small — all points fall out
      for (const p of getSubtreePoints(left)) {
        condensedTree.push({ parent: condensedId, child: p, lambda, childSize: 1 });
      }
      for (const p of getSubtreePoints(right)) {
        condensedTree.push({ parent: condensedId, child: p, lambda, childSize: 1 });
      }
    }
  }

  return condensedTree;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5.9: Cluster stability and flat cluster extraction (EOM)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Given the condensed tree, compute stability for each cluster and run
 * Excess of Mass (EOM) selection to produce flat clusters.
 *
 * Returns the set of point indices for each selected cluster.
 */
function extractClusters(
  condensedTree: CondensedTreeEntry[],
): number[][] {
  if (condensedTree.length === 0) return [];

  // Identify all condensed cluster IDs
  const clusterIds = new Set<number>();
  for (const entry of condensedTree) {
    clusterIds.add(entry.parent);
    if (entry.childSize > 1) clusterIds.add(entry.child);
  }

  // Birth lambda for each cluster
  const birthLambda = new Map<number, number>();
  // The root condensed cluster is born at lambda 0
  // Find the root: it's a parent that is never a child with childSize > 1
  const childClusterIds = new Set<number>();
  for (const entry of condensedTree) {
    if (entry.childSize > 1) childClusterIds.add(entry.child);
  }
  for (const id of clusterIds) {
    if (!childClusterIds.has(id)) {
      birthLambda.set(id, 0); // root
    }
  }
  for (const entry of condensedTree) {
    if (entry.childSize > 1) {
      birthLambda.set(entry.child, entry.lambda);
    }
  }

  // Compute stability for each cluster
  // stability(C) = sum over all entries with parent=C of (lambda - birthLambda(C)) * childSize
  const stability = new Map<number, number>();
  for (const id of clusterIds) stability.set(id, 0);

  for (const entry of condensedTree) {
    const birth = birthLambda.get(entry.parent) ?? 0;
    const contribution = (entry.lambda - birth) * entry.childSize;
    if (contribution > 0) {
      stability.set(entry.parent, (stability.get(entry.parent) ?? 0) + contribution);
    }
  }

  // Find parent-child cluster relationships
  const clusterChildren = new Map<number, number[]>();
  for (const entry of condensedTree) {
    if (entry.childSize > 1) {
      if (!clusterChildren.has(entry.parent)) clusterChildren.set(entry.parent, []);
      clusterChildren.get(entry.parent)!.push(entry.child);
    }
  }

  // EOM selection via recursive bottom-up traversal
  const selected = new Map<number, boolean>();

  function processEOM(clusterId: number): number {
    const children = clusterChildren.get(clusterId);
    if (!children || children.length === 0) {
      // Leaf cluster — always selected
      selected.set(clusterId, true);
      return stability.get(clusterId) ?? 0;
    }

    let childStabilitySum = 0;
    for (const child of children) {
      childStabilitySum += processEOM(child);
    }

    const myStability = stability.get(clusterId) ?? 0;
    if (myStability >= childStabilitySum) {
      // Parent more stable — select parent, deselect all descendants
      selected.set(clusterId, true);
      const deselect = (id: number) => {
        selected.set(id, false);
        const ch = clusterChildren.get(id);
        if (ch) ch.forEach(deselect);
      };
      for (const child of children) deselect(child);
      return myStability;
    } else {
      // Children more stable — propagate
      selected.set(clusterId, false);
      return childStabilitySum;
    }
  }

  // Find root and run EOM
  for (const id of clusterIds) {
    if (!childClusterIds.has(id)) {
      processEOM(id);
    }
  }

  // Collect points for each selected cluster
  // A point belongs to the deepest selected cluster that contains it
  const selectedIds = [...clusterIds].filter((id) => selected.get(id));

  // Precompute parent→children map to avoid O(k*m) rescans
  const childrenByParent = new Map<number, CondensedTreeEntry[]>();
  for (const entry of condensedTree) {
    let children = childrenByParent.get(entry.parent);
    if (!children) {
      children = [];
      childrenByParent.set(entry.parent, children);
    }
    children.push(entry);
  }

  // For each selected cluster, collect all points in its subtree
  // (points that fell out of it + points in non-selected children)
  function collectAllPoints(clusterId: number): number[] {
    const points: number[] = [];
    const children = childrenByParent.get(clusterId);
    if (!children) return points;
    for (const entry of children) {
      if (entry.childSize === 1) {
        points.push(entry.child);
      } else {
        points.push(...collectAllPoints(entry.child));
      }
    }
    return points;
  }

  function collectSelectedPoints(clusterId: number): number[] {
    const points: number[] = [];
    const children = childrenByParent.get(clusterId);
    if (!children) return points;
    for (const entry of children) {
      if (entry.childSize === 1) {
        points.push(entry.child);
      } else if (selected.get(entry.child)) {
        // Child is itself selected — don't claim its points
        continue;
      } else {
        // Child not selected — its points belong to us
        points.push(...collectAllPoints(entry.child));
      }
    }
    return points;
  }

  return selectedIds.map((id) => collectSelectedPoints(id));
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 5.10: Result formatting and public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
 *
 * Automatically determines the number of clusters, handles varying-density
 * regions, and explicitly identifies noise points. O(n^2) space and time —
 * practical up to ~50k embeddings.
 *
 * @param embeddings - Array of vectors to cluster. All must have the same dimensions.
 * @param options - Clustering options (minClusterSize, minSamples, metric, labels).
 * @returns Clusters, noise points, and per-point label assignments.
 */
export function hdbscan(embeddings: Vector[], options: HDBSCANOptions = {}): HDBSCANResult {
  const {
    minClusterSize = 5,
    minSamples = minClusterSize,
    metric = 'euclidean',
    labels: inputLabels,
  } = options;

  // Handle empty input
  if (embeddings.length === 0) {
    return { clusters: [], noise: { members: [], indices: [] }, labels: [] };
  }

  // Validate labels length
  if (inputLabels && inputLabels.length !== embeddings.length) {
    throw new ValidationError(
      `Labels length (${inputLabels.length}) must match embeddings length (${embeddings.length})`,
    );
  }

  const n = embeddings.length;
  const vecs = embeddings.map(toFloat32);

  // If fewer points than minClusterSize, all are noise
  if (n < minClusterSize) {
    const noiseLabels = inputLabels ? inputLabels.slice() : undefined;
    return {
      clusters: [],
      noise: {
        members: vecs,
        indices: Array.from({ length: n }, (_, i) => i),
        labels: noiseLabels,
      },
      labels: new Array(n).fill(-1),
    };
  }

  // Step 1: Compute pairwise distances
  const distFn = getDistanceFn(metric);
  const distMatrix = computeDistanceMatrix(vecs, distFn);

  // Step 2: Compute core distances and mutual reachability
  const coreDistances = computeCoreDistances(distMatrix, n, minSamples);
  const mrdMatrix = computeMutualReachabilityMatrix(distMatrix, coreDistances, n);

  // Step 3: Build MST
  const mstEdges = buildMST(mrdMatrix, n);

  // Step 4: Build condensed tree
  const condensedTree = buildCondensedTree(mstEdges, n, minClusterSize);

  // Step 5: Extract flat clusters via EOM
  const clusterPointSets = extractClusters(condensedTree);

  // Step 6: Format results
  const pointLabels = new Array<number>(n).fill(-1);
  const clusters: HDBSCANResult['clusters'] = [];
  const dim = vecs[0].length;

  for (let ci = 0; ci < clusterPointSets.length; ci++) {
    const indices = clusterPointSets[ci];
    if (indices.length === 0) continue;

    const memberVecs: Float32Array[] = [];
    const memberIndices: number[] = [];
    const memberLabels: string[] = [];

    for (const ptIdx of indices) {
      pointLabels[ptIdx] = clusters.length; // use actual cluster index after filtering empty
      memberVecs.push(vecs[ptIdx]);
      memberIndices.push(ptIdx);
      if (inputLabels) memberLabels.push(inputLabels[ptIdx]);
    }

    // Compute centroid
    const centroid = new Float32Array(dim);
    for (const mv of memberVecs) {
      for (let d = 0; d < dim; d++) centroid[d] += mv[d];
    }
    for (let d = 0; d < dim; d++) centroid[d] /= memberVecs.length;

    // Compute cohesion (average pairwise cosine similarity)
    let cohesion = 1;
    if (memberVecs.length > 1) {
      let totalSim = 0;
      let pairs = 0;
      for (let i = 0; i < memberVecs.length; i++) {
        for (let j = i + 1; j < memberVecs.length; j++) {
          totalSim += cosineSimilarity(memberVecs[i], memberVecs[j]);
          pairs++;
        }
      }
      cohesion = pairs > 0 ? totalSim / pairs : 1;
    }

    const clusterResult: Cluster & { indices: number[] } = {
      centroid,
      members: memberVecs,
      size: memberVecs.length,
      cohesion,
      indices: memberIndices,
    };
    if (inputLabels) clusterResult.labels = memberLabels;
    clusters.push(clusterResult);
  }

  // Collect noise points
  const noiseMembers: Float32Array[] = [];
  const noiseIndices: number[] = [];
  const noiseLabels: string[] = [];

  for (let i = 0; i < n; i++) {
    if (pointLabels[i] === -1) {
      noiseMembers.push(vecs[i]);
      noiseIndices.push(i);
      if (inputLabels) noiseLabels.push(inputLabels[i]);
    }
  }

  const noise: HDBSCANResult['noise'] = {
    members: noiseMembers,
    indices: noiseIndices,
  };
  if (inputLabels) noise.labels = noiseLabels;

  return { clusters, noise, labels: pointLabels };
}
