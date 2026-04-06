import { computeScore } from '../internal/metrics';
import { toFloat32 } from '../internal/vector-utils';
import type { SimilarityMetric, StoredItem, Vector } from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface HNSWOptions {
  /** Similarity metric. Default: 'cosine'. */
  metric?: SimilarityMetric;
  /** Max number of connections per node per layer. Default: 16. */
  M?: number;
  /** Size of the dynamic candidate list during construction. Default: 200. */
  efConstruction?: number;
}

export interface HNSWSearchOptions {
  /** Number of results to return. Default: 10. */
  topK?: number;
  /** Size of the dynamic candidate list during search. Default: 50. */
  efSearch?: number;
  /** Filter predicate — only include items where this returns true. */
  filter?: (item: StoredItem) => boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal node structure
// ─────────────────────────────────────────────────────────────────────────────

interface HNSWNode {
  id: string;
  vector: Float32Array;
  metadata?: Record<string, unknown>;
  /** Neighbor indices per layer. neighbors[layer] = Set<nodeIndex> */
  neighbors: Map<number, Set<number>>;
  /** The layer this node was assigned to (max layer). */
  level: number;
  /** Whether this node has been soft-deleted. */
  deleted: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Priority queue helpers (min-heap by score for beam search)
// ─────────────────────────────────────────────────────────────────────────────

interface ScoredIndex {
  index: number;
  score: number;
}

function insertSorted(arr: ScoredIndex[], item: ScoredIndex): void {
  // Binary search for insertion point (ascending by score)
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid].score < item.score) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, item);
}

// ─────────────────────────────────────────────────────────────────────────────
// HNSWIndex
// ─────────────────────────────────────────────────────────────────────────────

const HNSW_FORMAT_VERSION = 1;

/**
 * Hierarchical Navigable Small World (HNSW) index for approximate nearest
 * neighbor search.
 *
 * Supports configurable M, efConstruction, efSearch, all four similarity
 * metrics, string IDs with metadata, and binary serialization.
 *
 * @example
 * const index = new HNSWIndex({ metric: 'cosine' });
 * index.add('doc1', embedding1, { title: 'Hello' });
 * const results = index.search(queryEmbedding, { topK: 5 });
 */
export class HNSWIndex {
  private nodes: HNSWNode[] = [];
  private idToIndex = new Map<string, number>();
  private entryPoint: number = -1;
  private maxLevel: number = -1;
  private activeCount: number = 0;

  readonly metric: SimilarityMetric;
  readonly M: number;
  readonly efConstruction: number;
  private readonly Mmax0: number;
  private readonly mL: number;

  constructor(options?: HNSWOptions) {
    this.metric = options?.metric ?? 'cosine';
    this.M = options?.M ?? 16;
    this.efConstruction = options?.efConstruction ?? 200;
    this.Mmax0 = this.M * 2;
    this.mL = 1 / Math.log(this.M);
  }

  /** Number of active (non-deleted) items in the index. */
  get size(): number {
    return this.activeCount;
  }

  /** Add a single item. Overwrites if ID already exists. */
  add(id: string, vector: Vector, metadata?: Record<string, unknown>): void {
    const vec = toFloat32(vector);

    // If ID exists, overwrite by removing first
    if (this.idToIndex.has(id)) {
      this.remove(id);
    }

    const level = this.randomLevel();
    const nodeIndex = this.nodes.length;
    const node: HNSWNode = {
      id,
      vector: vec,
      metadata,
      neighbors: new Map(),
      level,
      deleted: false,
    };

    // Initialize neighbor sets for each layer
    for (let l = 0; l <= level; l++) {
      node.neighbors.set(l, new Set());
    }

    this.nodes.push(node);
    this.idToIndex.set(id, nodeIndex);
    this.activeCount++;

    // First node — set as entry point
    if (this.entryPoint === -1) {
      this.entryPoint = nodeIndex;
      this.maxLevel = level;
      return;
    }

    // Traverse from top layer down to level+1 using greedy search (ef=1)
    let currentEntry = this.entryPoint;
    for (let l = this.maxLevel; l > level; l--) {
      const nearest = this.searchLayer(vec, currentEntry, 1, l);
      if (nearest.length > 0) {
        currentEntry = nearest[0].index;
      }
    }

    // For each layer from min(level, maxLevel) down to 0, find and connect neighbors
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const candidates = this.searchLayer(vec, currentEntry, this.efConstruction, l);
      const maxConnections = l === 0 ? this.Mmax0 : this.M;
      const neighbors = this.selectNeighbors(candidates, maxConnections);

      // Connect new node to neighbors
      const nodeNeighbors = node.neighbors.get(l)!;
      for (const neighbor of neighbors) {
        nodeNeighbors.add(neighbor.index);
        // Connect neighbor back to new node
        const neighborNode = this.nodes[neighbor.index];
        let neighborSet = neighborNode.neighbors.get(l);
        if (!neighborSet) {
          neighborSet = new Set();
          neighborNode.neighbors.set(l, neighborSet);
        }
        neighborSet.add(nodeIndex);

        // Prune neighbor's connections if they exceed max
        if (neighborSet.size > maxConnections) {
          this.pruneConnections(neighbor.index, l, maxConnections);
        }
      }

      if (candidates.length > 0) {
        currentEntry = candidates[0].index;
      }
    }

    // Update entry point if new node has higher level
    if (level > this.maxLevel) {
      this.entryPoint = nodeIndex;
      this.maxLevel = level;
    }
  }

  /** Add multiple items at once. */
  addBatch(items: Array<{ id: string; vector: Vector; metadata?: Record<string, unknown> }>): void {
    for (const item of items) {
      this.add(item.id, item.vector, item.metadata);
    }
  }

  /** Get an item by ID. Returns undefined if not found or deleted. */
  get(id: string): StoredItem | undefined {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) return undefined;
    const node = this.nodes[idx];
    if (node.deleted) return undefined;
    return {
      id: node.id,
      embedding: node.vector,
      ...(node.metadata !== undefined ? { metadata: node.metadata } : {}),
    };
  }

  /** Remove an item by ID (lazy deletion). */
  remove(id: string): boolean {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) return false;
    const node = this.nodes[idx];
    if (node.deleted) return false;
    node.deleted = true;
    this.activeCount--;
    this.idToIndex.delete(id);

    // If we removed the entry point, find a new one
    if (idx === this.entryPoint) {
      this.findNewEntryPoint();
    }

    return true;
  }

  /** Search for the most similar items to a query vector. */
  search(
    query: Vector,
    options?: HNSWSearchOptions,
  ): Array<StoredItem & { score: number }> {
    const topK = options?.topK ?? 10;
    const efSearch = options?.efSearch ?? 50;
    const filter = options?.filter;

    if (this.entryPoint === -1 || this.activeCount === 0) {
      return [];
    }

    const q = toFloat32(query);

    // Greedy search from top to layer 1
    let currentEntry = this.entryPoint;
    for (let l = this.maxLevel; l > 0; l--) {
      const nearest = this.searchLayer(q, currentEntry, 1, l);
      if (nearest.length > 0) {
        currentEntry = nearest[0].index;
      }
    }

    // Search at layer 0 with efSearch beam width
    const ef = Math.max(efSearch, topK);
    const candidates = this.searchLayer(q, currentEntry, ef, 0);

    // Apply filter and collect results
    const results: Array<StoredItem & { score: number }> = [];
    for (const candidate of candidates) {
      const node = this.nodes[candidate.index];
      if (node.deleted) continue;

      const item: StoredItem = {
        id: node.id,
        embedding: node.vector,
        ...(node.metadata !== undefined ? { metadata: node.metadata } : {}),
      };

      if (filter && !filter(item)) continue;

      results.push({ ...item, score: candidate.score });

      if (results.length >= topK) break;
    }

    return results;
  }

  /** Remove all items and reset the index. */
  clear(): void {
    this.nodes = [];
    this.idToIndex.clear();
    this.entryPoint = -1;
    this.maxLevel = -1;
    this.activeCount = 0;
  }

  /** Serialize the index to a binary Uint8Array. */
  serialize(): Uint8Array {
    // Collect active nodes
    const activeNodes: { node: HNSWNode; originalIndex: number }[] = [];
    const oldToNew = new Map<number, number>();

    for (let i = 0; i < this.nodes.length; i++) {
      if (!this.nodes[i].deleted) {
        oldToNew.set(i, activeNodes.length);
        activeNodes.push({ node: this.nodes[i], originalIndex: i });
      }
    }

    // Encode header as JSON
    const dimensions = activeNodes.length > 0 ? activeNodes[0].node.vector.length : 0;
    const header = JSON.stringify({
      metric: this.metric,
      M: this.M,
      efConstruction: this.efConstruction,
      dimensions,
      nodeCount: activeNodes.length,
      maxLevel: this.maxLevel,
      entryPoint: this.entryPoint === -1 ? -1 : (oldToNew.get(this.entryPoint) ?? -1),
    });

    const headerBytes = new TextEncoder().encode(header);

    // Encode nodes
    const nodeBuffers: Uint8Array[] = [];
    let totalNodeBytes = 0;

    for (const { node } of activeNodes) {
      const idBytes = new TextEncoder().encode(node.id);
      const metaBytes = node.metadata
        ? new TextEncoder().encode(JSON.stringify(node.metadata))
        : new Uint8Array(0);

      // Collect neighbor data for all layers
      const layerData: { layer: number; neighbors: number[] }[] = [];
      for (let l = 0; l <= node.level; l++) {
        const neighbors = node.neighbors.get(l);
        if (neighbors && neighbors.size > 0) {
          const mapped = Array.from(neighbors)
            .map((idx) => oldToNew.get(idx))
            .filter((idx): idx is number => idx !== undefined);
          layerData.push({ layer: l, neighbors: mapped });
        }
      }

      // Node format:
      // 4 bytes: id length
      // N bytes: id
      // 4 bytes: metadata length (0 if none)
      // N bytes: metadata JSON
      // 4 bytes: vector byte length
      // N bytes: vector (float32)
      // 4 bytes: node level
      // 4 bytes: number of layers with neighbors
      // Per layer:
      //   4 bytes: layer index
      //   4 bytes: neighbor count
      //   4*count bytes: neighbor indices
      const vectorBytes = node.vector.byteLength;
      let layerByteLen = 0;
      for (const ld of layerData) {
        layerByteLen += 4 + 4 + ld.neighbors.length * 4; // layer + count + indices
      }

      const nodeSize =
        4 + idBytes.length + // id
        4 + metaBytes.length + // metadata
        4 + vectorBytes + // vector
        4 + // level
        4 + layerByteLen; // layers

      const buf = new ArrayBuffer(nodeSize);
      const view = new DataView(buf);
      let offset = 0;

      // ID
      view.setUint32(offset, idBytes.length, true);
      offset += 4;
      new Uint8Array(buf, offset, idBytes.length).set(idBytes);
      offset += idBytes.length;

      // Metadata
      view.setUint32(offset, metaBytes.length, true);
      offset += 4;
      if (metaBytes.length > 0) {
        new Uint8Array(buf, offset, metaBytes.length).set(metaBytes);
        offset += metaBytes.length;
      }

      // Vector
      view.setUint32(offset, vectorBytes, true);
      offset += 4;
      new Uint8Array(buf, offset, vectorBytes).set(new Uint8Array(node.vector.buffer, node.vector.byteOffset, vectorBytes));
      offset += vectorBytes;

      // Level
      view.setUint32(offset, node.level, true);
      offset += 4;

      // Layers
      view.setUint32(offset, layerData.length, true);
      offset += 4;

      for (const ld of layerData) {
        view.setUint32(offset, ld.layer, true);
        offset += 4;
        view.setUint32(offset, ld.neighbors.length, true);
        offset += 4;
        for (const n of ld.neighbors) {
          view.setUint32(offset, n, true);
          offset += 4;
        }
      }

      nodeBuffers.push(new Uint8Array(buf));
      totalNodeBytes += nodeSize;
    }

    // Final buffer: 1 (version) + 4 (header length) + header + nodes
    const totalSize = 1 + 4 + headerBytes.length + totalNodeBytes;
    const result = new Uint8Array(totalSize);
    let pos = 0;

    // Version byte
    result[pos++] = HNSW_FORMAT_VERSION;

    // Header length + header
    new DataView(result.buffer).setUint32(pos, headerBytes.length, true);
    pos += 4;
    result.set(headerBytes, pos);
    pos += headerBytes.length;

    // Nodes
    for (const nb of nodeBuffers) {
      result.set(nb, pos);
      pos += nb.length;
    }

    return result;
  }

  /** Deserialize an index from binary data. */
  static deserialize(data: Uint8Array): HNSWIndex {
    if (data.length < 5) {
      throw new Error('Corrupted HNSW data: too short');
    }

    let pos = 0;

    // Version check
    const version = data[pos++];
    if (version !== HNSW_FORMAT_VERSION) {
      throw new Error(`Unsupported HNSW format version: ${version}`);
    }

    // Header
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const headerLen = view.getUint32(pos, true);
    pos += 4;

    if (pos + headerLen > data.length) {
      throw new Error('Corrupted HNSW data: header extends beyond data');
    }

    const headerStr = new TextDecoder().decode(data.slice(pos, pos + headerLen));
    pos += headerLen;

    let header: {
      metric: SimilarityMetric;
      M: number;
      efConstruction: number;
      dimensions: number;
      nodeCount: number;
      maxLevel: number;
      entryPoint: number;
    };

    try {
      header = JSON.parse(headerStr);
    } catch {
      throw new Error('Corrupted HNSW data: invalid header JSON');
    }

    const index = new HNSWIndex({
      metric: header.metric,
      M: header.M,
      efConstruction: header.efConstruction,
    });

    index.maxLevel = header.maxLevel;
    index.entryPoint = header.entryPoint;

    // Read nodes
    for (let n = 0; n < header.nodeCount; n++) {
      if (pos + 4 > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n}`);
      }

      // ID
      const idLen = view.getUint32(pos, true);
      pos += 4;
      if (pos + idLen > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} id`);
      }
      const id = new TextDecoder().decode(data.slice(pos, pos + idLen));
      pos += idLen;

      // Metadata
      if (pos + 4 > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} metadata length`);
      }
      const metaLen = view.getUint32(pos, true);
      pos += 4;
      let metadata: Record<string, unknown> | undefined;
      if (metaLen > 0) {
        if (pos + metaLen > data.length) {
          throw new Error(`Corrupted HNSW data: unexpected end at node ${n} metadata`);
        }
        metadata = JSON.parse(new TextDecoder().decode(data.slice(pos, pos + metaLen)));
        pos += metaLen;
      }

      // Vector
      if (pos + 4 > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} vector length`);
      }
      const vectorByteLen = view.getUint32(pos, true);
      pos += 4;
      if (pos + vectorByteLen > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} vector`);
      }
      const vector = new Float32Array(vectorByteLen / 4);
      // Copy bytes to properly aligned buffer
      const vectorSrc = data.slice(pos, pos + vectorByteLen);
      new Uint8Array(vector.buffer).set(vectorSrc);
      pos += vectorByteLen;

      // Level
      if (pos + 4 > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} level`);
      }
      const level = view.getUint32(pos, true);
      pos += 4;

      // Neighbor layers
      if (pos + 4 > data.length) {
        throw new Error(`Corrupted HNSW data: unexpected end at node ${n} layer count`);
      }
      const layerCount = view.getUint32(pos, true);
      pos += 4;

      const neighbors = new Map<number, Set<number>>();
      // Initialize all layers up to node level
      for (let l = 0; l <= level; l++) {
        neighbors.set(l, new Set());
      }

      for (let lc = 0; lc < layerCount; lc++) {
        if (pos + 8 > data.length) {
          throw new Error(`Corrupted HNSW data: unexpected end at node ${n} layer ${lc}`);
        }
        const layer = view.getUint32(pos, true);
        pos += 4;
        const neighborCount = view.getUint32(pos, true);
        pos += 4;
        if (pos + neighborCount * 4 > data.length) {
          throw new Error(`Corrupted HNSW data: unexpected end at node ${n} layer ${lc} neighbors`);
        }
        const set = neighbors.get(layer) ?? new Set<number>();
        for (let nc = 0; nc < neighborCount; nc++) {
          set.add(view.getUint32(pos, true));
          pos += 4;
        }
        neighbors.set(layer, set);
      }

      const node: HNSWNode = { id, vector, metadata, neighbors, level, deleted: false };
      const nodeIndex = index.nodes.length;
      index.nodes.push(node);
      index.idToIndex.set(id, nodeIndex);
      index.activeCount++;
    }

    return index;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private methods
  // ───────────────────────────────────────────────────────────────────────────

  /** Generate a random level using exponential distribution. */
  private randomLevel(): number {
    return Math.floor(-Math.log(Math.random()) * this.mL);
  }

  /**
   * Greedy search with beam width `ef` at a specific layer.
   * Returns candidates sorted by score descending (best first).
   */
  private searchLayer(
    query: Float32Array,
    entryPointIdx: number,
    ef: number,
    layer: number,
  ): ScoredIndex[] {
    const entryNode = this.nodes[entryPointIdx];
    if (!entryNode || !entryNode.neighbors.has(layer)) {
      return [];
    }

    const entryScore = computeScore(query, entryNode.vector, this.metric);

    // candidates: sorted ascending by score (worst first for easy pop)
    // results: sorted ascending by score (worst first for trimming)
    const visited = new Set<number>();
    visited.add(entryPointIdx);

    // Use sorted arrays as priority queues
    // candidates sorted ascending by score
    let candidates: ScoredIndex[] = [{ index: entryPointIdx, score: entryScore }];
    // results sorted ascending by score
    let results: ScoredIndex[] = [{ index: entryPointIdx, score: entryScore }];

    while (candidates.length > 0) {
      // Get best candidate (highest score = last in ascending array)
      const current = candidates.pop()!;

      // Get worst result
      const worstResult = results[0];

      // If best candidate is worse than worst result, we're done
      if (current.score < worstResult.score && results.length >= ef) {
        break;
      }

      // Explore neighbors
      const neighbors = this.nodes[current.index].neighbors.get(layer);
      if (!neighbors) continue;

      for (const neighborIdx of neighbors) {
        if (visited.has(neighborIdx)) continue;
        visited.add(neighborIdx);

        const neighborNode = this.nodes[neighborIdx];
        if (!neighborNode) continue;

        const score = computeScore(query, neighborNode.vector, this.metric);

        if (results.length < ef || score > results[0].score) {
          insertSorted(candidates, { index: neighborIdx, score });
          insertSorted(results, { index: neighborIdx, score });

          if (results.length > ef) {
            results.shift(); // Remove worst
          }
        }
      }
    }

    // Return results sorted descending (best first)
    results.reverse();
    return results;
  }

  /** Select the best M neighbors from candidates. */
  private selectNeighbors(candidates: ScoredIndex[], maxConnections: number): ScoredIndex[] {
    // Simple selection: take the top maxConnections by score
    return candidates.slice(0, maxConnections);
  }

  /** Prune a node's connections at a layer to maxConnections. */
  private pruneConnections(nodeIdx: number, layer: number, maxConnections: number): void {
    const node = this.nodes[nodeIdx];
    const neighbors = node.neighbors.get(layer);
    if (!neighbors || neighbors.size <= maxConnections) return;

    // Score all neighbors and keep the best
    const scored: ScoredIndex[] = [];
    for (const nIdx of neighbors) {
      const nNode = this.nodes[nIdx];
      if (!nNode) continue;
      const score = computeScore(node.vector, nNode.vector, this.metric);
      scored.push({ index: nIdx, score });
    }

    // Sort descending by score
    scored.sort((a, b) => b.score - a.score);

    // Keep only the best
    const newNeighbors = new Set<number>();
    for (let i = 0; i < Math.min(maxConnections, scored.length); i++) {
      newNeighbors.add(scored[i].index);
    }

    // Remove back-references for pruned neighbors
    for (const oldN of neighbors) {
      if (!newNeighbors.has(oldN)) {
        const oldNode = this.nodes[oldN];
        if (oldNode) {
          const backSet = oldNode.neighbors.get(layer);
          if (backSet) {
            backSet.delete(nodeIdx);
          }
        }
      }
    }

    node.neighbors.set(layer, newNeighbors);
  }

  /** Find a new entry point after the current one is deleted. */
  private findNewEntryPoint(): void {
    this.entryPoint = -1;
    this.maxLevel = -1;
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      if (!node.deleted && node.level > this.maxLevel) {
        this.maxLevel = node.level;
        this.entryPoint = i;
      }
    }
  }
}
