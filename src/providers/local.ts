import type {
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  LocalProviderConfig,
  CacheProvider,
} from '../types';
import { EmbeddingUtilsError, ModelNotFoundError } from '../types';
import { truncateDimensions } from '../math/dimensions';
import { toFloat32 } from '../internal/vector-utils';
import { createInstanceRegistry } from '../internal/instance-registry';
import { createLRUCache } from '../storage/cache';
import { getModelInfo } from '../models/manager';

type PoolingMethod = 'mean' | 'cls' | 'last_token';

const DEFAULT_MODEL = 'Xenova/all-MiniLM-L12-v2';

/** Minimal interface for the @huggingface/transformers pipeline function result. */
interface FeatureExtractionPipeline {
  (
    inputs: string[],
    options?: { pooling?: PoolingMethod; normalize?: boolean }
  ): Promise<{
    tolist(): number[][];
  }>;
  /** Releases the underlying ONNX session (present on transformers.js pipelines). */
  dispose?(): Promise<void>;
}

/**
 * Process-level registry of feature-extraction pipelines, keyed by everything
 * that affects pipeline construction. Pipeline construction creates a native
 * ONNX InferenceSession (~1.5s and hundreds of MB per call for typical
 * models), and @huggingface/transformers does not cache sessions — so without
 * this, every `createLocalProvider` call with the same config paid that cost
 * again. Pooling and prefixes are deliberately NOT part of the key: they are
 * per-inference arguments, so providers that differ only in pooling/prefixes
 * can safely share one session.
 */
const MAX_SHARED_PIPELINES = 4;
const pipelineRegistry = createInstanceRegistry<FeatureExtractionPipeline>(MAX_SHARED_PIPELINES);

function pipelineKey(model: string, config?: LocalProviderConfig): string {
  return JSON.stringify([
    model,
    config?.precision ?? 'fp32',
    config?.device ?? '',
    config?.modelPath ?? '',
    config?.cacheDir ?? '',
    config?.allowRemoteModels ?? true,
  ]);
}

/**
 * Disposes every shared pipeline held by the process-level registry and
 * empties it. Call on shutdown or between tests. Providers created before
 * this call must not be used afterwards (their session is released); new
 * providers construct fresh pipelines.
 */
export async function disposeLocalPipelines(): Promise<void> {
  const pipelines = await pipelineRegistry.clear();
  for (const pipe of pipelines) {
    try {
      await pipe.dispose?.();
    } catch {
      // Best-effort: a failed dispose must not mask the others.
    }
  }
}

/**
 * Creates a local embedding provider using @huggingface/transformers (ONNX).
 * @param config - Optional configuration: model, precision, prefixes
 * @returns An EmbeddingProvider that runs inference locally
 * @example
 * const provider = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });
 * const result = await provider.embed('hello world');
 */
export function createLocalProvider(config?: LocalProviderConfig): EmbeddingProvider {
  const model = config?.model ?? DEFAULT_MODEL;
  // Resolve pooling and prefixes from explicit config first, then the model's
  // registry metadata, then sensible defaults. Pooling especially must match
  // how the model was trained (e.g. BGE/mxbai use 'cls') or embeddings are
  // silently wrong; asymmetric models (E5, BGE) need their prefixes applied.
  const registryInfo = getModelInfo(model);
  const documentPrefix = config?.documentPrefix ?? registryInfo?.prefixes?.document ?? '';
  const queryPrefix = config?.queryPrefix ?? registryInfo?.prefixes?.query ?? '';
  const pooling: PoolingMethod = config?.pooling ?? registryInfo?.pooling ?? 'mean';

  const cache: CacheProvider = createLRUCache(config?.cache ?? { maxSize: 1000 });

  async function buildPipeline(): Promise<FeatureExtractionPipeline> {
    try {
      const transformers = await import('@huggingface/transformers');

      // Honor the advertised environment config. Previously these declared
      // LocalProviderConfig fields were ignored (latent bug); wiring them
      // here lets callers point transformers at local models / custom caches
      // and toggle remote downloads.
      transformers.env.allowRemoteModels = config?.allowRemoteModels ?? true;
      if (config?.modelPath) {
        transformers.env.localModelPath = config.modelPath;
      }
      if (config?.cacheDir) {
        transformers.env.cacheDir = config.cacheDir;
      }

      // Build pipeline options. precision maps to dtype (now incl. 'q4').
      // device is only passed for non-webgpu providers, mirroring
      // semantic-chunking's guard — passing device for 'webgpu' breaks it.
      const pipelineOptions: Record<string, unknown> = {
        dtype: config?.precision ?? 'fp32',
      };
      if (config?.device && config.device !== 'webgpu') {
        pipelineOptions.device = config.device;
      }

      const pipe = await transformers.pipeline(
        'feature-extraction',
        model,
        pipelineOptions as Parameters<typeof transformers.pipeline>[2]
      );
      return pipe as unknown as FeatureExtractionPipeline;
    } catch (error: unknown) {
      const err = error instanceof Error ? error : null;
      const code = (error as { code?: string })?.code;
      if (
        code === 'ERR_MODULE_NOT_FOUND' ||
        code === 'MODULE_NOT_FOUND' ||
        err?.message?.includes('Cannot find module')
      ) {
        throw new ModelNotFoundError(
          '@huggingface/transformers is not installed. ' +
            'Install it with: npm install @huggingface/transformers'
        );
      }
      throw error;
    }
  }

  // Shared path (default): one pipeline per unique construction config for
  // the whole process, via the bounded registry. Opt-out with `reuse: false`
  // for a private session (e.g. to dispose it independently).
  const reuse = config?.reuse ?? true;
  let privatePipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

  function getPipeline(): Promise<FeatureExtractionPipeline> {
    if (reuse) {
      return pipelineRegistry.getOrCreate(pipelineKey(model, config), buildPipeline);
    }
    if (!privatePipelinePromise) {
      privatePipelinePromise = buildPipeline().catch((error: unknown) => {
        privatePipelinePromise = null; // allow retry after failure
        throw error;
      });
    }
    return privatePipelinePromise;
  }

  // Per-input cache key: keyed on the prefixed text so repeated/overlapping
  // texts hit the cache regardless of which batch they arrive in.
  function getCacheKey(text: string, dimensions?: number): string {
    return `${model}:${dimensions ?? ''}:${text}`;
  }

  return {
    name: 'local',
    dimensions: null,

    async embed(input: string | string[], options?: EmbedOptions): Promise<EmbeddingResult> {
      if (options?.signal?.aborted) {
        throw new EmbeddingUtilsError('Aborted');
      }

      const inputs = Array.isArray(input) ? input : [input];

      // Empty-input guard: bail before any cache/pipeline work so we never
      // dereference embeddings[0] on an empty result.
      if (inputs.length === 0) {
        return {
          embeddings: [],
          model,
          dimensions: 0,
        };
      }

      // Apply prefixes based on inputType
      const prefixedInputs = inputs.map((text) => {
        if (options?.inputType === 'document' && documentPrefix) {
          return documentPrefix + text;
        }
        if (options?.inputType === 'query' && queryPrefix) {
          return queryPrefix + text;
        }
        return text;
      });

      // Per-input cache lookup. Each prefixed text is keyed individually so
      // repeated/overlapping texts (within or across batches) hit the cache.
      // Raw (untruncated) embeddings are stored as single-element arrays.
      const results: (Float32Array | undefined)[] = new Array(prefixedInputs.length);
      const missTexts: string[] = [];
      const missIndices: number[] = [];

      for (let i = 0; i < prefixedInputs.length; i++) {
        const key = getCacheKey(prefixedInputs[i], options?.dimensions);
        const cached = await cache.get(key);
        if (cached && cached.length > 0) {
          results[i] = cached[0];
        } else {
          missTexts.push(prefixedInputs[i]);
          missIndices.push(i);
        }
      }

      // Only run the pipeline on cache misses.
      if (missTexts.length > 0) {
        const pipe = await getPipeline();
        const output = await pipe(missTexts, { pooling, normalize: true });
        const missEmbeddings: Float32Array[] = output.tolist().map(toFloat32);

        for (let j = 0; j < missIndices.length; j++) {
          const originalIndex = missIndices[j];
          const embedding = missEmbeddings[j];
          results[originalIndex] = embedding;
          // Store each raw embedding under its own per-text key.
          const key = getCacheKey(prefixedInputs[originalIndex], options?.dimensions);
          await cache.set(key, [embedding]);
        }
      }

      // Reassemble in original input order.
      let embeddings: Float32Array[] = results as Float32Array[];

      // Truncate if requested.
      if (options?.dimensions) {
        embeddings = truncateDimensions(embeddings, options.dimensions);
      }

      return {
        embeddings,
        model,
        dimensions: embeddings[0].length,
      };
    },
  };
}
