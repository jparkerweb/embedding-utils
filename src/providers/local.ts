import type {
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  LocalProviderConfig,
  CacheProvider,
} from '../types';
import { EmbeddingUtilsError, ModelNotFoundError } from '../types';
import { truncateDimensions } from '../math/dimensions';
import { createLRUCache } from '../storage/cache';

const DEFAULT_MODEL = 'Xenova/all-MiniLM-L12-v2';

/** Minimal interface for the @huggingface/transformers pipeline function result. */
interface FeatureExtractionPipeline {
  (inputs: string[], options?: { pooling?: string; normalize?: boolean }): Promise<{
    tolist(): number[][];
  }>;
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
  const documentPrefix = config?.documentPrefix ?? '';
  const queryPrefix = config?.queryPrefix ?? '';

  let pipelineInstance: FeatureExtractionPipeline | null = null;
  let pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;
  const cache: CacheProvider = createLRUCache(config?.cache ?? { maxSize: 1000 });

  async function getPipeline(): Promise<FeatureExtractionPipeline> {
    if (pipelineInstance) return pipelineInstance;
    if (pipelinePromise) return pipelinePromise;

    pipelinePromise = (async () => {
      try {
        const transformers = await import('@huggingface/transformers');
        const pipe = await transformers.pipeline('feature-extraction', model, {
          dtype: config?.precision ?? 'fp32',
        });
        pipelineInstance = pipe as unknown as FeatureExtractionPipeline;
        return pipelineInstance;
      } catch (error: unknown) {
        pipelinePromise = null;
        const err = error instanceof Error ? error : null;
        const code = (error as { code?: string })?.code;
        if (
          code === 'ERR_MODULE_NOT_FOUND' ||
          code === 'MODULE_NOT_FOUND' ||
          err?.message?.includes('Cannot find module')
        ) {
          throw new ModelNotFoundError(
            '@huggingface/transformers is not installed. ' +
              'Install it with: npm install @huggingface/transformers',
          );
        }
        throw error;
      }
    })();

    return pipelinePromise;
  }

  function getCacheKey(inputs: string[], dimensions?: number): string {
    return model + ':' + (dimensions ?? '') + ':' + JSON.stringify(inputs);
  }

  return {
    name: 'local',
    dimensions: null,

    async embed(
      input: string | string[],
      options?: EmbedOptions,
    ): Promise<EmbeddingResult> {
      if (options?.signal?.aborted) {
        throw new EmbeddingUtilsError('Aborted');
      }

      const inputs = Array.isArray(input) ? input : [input];

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

      // Check cache
      const cacheKey = getCacheKey(prefixedInputs, options?.dimensions);
      const cached = await cache.get(cacheKey);
      if (cached) {
        let embeddings = cached;
        if (options?.dimensions) {
          embeddings = truncateDimensions(embeddings, options.dimensions);
        }
        return {
          embeddings,
          model,
          dimensions: embeddings[0].length,
        };
      }

      const pipe = await getPipeline();
      const output = await pipe(prefixedInputs, { pooling: 'mean', normalize: true });
      let embeddings: number[][] = output.tolist();

      // Cache the raw embeddings
      await cache.set(cacheKey, embeddings);

      // Truncate if requested
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
