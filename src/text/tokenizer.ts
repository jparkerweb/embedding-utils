import type { TokenizerInfo, CreateTokenizerOptions, LocalTokenizer } from '../types';
import { EmbeddingUtilsError, ModelNotFoundError } from '../types';
import { MODEL_REGISTRY } from '../models/registry';

/**
 * Looks up tokenizer information for a model in the built-in registry.
 *
 * @param model - Model identifier (e.g., 'Xenova/all-MiniLM-L12-v2')
 * @returns TokenizerInfo with maxTokens and modelId, or undefined if not found
 */
export function getTokenizerInfo(model: string): TokenizerInfo | undefined {
  const info = MODEL_REGISTRY[model];
  if (!info) return undefined;
  return {
    maxTokens: info.maxTokens,
    modelId: info.id,
  };
}

/** Minimal interface for a loaded @huggingface/transformers tokenizer instance. */
interface TokenizerInstance {
  (text: string): { input_ids: { size: number } };
}

/**
 * Creates an exact token counter backed by a model's @huggingface/transformers
 * tokenizer.
 *
 * `count()` mirrors transformers' unpadded single-sequence `input_ids.size`,
 * making it suitable for chunking decisions that must match what the model
 * actually sees. Call {@link LocalTokenizer.load} once before counting; the
 * count methods are synchronous so they can be used inline in hot loops.
 *
 * @param model - HuggingFace model identifier (e.g., 'Xenova/all-MiniLM-L6-v2').
 * @param opts - Optional loading options (local path, cache dir, remote toggle).
 * @returns A {@link LocalTokenizer} for the given model.
 * @throws {ModelNotFoundError} From {@link LocalTokenizer.load} when
 *   @huggingface/transformers is not installed.
 * @throws {EmbeddingUtilsError} From `count`/`countBatch` when called before
 *   {@link LocalTokenizer.load} has completed.
 * @example
 * const tok = createTokenizer('Xenova/all-MiniLM-L6-v2');
 * await tok.load();
 * tok.count('hello world'); // => exact token count
 */
export function createTokenizer(model: string, opts?: CreateTokenizerOptions): LocalTokenizer {
  const maxTokens = MODEL_REGISTRY[model]?.maxTokens ?? 0;

  let instance: TokenizerInstance | null = null;
  let loadPromise: Promise<void> | null = null;

  function loaded(): TokenizerInstance {
    if (!instance) {
      throw new EmbeddingUtilsError('Tokenizer not loaded; call load() first.');
    }
    return instance;
  }

  return {
    maxTokens,
    modelId: model,

    async load(): Promise<void> {
      if (instance) return;
      if (loadPromise) return loadPromise;

      loadPromise = (async () => {
        try {
          const transformers = await import('@huggingface/transformers');
          transformers.env.allowRemoteModels = opts?.allowRemoteModels ?? true;
          if (opts?.modelPath) {
            transformers.env.localModelPath = opts.modelPath;
          }
          if (opts?.cacheDir) {
            transformers.env.cacheDir = opts.cacheDir;
          }
          const tokenizer = await transformers.AutoTokenizer.from_pretrained(model);
          instance = tokenizer as unknown as TokenizerInstance;
        } catch (error: unknown) {
          loadPromise = null;
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

      return loadPromise;
    },

    count(text: string): number {
      return loaded()(text).input_ids.size;
    },

    countBatch(texts: string[]): number[] {
      return texts.map((t) => this.count(t));
    },
  };
}
