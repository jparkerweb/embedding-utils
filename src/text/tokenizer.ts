import type { TokenizerInfo, CreateTokenizerOptions, LocalTokenizer } from '../types';
import { EmbeddingUtilsError, ModelNotFoundError } from '../types';
import { MODEL_REGISTRY } from '../models/registry';
import { createInstanceRegistry } from '../internal/instance-registry';

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
 * Process-level registry of loaded tokenizer instances, keyed by everything
 * that affects tokenizer loading. `AutoTokenizer.from_pretrained` re-parses
 * the tokenizer files on every call (~350ms for typical sentence-transformer
 * models) — sharing the loaded instance makes repeated `createTokenizer` +
 * `load()` cycles effectively free. Tokenizers are stateless pure-JS objects,
 * so sharing is safe and there is nothing native to dispose.
 */
const MAX_SHARED_TOKENIZERS = 8;
const tokenizerRegistry = createInstanceRegistry<TokenizerInstance>(MAX_SHARED_TOKENIZERS);

function tokenizerKey(model: string, opts?: CreateTokenizerOptions): string {
  return JSON.stringify([
    model,
    opts?.modelPath ?? '',
    opts?.cacheDir ?? '',
    opts?.allowRemoteModels ?? true,
  ]);
}

/**
 * Empties the process-level tokenizer registry. Tokenizers already handed to
 * callers keep working (they are plain JS objects); this only releases the
 * registry's references so subsequent `load()` calls parse fresh instances.
 * Mainly useful in tests.
 */
export function disposeLocalTokenizers(): void {
  // Fire-and-forget: nothing native to dispose, just drop the references.
  void tokenizerRegistry.clear();
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

  async function buildTokenizer(): Promise<TokenizerInstance> {
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
      return tokenizer as unknown as TokenizerInstance;
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

  // Shared path (default): one loaded tokenizer per unique config for the
  // whole process. Opt-out with `reuse: false` for a private instance.
  const reuse = opts?.reuse ?? true;

  return {
    maxTokens,
    modelId: model,

    async load(): Promise<void> {
      if (instance) return;
      if (loadPromise) return loadPromise;

      loadPromise = (async () => {
        try {
          instance = reuse
            ? await tokenizerRegistry.getOrCreate(tokenizerKey(model, opts), buildTokenizer)
            : await buildTokenizer();
        } catch (error: unknown) {
          loadPromise = null; // allow retry after failure
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
