import type { TokenizerInfo } from '../types';
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
