import type { ModelInfo } from '../types';
import { MODEL_REGISTRY } from './registry';

let defaultCacheDir = '';

function getDefaultCacheDir(): string {
  if (defaultCacheDir) return defaultCacheDir;
  const home = process.env.HOME || process.env.USERPROFILE || '';
  return `${home}/.cache/huggingface/hub`;
}

function modelIdToDirName(modelId: string): string {
  return modelId.replace('/', '--');
}

function dirNameToModelId(dirName: string): string {
  return dirName.replace('--', '/');
}

/**
 * Sets the default cache directory for downloaded models.
 * @param path - Absolute path to the cache directory
 * @example
 * setModelPath('/home/user/.models');
 */
export function setModelPath(path: string): void {
  defaultCacheDir = path;
}

/**
 * Looks up model metadata from the built-in registry.
 * @param modelId - The model identifier (e.g. 'Xenova/all-MiniLM-L12-v2')
 * @returns ModelInfo if found, undefined otherwise
 * @example
 * getModelInfo('Xenova/all-MiniLM-L12-v2');
 * // { id: '...', dimensions: 384, description: '...' }
 */
export function getModelInfo(modelId: string): ModelInfo | undefined {
  return MODEL_REGISTRY[modelId];
}

/**
 * Downloads a HuggingFace model to the local cache.
 * @param modelId - The model identifier (e.g. 'Xenova/all-MiniLM-L6-v2')
 * @param options - Optional cache directory override
 * @returns Path to the downloaded model directory
 * @example
 * const path = await downloadModel('Xenova/all-MiniLM-L6-v2');
 */
export async function downloadModel(
  modelId: string,
  options?: { cacheDir?: string },
): Promise<string> {
  const transformers = await import('@huggingface/transformers');
  const cacheDir = options?.cacheDir ?? getDefaultCacheDir();

  await transformers.pipeline('feature-extraction', modelId, {
    cache_dir: cacheDir,
  });

  return `${cacheDir}/${modelIdToDirName(modelId)}`;
}

/**
 * Lists all locally cached models.
 * @param cacheDir - Optional cache directory path (uses default if not provided)
 * @returns Array of ModelInfo for each cached model
 * @example
 * const models = await listModels();
 */
export async function listModels(cacheDir?: string): Promise<ModelInfo[]> {
  const fs = await import('node:fs/promises');
  const dir = cacheDir ?? getDefaultCacheDir();

  try {
    await fs.access(dir);
  } catch {
    return [];
  }

  const entries = await fs.readdir(dir, { withFileTypes: true });
  const models: ModelInfo[] = [];

  for (const entry of entries) {
    if (entry.isDirectory()) {
      const modelId = dirNameToModelId(entry.name);
      const registryInfo = MODEL_REGISTRY[modelId];
      if (registryInfo) {
        models.push(registryInfo);
      } else {
        models.push({
          id: modelId,
          dimensions: 0,
          maxTokens: 0,
          description: 'Unknown model',
        });
      }
    }
  }

  return models;
}

/**
 * Deletes a model from the local cache.
 * @param modelId - The model identifier to delete
 * @param cacheDir - Optional cache directory path (uses default if not provided)
 * @example
 * await deleteModel('Xenova/all-MiniLM-L6-v2');
 */
export async function deleteModel(
  modelId: string,
  cacheDir?: string,
): Promise<void> {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');
  const dir = cacheDir ?? getDefaultCacheDir();
  const modelDir = path.join(dir, modelIdToDirName(modelId));

  await fs.rm(modelDir, { recursive: true, force: true });
}
