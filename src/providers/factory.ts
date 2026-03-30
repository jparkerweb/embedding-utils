import type {
  EmbeddingProvider,
  ProviderType,
  OpenAICompatibleConfig,
  CohereConfig,
  GoogleVertexConfig,
  LocalProviderConfig,
} from '../types';
import { ValidationError } from '../types';
import { createOpenAICompatibleProvider } from './openai-compatible';
import { createCohereProvider } from './cohere';
import { createGoogleVertexProvider } from './google';
import { createLocalProvider } from './local';

const ALIAS_BASE_URLS: Record<string, string> = {
  voyage: 'https://api.voyageai.com/v1',
  mistral: 'https://api.mistral.ai/v1',
  jina: 'https://api.jina.ai/v1',
  openrouter: 'https://openrouter.ai/api/v1',
};

/**
 * Factory that creates an embedding provider by type name.
 * @param type - Provider type: 'local', 'openai', 'cohere', 'google-vertex', 'voyage', 'mistral', 'jina', 'openrouter'
 * @param config - Provider-specific configuration object
 * @returns An EmbeddingProvider for the specified type
 * @throws {ValidationError} If the provider type is unknown
 * @example
 * const provider = createProvider('openai', { apiKey: 'sk-...', model: 'text-embedding-3-small' });
 */
export function createProvider(type: ProviderType, config: any): EmbeddingProvider {
  switch (type) {
    case 'local':
      return createLocalProvider(config as LocalProviderConfig);

    case 'openai':
      return createOpenAICompatibleProvider(config as OpenAICompatibleConfig);

    case 'cohere':
      return createCohereProvider(config as CohereConfig);

    case 'google-vertex':
      return createGoogleVertexProvider(config as GoogleVertexConfig);

    case 'voyage':
    case 'mistral':
    case 'jina':
    case 'openrouter':
      return createOpenAICompatibleProvider({
        ...config,
        baseUrl: ALIAS_BASE_URLS[type],
      } as OpenAICompatibleConfig);

    default:
      throw new ValidationError(`Unknown provider type: ${type}`);
  }
}
