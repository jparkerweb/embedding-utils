import type {
  EmbeddingProvider,
  ProviderType,
  ProviderConfigMap,
  OpenAICompatibleConfig,
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
  together: 'https://api.together.xyz/v1',
  fireworks: 'https://api.fireworks.ai/inference/v1',
  nomic: 'https://api-atlas.nomic.ai/v1',
  mixedbread: 'https://api.mixedbread.ai/v1',
};

/**
 * Factory that creates an embedding provider by type name.
 * @param type - Provider type: 'local', 'openai', 'cohere', 'google-vertex', 'voyage', 'mistral', 'jina', 'openrouter'
 * @param config - Provider-specific configuration object (type-checked per provider)
 * @returns An EmbeddingProvider for the specified type
 * @throws {ValidationError} If the provider type is unknown
 * @example
 * const provider = createProvider('openai', { apiKey: 'sk-...', model: 'text-embedding-3-small' });
 */
export function createProvider<T extends ProviderType>(
  type: T,
  config: ProviderConfigMap[T],
): EmbeddingProvider {
  switch (type) {
    case 'local':
      return createLocalProvider(config as ProviderConfigMap['local']);

    case 'openai':
      return createOpenAICompatibleProvider(config as ProviderConfigMap['openai']);

    case 'cohere':
      return createCohereProvider(config as ProviderConfigMap['cohere']);

    case 'google-vertex':
      return createGoogleVertexProvider(config as ProviderConfigMap['google-vertex']);

    case 'voyage':
    case 'mistral':
    case 'jina':
    case 'openrouter':
    case 'together':
    case 'fireworks':
    case 'nomic':
    case 'mixedbread':
      return createOpenAICompatibleProvider({
        ...(config as OpenAICompatibleConfig),
        baseUrl: ALIAS_BASE_URLS[type],
      });

    default:
      throw new ValidationError(`Unknown provider type: ${type}`);
  }
}
