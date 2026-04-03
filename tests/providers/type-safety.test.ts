import { describe, it, expect } from 'vitest';
import { createProvider } from '../../src/providers/factory';

describe('createProvider type safety', () => {
  it('accepts correct config for openai provider', () => {
    const provider = createProvider('openai', {
      apiKey: 'sk-test',
      model: 'text-embedding-3-small',
    });
    expect(provider.name).toBe('openai-compatible');
  });

  it('accepts correct config for cohere provider', () => {
    const provider = createProvider('cohere', {
      apiKey: 'co-test',
    });
    expect(provider.name).toBe('cohere');
  });

  it('accepts correct config for google-vertex provider', () => {
    const provider = createProvider('google-vertex', {
      projectId: 'my-project',
      accessToken: 'ya29-test',
    });
    expect(provider.name).toBe('google-vertex');
  });

  it('accepts correct config for local provider', () => {
    const provider = createProvider('local', {});
    expect(provider.name).toBe('local');
  });

  // These tests verify compile-time type checking using @ts-expect-error.
  // If the type system is working correctly, these lines would produce
  // TypeScript errors without the @ts-expect-error directive.
  it('rejects wrong config types at compile time', () => {
    // @ts-expect-error - openai requires apiKey and model
    createProvider('openai', { projectId: 'wrong' });

    // @ts-expect-error - cohere does not accept projectId
    createProvider('cohere', { projectId: 'wrong' });

    // @ts-expect-error - google-vertex requires projectId and accessToken
    createProvider('google-vertex', { apiKey: 'wrong', model: 'wrong' });
  });
});
