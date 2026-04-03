import { describe, it, expect, beforeEach } from 'vitest';
import { MODEL_REGISTRY, registerModel, getRecommendedModel } from '../../src/models/registry';
import { ValidationError } from '../../src/types';

describe('registerModel', () => {
  const customModel = {
    id: 'test-org/test-model',
    dimensions: 512,
    maxTokens: 1024,
    description: 'Test model for unit tests',
  };

  beforeEach(() => {
    // Clean up any test models added in previous tests
    delete MODEL_REGISTRY['test-org/test-model'];
    delete MODEL_REGISTRY['test-org/another-model'];
  });

  it('adds a new model to the registry', () => {
    registerModel(customModel);
    expect(MODEL_REGISTRY['test-org/test-model']).toEqual(customModel);
  });

  it('rejects duplicate model ID without overwrite', () => {
    registerModel(customModel);
    expect(() => registerModel(customModel)).toThrow(ValidationError);
    expect(() => registerModel(customModel)).toThrow('already registered');
  });

  it('allows overwrite with option', () => {
    registerModel(customModel);
    const updated = { ...customModel, description: 'Updated' };
    registerModel(updated, { overwrite: true });
    expect(MODEL_REGISTRY['test-org/test-model'].description).toBe('Updated');
  });

  it('rejects empty model ID', () => {
    expect(() =>
      registerModel({ id: '', dimensions: 128, maxTokens: 256, description: 'bad' }),
    ).toThrow(ValidationError);
  });

  it('rejects non-positive dimensions', () => {
    expect(() =>
      registerModel({ id: 'test-org/another-model', dimensions: 0, maxTokens: 256, description: 'bad' }),
    ).toThrow(ValidationError);
    expect(() =>
      registerModel({ id: 'test-org/another-model', dimensions: -1, maxTokens: 256, description: 'bad' }),
    ).toThrow(ValidationError);
  });

  it('rejects non-integer dimensions', () => {
    expect(() =>
      registerModel({ id: 'test-org/another-model', dimensions: 3.5, maxTokens: 256, description: 'bad' }),
    ).toThrow(ValidationError);
  });
});

describe('getRecommendedModel', () => {
  it('returns MiniLM-L6-v2 for speed', () => {
    const model = getRecommendedModel('speed');
    expect(model.id).toBe('Xenova/all-MiniLM-L6-v2');
  });

  it('returns all-mpnet-base-v2 for balanced', () => {
    const model = getRecommendedModel('balanced');
    expect(model.id).toBe('Xenova/all-mpnet-base-v2');
  });

  it('returns all-mpnet-base-v2 for quality', () => {
    const model = getRecommendedModel('quality');
    expect(model.id).toBe('Xenova/all-mpnet-base-v2');
  });

  it('returns multilingual-MiniLM-L12-v2 for multilingual', () => {
    const model = getRecommendedModel('multilingual');
    expect(model.id).toBe('Xenova/multilingual-MiniLM-L12-v2');
  });
});

describe('MODEL_REGISTRY completeness', () => {
  it('all models have maxTokens defined', () => {
    for (const [id, model] of Object.entries(MODEL_REGISTRY)) {
      expect(model.maxTokens, `${id} is missing maxTokens`).toBeGreaterThan(0);
    }
  });

  it('contains the 3 new models', () => {
    expect(MODEL_REGISTRY['Xenova/multilingual-MiniLM-L12-v2']).toBeDefined();
    expect(MODEL_REGISTRY['nomic-ai/nomic-embed-text-v1.5']).toBeDefined();
    expect(MODEL_REGISTRY['Xenova/distilroberta-base']).toBeDefined();
  });

  it('new models have correct dimensions', () => {
    expect(MODEL_REGISTRY['Xenova/multilingual-MiniLM-L12-v2'].dimensions).toBe(384);
    expect(MODEL_REGISTRY['nomic-ai/nomic-embed-text-v1.5'].dimensions).toBe(768);
    expect(MODEL_REGISTRY['Xenova/distilroberta-base'].dimensions).toBe(768);
  });
});
