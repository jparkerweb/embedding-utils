import { describe, it, expect } from 'vitest';
import { serialize, deserialize } from '../../src/storage/serialize';
import { ValidationError } from '../../src/types';

const singleEmbedding = [[0.1, 0.2, 0.3, 0.4, 0.5]];
const batchEmbeddings = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
];

describe('serialize / deserialize — JSON format', () => {
  it('roundtrips a single embedding', () => {
    const data = serialize(singleEmbedding, 'json');
    const result = deserialize(data, 'json');
    expect(result.embeddings).toEqual(singleEmbedding);
  });

  it('roundtrips batch embeddings', () => {
    const data = serialize(batchEmbeddings, 'json');
    const result = deserialize(data, 'json');
    expect(result.embeddings).toEqual(batchEmbeddings);
  });

  it('preserves numeric precision', () => {
    const precise = [[1.234567890123456]];
    const data = serialize(precise, 'json');
    const result = deserialize(data, 'json');
    expect(result.embeddings[0][0]).toBe(1.234567890123456);
  });

  it('output is a valid JSON string', () => {
    const data = serialize(singleEmbedding, 'json');
    expect(typeof data).toBe('string');
    expect(() => JSON.parse(data as string)).not.toThrow();
  });

  it('throws ValidationError for empty embedding array', () => {
    expect(() => serialize([], 'json')).toThrow(ValidationError);
  });
});

describe('serialize / deserialize — binary format (v2)', () => {
  it('roundtrips a single embedding', () => {
    const data = serialize(singleEmbedding, 'binary');
    const result = deserialize(data, 'binary');
    expect(result.embeddings.length).toBe(1);
    for (let i = 0; i < singleEmbedding[0].length; i++) {
      expect(result.embeddings[0][i]).toBeCloseTo(singleEmbedding[0][i], 6);
    }
  });

  it('roundtrips batch embeddings', () => {
    const data = serialize(batchEmbeddings, 'binary');
    const result = deserialize(data, 'binary');
    expect(result.embeddings.length).toBe(batchEmbeddings.length);
    for (let i = 0; i < batchEmbeddings.length; i++) {
      for (let j = 0; j < batchEmbeddings[i].length; j++) {
        expect(result.embeddings[i][j]).toBeCloseTo(batchEmbeddings[i][j], 6);
      }
    }
  });

  it('preserves Float32 precision', () => {
    const val = Math.fround(0.123456789);
    const data = serialize([[val]], 'binary');
    const result = deserialize(data, 'binary');
    expect(result.embeddings[0][0]).toBe(val);
  });

  it('output is a Uint8Array', () => {
    const data = serialize(singleEmbedding, 'binary');
    expect(data).toBeInstanceOf(Uint8Array);
  });

  it('throws ValidationError for empty embedding array', () => {
    expect(() => serialize([], 'binary')).toThrow(ValidationError);
  });
});

describe('serialize / deserialize — base64 format', () => {
  it('roundtrips a single embedding', () => {
    const data = serialize(singleEmbedding, 'base64');
    const result = deserialize(data, 'base64');
    expect(result.embeddings.length).toBe(1);
    for (let i = 0; i < singleEmbedding[0].length; i++) {
      expect(result.embeddings[0][i]).toBeCloseTo(singleEmbedding[0][i], 6);
    }
  });

  it('roundtrips batch embeddings', () => {
    const data = serialize(batchEmbeddings, 'base64');
    const result = deserialize(data, 'base64');
    expect(result.embeddings.length).toBe(batchEmbeddings.length);
    for (let i = 0; i < batchEmbeddings.length; i++) {
      for (let j = 0; j < batchEmbeddings[i].length; j++) {
        expect(result.embeddings[i][j]).toBeCloseTo(batchEmbeddings[i][j], 6);
      }
    }
  });

  it('output is a valid base64 string', () => {
    const data = serialize(singleEmbedding, 'base64');
    expect(typeof data).toBe('string');
    const base64Regex = /^[A-Za-z0-9+/]+=*$/;
    expect(base64Regex.test(data as string)).toBe(true);
  });

  it('throws ValidationError for empty embedding array', () => {
    expect(() => serialize([], 'base64')).toThrow(ValidationError);
  });
});
