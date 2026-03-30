import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  downloadModel,
  listModels,
  deleteModel,
  setModelPath,
  getModelInfo,
} from '../../src/models/manager';
import { MODEL_REGISTRY } from '../../src/models/registry';

// Mock @huggingface/transformers
const mockPipeline = vi.fn();
vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipeline,
}));

// Mock fs/promises
const mockReaddir = vi.fn();
const mockRm = vi.fn();
const mockStat = vi.fn();
const mockAccess = vi.fn();
vi.mock('node:fs/promises', () => ({
  readdir: (...args: any[]) => mockReaddir(...args),
  rm: (...args: any[]) => mockRm(...args),
  stat: (...args: any[]) => mockStat(...args),
  access: (...args: any[]) => mockAccess(...args),
}));

// Mock node:path
vi.mock('node:path', async () => {
  const actual = await vi.importActual('node:path');
  return actual;
});

describe('model manager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPipeline.mockResolvedValue(() => {});
  });

  describe('downloadModel', () => {
    it('should call pipeline to trigger model download', async () => {
      mockPipeline.mockResolvedValue(() => {});

      const result = await downloadModel('Xenova/all-MiniLM-L12-v2');

      expect(mockPipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'Xenova/all-MiniLM-L12-v2',
        expect.any(Object),
      );
      expect(typeof result).toBe('string');
    });

    it('should pass cacheDir option', async () => {
      await downloadModel('Xenova/all-MiniLM-L6-v2', {
        cacheDir: '/custom/cache',
      });

      expect(mockPipeline).toHaveBeenCalledWith(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        expect.objectContaining({ cache_dir: '/custom/cache' }),
      );
    });
  });

  describe('listModels', () => {
    it('should return array of model info from cache directory', async () => {
      mockAccess.mockResolvedValue(undefined);
      mockReaddir.mockResolvedValue([
        { name: 'Xenova--all-MiniLM-L12-v2', isDirectory: () => true },
        { name: 'Xenova--bge-small-en-v1.5', isDirectory: () => true },
      ]);
      mockStat.mockResolvedValue({ mtimeMs: Date.now() });

      const models = await listModels('/cache/dir');

      expect(models.length).toBe(2);
      expect(models[0]).toHaveProperty('id');
    });

    it('should return empty array if cache dir does not exist', async () => {
      mockAccess.mockRejectedValue(new Error('ENOENT'));

      const models = await listModels('/nonexistent');

      expect(models).toEqual([]);
    });
  });

  describe('deleteModel', () => {
    it('should remove model directory from cache', async () => {
      mockRm.mockResolvedValue(undefined);

      await deleteModel('Xenova/all-MiniLM-L12-v2', '/cache/dir');

      expect(mockRm).toHaveBeenCalledWith(
        expect.stringContaining('Xenova--all-MiniLM-L12-v2'),
        { recursive: true, force: true },
      );
    });
  });

  describe('setModelPath', () => {
    it('should override default cache directory for subsequent calls', async () => {
      setModelPath('/my/custom/path');
      mockAccess.mockResolvedValue(undefined);
      mockReaddir.mockResolvedValue([]);

      await listModels();

      expect(mockReaddir).toHaveBeenCalledWith(
        '/my/custom/path',
        expect.any(Object),
      );

      // Reset for other tests
      setModelPath('');
    });
  });

  describe('getModelInfo', () => {
    it('should return metadata for a known model from registry', () => {
      const info = getModelInfo('Xenova/all-MiniLM-L12-v2');

      expect(info).toBeDefined();
      expect(info!.id).toBe('Xenova/all-MiniLM-L12-v2');
      expect(info!.dimensions).toBe(384);
      expect(info!.description).toBeTruthy();
    });

    it('should return undefined for unknown model', () => {
      const info = getModelInfo('unknown/model');

      expect(info).toBeUndefined();
    });

    it('should return info matching MODEL_REGISTRY', () => {
      const info = getModelInfo('Xenova/bge-base-en-v1.5');

      expect(info).toEqual(MODEL_REGISTRY['Xenova/bge-base-en-v1.5']);
    });
  });
});
