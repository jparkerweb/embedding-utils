/**
 * Executes async tasks with bounded concurrency, preserving result order.
 *
 * Maintains a pool of up to `maxConcurrency` active promises. As each resolves,
 * the next queued task starts. On first rejection, no new tasks start and the
 * error propagates to the caller.
 *
 * @internal Not exported from the public API.
 */
export async function promisePool<T>(
  tasks: Array<() => Promise<T>>,
  maxConcurrency: number,
): Promise<T[]> {
  const results = new Array<T>(tasks.length);
  let nextIndex = 0;
  let rejected = false;

  async function runNext(): Promise<void> {
    while (nextIndex < tasks.length && !rejected) {
      const i = nextIndex++;
      try {
        results[i] = await tasks[i]();
      } catch (error) {
        rejected = true;
        throw error;
      }
    }
  }

  const workers = Array.from(
    { length: Math.min(maxConcurrency, tasks.length) },
    () => runNext(),
  );

  await Promise.all(workers);
  return results;
}
