#include "ds4_expert_io.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

struct ds4_io_pool {
    pthread_t *threads;
    int n_threads;
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    ds4_io_task *tasks;
    int num_tasks;
    int next_task;
    int completed;
    volatile bool shutdown;
};

static void *io_worker(void *arg) {
    ds4_io_pool *pool = (ds4_io_pool *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->next_task >= pool->num_tasks && !pool->shutdown) {
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        int idx = pool->next_task++;
        ds4_io_task task = pool->tasks[idx];
        pthread_mutex_unlock(&pool->mutex);

        ssize_t result = pread(task.fd, task.dst, task.size, task.offset);

        pthread_mutex_lock(&pool->mutex);
        pool->tasks[idx].result = result;
        pool->completed++;
        if (pool->completed == pool->num_tasks) {
            pthread_cond_signal(&pool->work_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
}

ds4_io_pool *ds4_io_pool_create(int n_threads) {
    ds4_io_pool *pool = calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    pool->n_threads = n_threads;
    pool->threads = calloc((size_t)n_threads, sizeof(pthread_t));
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);
    for (int i = 0; i < n_threads; i++) {
        pthread_create(&pool->threads[i], NULL, io_worker, pool);
    }
    return pool;
}

void ds4_io_pool_dispatch_async(ds4_io_pool *pool, ds4_io_task *tasks, int n) {
    if (!pool || !tasks || n <= 0) return;
    pthread_mutex_lock(&pool->mutex);
    pool->tasks = tasks;
    pool->num_tasks = n;
    pool->next_task = 0;
    pool->completed = 0;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
}

void ds4_io_pool_wait(ds4_io_pool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    while (pool->completed < pool->num_tasks) {
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

void ds4_io_pool_dispatch(ds4_io_pool *pool, ds4_io_task *tasks, int n) {
    ds4_io_pool_dispatch_async(pool, tasks, n);
    ds4_io_pool_wait(pool);
}

void ds4_io_pool_destroy(ds4_io_pool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_ready);
    pthread_cond_destroy(&pool->work_done);
    free(pool->threads);
    free(pool);
}
