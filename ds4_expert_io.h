#ifndef DS4_EXPERT_IO_H
#define DS4_EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

#define DS4_IO_MAX_TASKS 18

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
} ds4_io_task;

typedef struct ds4_io_pool ds4_io_pool;

ds4_io_pool *ds4_io_pool_create(int n_threads);
void ds4_io_pool_dispatch(ds4_io_pool *pool, ds4_io_task *tasks, int n);
void ds4_io_pool_dispatch_async(ds4_io_pool *pool, ds4_io_task *tasks, int n);
void ds4_io_pool_wait(ds4_io_pool *pool);
void ds4_io_pool_destroy(ds4_io_pool *pool);

#endif
