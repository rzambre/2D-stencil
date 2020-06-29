/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code parallelized by multiple threads with MPI_THREAD_FUNNELED.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations with derived data types to exchange
 * grid points in a halo region with neighbors. Computation is parallelized by multiple threads.
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))
#define COMPUTE 0
#define PACK 1
#define FINE_TIME 1

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag);

void alloc_bufs(size_t bx, size_t by,
                double **aold_ptr, double **anew_ptr,
                double **sbufnorth_ptr, double **sbufsouth_ptr,
                double **sbufeast_ptr, double **sbufwest_ptr,
                double **rbufnorth_ptr, double **rbufsouth_ptr,
                double **rbufeast_ptr, double **rbufwest_ptr);

#if PACK
static inline void pack_data(size_t bx, size_t by, double *aold,
               double *sbufnorth, double *sbufsouth, double *sbfueast, double *sbufwest);

static inline void unpack_data(size_t bx, size_t by, double *aold,
                 double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest);
#endif

int main(int argc, char **argv)
{
    int rank, size, provided;
    int n, energy, niters, px, py;

    int rx, ry;
    int north, south, west, east;
    size_t bx, by;

#if FINE_TIME
    double t_pack_start, t_comm_start;
    double t_pack, t_comm, t_halox;
    double *t_comm_procs, *t_pack_procs, *t_halox_procs;
#else
    double t1, t2;
#endif

    int iter;
    int nthreads;

    double *aold, *anew;
    double *north_in, *south_in, *east_in, *west_in;
    double *north_out, *south_out, *east_out, *west_out;

#if COMPUTE
    int i, j;
#endif
    double *tmp;

    int final_flag;

    /* initialize MPI envrionment */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    nthreads = omp_get_max_threads();

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    north = (ry - 1) * px + rx;
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    south = (ry + 1) * px + rx;
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    west = ry * px + rx - 1;
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    east = ry * px + rx + 1;
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */

    /* allocate and initialize working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew,
               &north_out, &south_out, &east_out, &west_out,
               &north_in, &south_in, &east_in, &west_in);

#if COMPUTE
#pragma omp parallel private(iter,i,j)
#else
#pragma omp parallel private(iter)
#endif
    {
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#pragma omp barrier
#if FINE_TIME
#pragma omp master
        {
            t_pack = t_comm = 0;
        }
#else
#pragma omp master
        {
            t1 = MPI_Wtime(); /* take time */
        }
#endif
        for (iter = 0; iter < niters; ++iter) {
#pragma omp master
            {
                /* exchange data with neighbors */
                MPI_Request reqs[8];
#if PACK
#if FINE_TIME
                t_pack_start = MPI_Wtime();
#endif
                /* TODO: Pack with multiple threads */
                pack_data(bx, by, aold, north_out, south_out, east_out, west_out);
#if FINE_TIME
                t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif
                /* prevent cost of load imbalance in communication time */
                MPI_Barrier(MPI_COMM_WORLD);
#if FINE_TIME
                t_comm_start = MPI_Wtime();
#endif
                MPI_Irecv(south_in, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[0]);
                MPI_Irecv(north_in, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[1]);
                MPI_Irecv(west_in, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[2]);
                MPI_Irecv(east_in, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[3]);
                MPI_Isend(south_out, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[4]);
                MPI_Isend(north_out, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[5]);
                MPI_Isend(west_out, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[6]);
                MPI_Isend(east_out, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[7]);
 
                MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
#if FINE_TIME
                t_comm += (MPI_Wtime() - t_comm_start);
#endif
#if PACK
#if FINE_TIME
                t_pack_start = MPI_Wtime();
#endif
                /* TODO: Unpack with multiple threads */
                unpack_data(bx, by, aold, north_in, south_in, east_in, west_in);
#if FINE_TIME
                t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif
            }
#pragma omp barrier

#if COMPUTE
            /* update grid points */
#pragma omp for schedule(static)
            for (i = 1; i < bx + 1; ++i) {
                for (j = 1; j < by + 1; ++j) {
                    anew[ind(i, j)] =
                        anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                                 aold[ind(i, j - 1)] +
                                                 aold[ind(i, j + 1)]) / 4.0 / 2.0;
                }
            }
#endif
#pragma omp master
            {
                /* swap working arrays */
                tmp = anew;
                anew = aold;
                aold = tmp;
            }
            /* Barrier not needed here since master is doing the packing/unpacking */
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
#if FINE_TIME
    t_halox = t_pack + t_comm;

    if (rank == 0) {
        t_pack_procs = calloc(size, sizeof(double));
        if (!t_pack_procs) {
            fprintf(stderr, "Unable to allocate memory for t_pack_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_comm_procs = calloc(size, sizeof(double));
        if (!t_comm_procs) {
            fprintf(stderr, "Unable to allocate memory for t_comm_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_halox_procs = calloc(size, sizeof(double));
        if (!t_halox_procs) {
            fprintf(stderr, "Unable to allocate memory for t_halox_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else
        t_pack_procs = t_comm_procs = t_halox_procs = NULL;
    MPI_Gather(&t_pack, 1, MPI_DOUBLE,
            t_pack_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comm, 1, MPI_DOUBLE,
            t_comm_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_halox, 1, MPI_DOUBLE,
            t_halox_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    t2 = MPI_Wtime();
#endif

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);

    if (rank == 0) {
#if FINE_TIME
        int pi;
        double min_t_comm, max_t_comm, mean_t_comm, sum_t_comm;
        double min_t_pack, max_t_pack, mean_t_pack, sum_t_pack;
        double min_t_halox, max_t_halox, mean_t_halox, sum_t_halox;
 
        min_t_comm = min_t_pack = min_t_halox = 9999;
        max_t_comm = max_t_pack = max_t_halox = -1;
        sum_t_comm = sum_t_pack = sum_t_halox = 0;

        for (pi = 0; pi < size; pi++) {
            if (max_t_comm < t_comm_procs[pi]) {
                max_t_comm = t_comm_procs[pi];
            }
            if (min_t_comm > t_comm_procs[pi]) {
                min_t_comm = t_comm_procs[pi];
            }
            sum_t_comm += t_comm_procs[pi];

            if (max_t_pack < t_pack_procs[pi]) {
                max_t_pack = t_pack_procs[pi];
            }
            if (min_t_pack > t_pack_procs[pi]) {
                min_t_pack = t_pack_procs[pi];
            }
            sum_t_pack += t_pack_procs[pi];
            
            if (max_t_halox < t_halox_procs[pi]) {
                max_t_halox = t_halox_procs[pi];
            }
            if (min_t_halox > t_halox_procs[pi]) {
                min_t_halox = t_halox_procs[pi];
            }
            sum_t_halox += t_halox_procs[pi];
        }
        
        /* Divide by size since all ranks participate in communication */
        mean_t_pack = sum_t_pack / size;
        mean_t_comm = sum_t_comm / size;
        mean_t_halox = sum_t_halox / size;

        //printf("Rank %d had the max comm time and rank %d has the max pack time\n", max_comm_rank, max_pack_rank);
        printf("mesh_dim,px,py,iters,threads,min_comm_time,max_comm_time,mean_comm_time,min_pack_time,max_pack_time,mean_pack_time,min_halox_time,max_halox_time,mean_halox_time\n");
        printf("%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", n, px, py, niters, nthreads, min_t_comm, max_t_comm, mean_t_comm, min_t_pack, max_t_pack, mean_t_pack, min_t_halox, max_t_halox, mean_t_halox);
#else
        printf("mesh_dim,px,py,iters,threads,time\n");
        printf("%d,%d,%d,%d,%d,%f\n", n, px, py, niters, nthreads, t2 - t1);
#endif
    }

#if FINE_TIME
    free(t_pack_procs);
    free(t_comm_procs);
    free(t_halox_procs);
#endif
    
    MPI_Finalize();
    return 0;
}

#if PACK
static inline void pack_data(size_t bx, size_t by, double *aold,
               double *sbufnorth, double *sbufsouth, double *sbufeast, double *sbufwest)
{
    size_t i;
    for (i = 0; i < bx; ++i)
        sbufnorth[i] = aold[ind(i + 1, 1)];     /* #1 row */
    for (i = 0; i < bx; ++i)
        sbufsouth[i] = aold[ind(i + 1, by)];    /* #(by) row */
    for (i = 0; i < by; ++i)
        sbufeast[i] = aold[ind(bx, i + 1)];     /* #(bx) col */
    for (i = 0; i < by; ++i)
        sbufwest[i] = aold[ind(1, i + 1)];      /* #1 col */
}

static inline void unpack_data(size_t bx, size_t by, double *aold,
                 double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest)
{
    size_t i;
    for (i = 0; i < bx; ++i)
        aold[ind(i + 1, 0)] = rbufnorth[i];     /* #0 row */
    for (i = 0; i < bx; ++i)
        aold[ind(i + 1, by + 1)] = rbufsouth[i];        /* #(by+1) row */
    for (i = 0; i < by; ++i)
        aold[ind(bx + 1, i + 1)] = rbufeast[i]; /* #(bx+1) col */
    for (i = 0; i < by; ++i)
        aold[ind(0, i + 1)] = rbufwest[i];      /* #0 col */
}
#endif

void alloc_bufs(size_t bx, size_t by, double **aold_ptr, double **anew_ptr,
                double **sbufnorth_ptr, double **sbufsouth_ptr,
                double **sbufeast_ptr, double **sbufwest_ptr,
                double **rbufnorth_ptr, double **rbufsouth_ptr,
                double **rbufeast_ptr, double **rbufwest_ptr)
{
    double *aold, *anew;
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;

    /* allocate two working arrays */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* allocate communication buffers */
    sbufnorth = (double *) malloc(bx * sizeof(double)); /* send buffers */
    sbufsouth = (double *) malloc(bx * sizeof(double));
    sbufeast = (double *) malloc(by * sizeof(double));
    sbufwest = (double *) malloc(by * sizeof(double));
    rbufnorth = (double *) malloc(bx * sizeof(double)); /* receive buffers */
    rbufsouth = (double *) malloc(bx * sizeof(double));
    rbufeast = (double *) malloc(by * sizeof(double));
    rbufwest = (double *) malloc(by * sizeof(double));

    memset(sbufnorth, 0, bx * sizeof(double));
    memset(sbufsouth, 0, bx * sizeof(double));
    memset(sbufeast, 0, by * sizeof(double));
    memset(sbufwest, 0, by * sizeof(double));
    memset(rbufnorth, 0, bx * sizeof(double));
    memset(rbufsouth, 0, bx * sizeof(double));
    memset(rbufeast, 0, by * sizeof(double));
    memset(rbufwest, 0, by * sizeof(double));

    (*aold_ptr) = aold;
    (*anew_ptr) = anew;
    (*sbufnorth_ptr) = sbufnorth;
    (*sbufsouth_ptr) = sbufsouth;
    (*sbufeast_ptr) = sbufeast;
    (*sbufwest_ptr) = sbufwest;
    (*rbufnorth_ptr) = rbufnorth;
    (*rbufsouth_ptr) = rbufsouth;
    (*rbufeast_ptr) = rbufeast;
    (*rbufwest_ptr) = rbufwest;
}

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag)
{
    int n, energy, niters, px, py;

    (*final_flag) = 0;

    if (argc < 6) {
        if (!rank)
            printf("usage: stencil_mpi <n> <energy> <niters> <px> <py>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */
    px = atoi(argv[4]); /* 1st dim processes */
    py = atoi(argv[5]); /* 2nd dim processes */

    if (px * py != proc)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
    if (n % py != 0)
        MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
    if (n % px != 0)
        MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */

    (*n_ptr) = n;
    (*energy_ptr) = energy;
    (*niters_ptr) = niters;
    (*px_ptr) = px;
    (*py_ptr) = py;
}
