/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code parallelized by multiple threads with MPI_THREAD_MULTIPLE using multiple
 * communicators.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * The computation over x-axis is multithreaded. In every iteration, each thread calls nonblocking
 * operations with derived data types to exchange grid points in a halo region with corresponding
 * threads of neighbor nodes. Threads use different communicators to avoid contentions.
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PAGE_SIZE 4096

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))
#define COMPUTE 0
#define PACK 1
#define FINE_TIME 1 

#define THX_START (Th_dim_x == 0 ? 1 : Th_dim_x * Thx + 1)
#define THX_END (Th_dim_x == Th_dim - 1 ? bx : (Th_dim_x + 1) * Thx)
#define THY_START (Th_dim_y == 0 ? 1 : Th_dim_y * Thy + 1)
#define THY_END (Th_dim_y == Th_dim - 1 ? by : (Th_dim_y + 1) * Thy)

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag);

int main(int argc, char **argv)
{
    int rank, size, provided;
    int n, energy, niters, px, py;
    MPI_Comm *north_south_comms_a, *north_south_comms_b;
    MPI_Comm *east_west_comms_a, *east_west_comms_b;

    int rx, ry;
    int north, south, west, east;
    size_t bx, by;

#if FINE_TIME
    double *t_pack, *t_comm, *t_halox;
    double *t_comm_workers, *t_pack_workers, *t_halox_workers;
#else
    double t1, t2;
#endif

    int loop_i;

    double *aold, *anew;
    double *tmp;

    int final_flag;

    int nthreads, Th_dim, Thx, Thy;
    double Th_dim_dbl;

    nthreads = omp_get_max_threads();
    Th_dim_dbl = sqrt(nthreads); 
    Th_dim = Th_dim_dbl;
    if (Th_dim != Th_dim_dbl) {
        fprintf(stderr, "Supporting only number of threads that are perfect squares for regular partitioning\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* initialize MPI envrionment */
    if (nthreads > 1) {
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE)
            MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
        if (provided < MPI_THREAD_SINGLE)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /*char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    FILE *fp;
    char path[1000];
    MPI_Get_processor_name(processor_name, &name_len);
    fp = popen("grep Cpus_allowed_list /proc/$$/status", "r");
    while (fgets(path, 1000, fp) != NULL) {
        printf("%s[%d]: %s", processor_name, rank, path);
    }*/
    
    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

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

    /* divide blocks in x and y amongst threads */
    Thx = bx / Th_dim;
    Thy = by / Th_dim; 

    if (Thx == 0 || Thy == 0) {
        if (rank == 0) {
            fprintf(stderr, "Domain size too small for number of threads\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* duplicate as many comm_world communicators as number of threads in one dimension for north-south communication */
    north_south_comms_a = (MPI_Comm *) malloc(sizeof(MPI_Comm) * Th_dim);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &north_south_comms_a[loop_i]);
    
    north_south_comms_b = (MPI_Comm *) malloc(sizeof(MPI_Comm) * Th_dim);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &north_south_comms_b[loop_i]);
    
    /* duplicate as many comm_world communicators as number of threads in one dimension for east-west communication */
    east_west_comms_a = (MPI_Comm *) malloc(sizeof(MPI_Comm) * Th_dim);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &east_west_comms_a[loop_i]);

    east_west_comms_b = (MPI_Comm *) malloc(sizeof(MPI_Comm) * Th_dim);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_dup(MPI_COMM_WORLD, &east_west_comms_b[loop_i]);

    /* allocate working arrays & communication buffers */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    if (!aold) {
        fprintf(stderr, "Unable to allocate memory for aold\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    if (!anew) {
        fprintf(stderr, "Unable to allocate memory for anew\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#if FINE_TIME
    t_comm = calloc(nthreads, sizeof(double));
    if (!t_comm) {
        fprintf(stderr, "Unable to allocate memory for t_comm\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    t_pack = calloc(nthreads, sizeof(double));
    if (!t_pack) {
        fprintf(stderr, "Unable to allocate memory for t_pack\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    t_halox = calloc(nthreads, sizeof(double));
    if (!t_halox) {
        fprintf(stderr, "Unable to allocate memory for t_halox\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
    
    //printf("Rank %d: North %d, South %d, East %d, West %d\n", rank, north, south, east, west); exit(0);

#pragma omp parallel
    {
#if FINE_TIME
        double t_comm_start, my_t_comm;
        double t_pack_start, my_t_pack;
#endif
        double *north_out, *south_out, *east_out, *west_out;
        double *north_in, *south_in, *east_in, *west_in;
        int i, j, iter;
        int thread_id = omp_get_thread_num();
        int Th_dim_x = thread_id % Th_dim;
        int Th_dim_y = thread_id / Th_dim;
        int my_north = north;
        int my_south = south;
        int my_east = east;
        int my_west = west;
        int my_niters = niters;

        MPI_Comm my_north_neighbor_comm, my_south_neighbor_comm;
        MPI_Comm my_east_neighbor_comm, my_west_neighbor_comm;

#if FINE_TIME
        t_comm[thread_id] = 0;
        t_pack[thread_id] = 0;
#endif
        int xstart = THX_START;
        int xend = THX_END;
        int ystart = THY_START;
        int yend = THY_END;
        int xrange = xend - xstart + 1;
        int yrange = yend - ystart + 1;
        if (xrange != Thx || yrange != Thy)
            printf("Error: xrange not the same as Thx OR yrange not the same as Thy!\n");

        /* Initialize aold and anew using "first touch", *including boundaries* */
        /* Note that MPI_Alloc_mem does not initialize the memory
         * (this is a good thing in this case) */
        /* temporarily update xstart and xend for the initialization
         * and reset them once we are done */
        if (xstart == 1)
            xstart = 0;
        if (xend == bx)
            xend = bx + 2;
        if (ystart == 1)
            ystart = 0;
        if (yend == by)
            yend = by + 2;
        for (j = ystart; j < yend; ++j) {
            for (i = xstart; i < xend; ++i) {
                aold[ind(i, j)] = 0.0;
                anew[ind(i, j)] = 0.0;
            }
        }
        xstart = THX_START;
        xend = THX_END;
        ystart = THY_START;
        yend = THY_END;

        /* If I am not at the northern edge, I will not participate in communications involving my north neighbor */
        if (ystart != 1)
            my_north = MPI_PROC_NULL;
        /* If I am not at the eastern edge, I will not participate in communications involving my east neighbor */
        if (xend != bx)
            my_east = MPI_PROC_NULL;
        /* If I am not at the southern edge, I will not participate in communications involving my south neighbor */
        if (yend != by)
            my_south = MPI_PROC_NULL;
        /* If I am not at the western edge, I will not participate in communications involving my west neighbor */
        if (xstart != 1)
            my_west = MPI_PROC_NULL;

        /* Choose the correct communicators for exchange with neighbors */
        my_north_neighbor_comm = (ry % 2) ? north_south_comms_b[Th_dim_x] : north_south_comms_a[Th_dim_x];
        my_south_neighbor_comm = (ry % 2) ? north_south_comms_a[Th_dim_x] : north_south_comms_b[Th_dim_x];
        my_east_neighbor_comm = (rx % 2) ? east_west_comms_b[Th_dim_y]   : east_west_comms_a[Th_dim_y];
        my_west_neighbor_comm = (rx % 2) ? east_west_comms_a[Th_dim_y]   : east_west_comms_b[Th_dim_y];
        
        /* Allocate halo buffers to pack/unpack into */
        posix_memalign((void**) &north_out, PAGE_SIZE, xrange * sizeof(double));
        //north_out = (double *) calloc(xrange, sizeof(double));
        if (!north_out) {
            fprintf(stderr, "Unable to allocate memory for north_out\n");
            exit(0);
        }
        posix_memalign((void**) &south_out, PAGE_SIZE, xrange * sizeof(double));
        //south_out = (double *) calloc(xrange, sizeof(double));
        if (!south_out) {
            fprintf(stderr, "Unable to allocate memory for south_out\n");
            exit(0);
        }
        posix_memalign((void**) &east_out, PAGE_SIZE, yrange * sizeof(double));
        //east_out = (double *) calloc(yrange, sizeof(double));
        if (!east_out) {
            fprintf(stderr, "Unable to allocate memory for east_out\n");
            exit(0);
        }
        posix_memalign((void**) &west_out, PAGE_SIZE, yrange * sizeof(double));
        //west_out = (double *) calloc(yrange, sizeof(double));
        if (!west_out) {
            fprintf(stderr, "Unable to allocate memory for west_out\n");
            exit(0);
        }

        posix_memalign((void**) &north_in, PAGE_SIZE, xrange * sizeof(double));
        //north_in = (double *) calloc(xrange, sizeof(double));
        if (!north_in) {
            fprintf(stderr, "Unable to allocate memory for north_in\n");
            exit(0);
        }
        posix_memalign((void**) &south_in, PAGE_SIZE, xrange * sizeof(double));
        //south_in = (double *) calloc(xrange, sizeof(double));
        if (!south_in) {
            fprintf(stderr, "Unable to allocate memory for south_in\n");
            exit(0);
        }
        posix_memalign((void**) &east_in, PAGE_SIZE, yrange * sizeof(double));
        //east_in = (double *) calloc(yrange, sizeof(double));
        if (!east_in) {
            fprintf(stderr, "Unable to allocate memory for east_in\n");
            exit(0);
        }
        posix_memalign((void**) &west_in, PAGE_SIZE, yrange * sizeof(double));
        //west_in = (double *) calloc(yrange, sizeof(double));
        if (!west_in) {
            fprintf(stderr, "Unable to allocate memory for west_in\n");
            exit(0);
        }

#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#pragma omp barrier
#if FINE_TIME
        my_t_pack = my_t_comm = 0;
#else
#pragma omp master
        {
            t1 = MPI_Wtime();   /* take time */
        }
#endif
        
        //double t_init, t_prog;
        //double t_init_start, t_prog_start;

        //t_init = t_prog = 0;

        //if (rank == 4) printf("Rank 4 Thread %d: North %d, South %d, East %d, West %d\n", thread_id, my_north, my_south, my_east, my_west);

        //printf("Rank %d Thread %d: North %d, South %d, East %d, West %d\n", rank, thread_id, my_north_neighbor_comm, my_south_neighbor_comm, my_east_neighbor_comm, my_west_neighbor_comm);
        
        //printf("Rank %d Thread %d: xrange %d, yrange %d\n", rank, thread_id, xrange, yrange);
        
        for (iter = 0; iter < my_niters; ++iter) {
            /* create request array */
            MPI_Request reqs[8];

#if PACK
#if FINE_TIME
            t_pack_start = MPI_Wtime();
#endif
            /* Pack */
            if (my_north != MPI_PROC_NULL)
                for (i = 0; i < xrange; i++)
                    north_out[i] = aold[ind(xstart + i, 1)];
            if (my_south != MPI_PROC_NULL)
                for (i = 0; i < xrange; i++)
                    south_out[i] = aold[ind(xstart + i, by)];
            if (my_east != MPI_PROC_NULL)
                for (i = 0; i < yrange; i++)
                    east_out[i] = aold[ind(bx, ystart + i)];
            if (my_west != MPI_PROC_NULL)
                for (i = 0; i < yrange; i++)
                    west_out[i] = aold[ind(1, ystart + i)];
#if FINE_TIME
            my_t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif

            /* prevent cost of load imbalance in communication time */
#pragma omp master
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#pragma omp barrier

            /* Exchange data with neighbors.
             * Use (comm, tag) = (my_north_neighbor_comm, Th_dim_x) for exchange with north neighbor.
             * Use (comm, tag) = (my_south_neighbor_comm, Th_dim_x) for exchange with south neighbor.
             * Use (comm, tag) = (my_east_neighbor_comm, Th_dim_y) for exchange with east neighbor.
             * Use (comm, tag) = (my_west_neighbor_comm, Th_dim_y) for exchange with west neighbor.
             *
             * The south-neighbor of my north-neighbor is seemingly using a different communicator.
             * However, this is not the case. The assginment to the thread-local communicators
             * ensures correctness. For example, my my_north_neighbor_comm and my north-neighbor's
             * my_south_neighbor_comm are the same.
             */
#if FINE_TIME
            t_comm_start = MPI_Wtime();
#endif
            //t_init_start = MPI_Wtime();
            if (my_south != MPI_PROC_NULL)
                MPI_Irecv(south_in, xrange, MPI_DOUBLE, my_south, Th_dim_x, my_south_neighbor_comm, &reqs[0]);
            else
                reqs[0] = MPI_REQUEST_NULL;

            if (my_north != MPI_PROC_NULL)
                MPI_Irecv(north_in, xrange, MPI_DOUBLE, my_north, Th_dim_x, my_north_neighbor_comm, &reqs[1]);
            else 
                reqs[1] = MPI_REQUEST_NULL;
            
            if (my_west != MPI_PROC_NULL)
                MPI_Irecv(west_in, yrange, MPI_DOUBLE, my_west, Th_dim_y, my_west_neighbor_comm, &reqs[2]);
            else
                reqs[2] = MPI_REQUEST_NULL;
            
            if (my_east != MPI_PROC_NULL)
                MPI_Irecv(east_in , yrange, MPI_DOUBLE, my_east, Th_dim_y, my_east_neighbor_comm, &reqs[3]);
            else    
                reqs[3] = MPI_REQUEST_NULL;
            
            if (my_south != MPI_PROC_NULL)
                MPI_Isend(south_out, xrange, MPI_DOUBLE, my_south, Th_dim_x, my_south_neighbor_comm, &reqs[4]);
            else
                reqs[4] = MPI_REQUEST_NULL;
            
            if (my_north != MPI_PROC_NULL)
                MPI_Isend(north_out, xrange, MPI_DOUBLE, my_north, Th_dim_x, my_north_neighbor_comm, &reqs[5]);
            else
                reqs[5] = MPI_REQUEST_NULL;
            
            if (my_west != MPI_PROC_NULL)
                MPI_Isend(west_out, yrange, MPI_DOUBLE, my_west, Th_dim_y, my_west_neighbor_comm, &reqs[6]);
            else
                reqs[6] = MPI_REQUEST_NULL;
            
            if (my_east != MPI_PROC_NULL)
                MPI_Isend(east_out, yrange, MPI_DOUBLE, my_east, Th_dim_y, my_east_neighbor_comm, &reqs[7]);
            else
                reqs[7] = MPI_REQUEST_NULL;
            
            //t_init += (MPI_Wtime() - t_init_start);

            //t_prog_start = MPI_Wtime();
            MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
            //t_prog += (MPI_Wtime() - t_prog_start);
#if FINE_TIME
            my_t_comm += (MPI_Wtime() - t_comm_start);
#endif

#if PACK
#if FINE_TIME
            t_pack_start = MPI_Wtime();
#endif
            /* Unpack */
            if (my_north != MPI_PROC_NULL)
                for (i = 0; i < xrange; i++)
                    aold[ind(xstart + i, 0)] = north_in[i];;
            if (my_south != MPI_PROC_NULL)
                for (i = 0; i < xrange; i++)
                    aold[ind(xstart + i, by + 1)] = south_in[i];
            if (my_east != MPI_PROC_NULL)
                for (i = 0; i < yrange; i++)
                    aold[ind(0, ystart + i)] = west_in[i];
            if (my_west != MPI_PROC_NULL)
                for (i = 0; i < yrange; i++)
                    aold[ind(bx + 1, ystart + i)] = east_in[i];
#if FINE_TIME
            my_t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif

#if COMPUTE
            /* update grid */
            for (i = xstart; i <= xend; ++i) {
                for (j = ystart; j <= yend; ++j) {
                    anew[ind(i, j)] =
                        anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                                 aold[ind(i, j - 1)] +
                                                 aold[ind(i, j + 1)]) / 4.0 / 2.0;
                }
            }
#endif
#pragma omp barrier /* This barrier can be removed if we assign each thread their tile in the beginning of each iteration (the swap below would not depend on the completion of the compute above) */
#pragma omp master
            {
                /* swap working arrays */
                tmp = anew;
                anew = aold;
                aold = tmp;
            }
#pragma omp barrier
        }

#if FINE_TIME
        if (my_north == MPI_PROC_NULL &&
            my_south == MPI_PROC_NULL &&
            my_east == MPI_PROC_NULL &&
            my_west == MPI_PROC_NULL)
            t_comm[thread_id] = t_pack[thread_id] = t_halox[thread_id] = 0;
        else {
            t_comm[thread_id] = my_t_comm;
            t_pack[thread_id] = my_t_pack;
            t_halox[thread_id] = my_t_comm + my_t_pack;
        }
#endif

        //printf("Rank %d, thread %d: init_time %f\t prog_time %f\n", rank, thread_id, t_init, t_prog);

        free(north_out);
        free(south_out);
        free(east_out);
        free(west_out);
        free(north_in);
        free(south_in);
        free(east_in);
        free(west_in);
    }

    MPI_Barrier(MPI_COMM_WORLD);
#if FINE_TIME
    if (rank == 0) {
        t_comm_workers = calloc(size*nthreads, sizeof(double));
        if (!t_comm_workers) {
            fprintf(stderr, "Unable to allocate memory for t_comm_workers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_pack_workers = calloc(size*nthreads, sizeof(double));
        if (!t_pack_workers) {
            fprintf(stderr, "Unable to allocate memory for t_pack_workers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        t_halox_workers = calloc(size*nthreads, sizeof(double));
        if (!t_halox_workers) {
            fprintf(stderr, "Unable to allocate memory for t_halox_workers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else
        t_comm_workers = t_pack_workers = t_halox_workers = NULL;
    MPI_Gather(t_comm, nthreads, MPI_DOUBLE,
            t_comm_workers, nthreads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_pack, nthreads, MPI_DOUBLE,
            t_pack_workers, nthreads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_halox, nthreads, MPI_DOUBLE,
            t_halox_workers, nthreads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    t2 = MPI_Wtime();
#endif

    if (rank == 0) {
#if FINE_TIME
        int wi;
        double min_t_comm, max_t_comm, mean_t_comm, sum_t_comm;
        double min_t_pack, max_t_pack, mean_t_pack, sum_t_pack;
        double min_t_halox, max_t_halox, mean_t_halox, sum_t_halox;
        int workers_who_communicated;
 
        min_t_comm = min_t_pack = min_t_halox = 9999;
        max_t_comm = max_t_pack = max_t_halox = -1;
        sum_t_comm = sum_t_pack = sum_t_halox = 0;

        workers_who_communicated = 0;

        for (wi = 0; wi < size*nthreads; wi++) {
            if (t_comm_workers[wi] > 0) {
                if (max_t_comm < t_comm_workers[wi]) {
                    max_t_comm = t_comm_workers[wi];
                }
                if (min_t_comm > t_comm_workers[wi]) {
                    min_t_comm = t_comm_workers[wi];
                }
                sum_t_comm += t_comm_workers[wi];
                /* Those who communicated will have non-zero comm,pack,halox values*/
                workers_who_communicated++;
            }

            if (t_pack_workers[wi] > 0) {
                if (max_t_pack < t_pack_workers[wi]) {
                    max_t_pack = t_pack_workers[wi];
                }
                if (min_t_pack > t_pack_workers[wi]) {
                    min_t_pack = t_pack_workers[wi];
                }
                sum_t_pack += t_pack_workers[wi];
            }
 
            if (t_halox_workers[wi] > 0) {
                if (max_t_halox < t_halox_workers[wi]) {
                    max_t_halox = t_halox_workers[wi];
                }
                if (min_t_halox > t_halox_workers[wi]) {
                    min_t_halox = t_halox_workers[wi];
                }
                sum_t_halox += t_halox_workers[wi];
            }
        }

        mean_t_comm = sum_t_comm / workers_who_communicated;
        mean_t_pack = sum_t_pack / workers_who_communicated;
        mean_t_halox = sum_t_halox / workers_who_communicated;

        //printf("Rank %d had the max comm time and rank %d has the max pack time\n", max_comm_rank, max_pack_rank);
        printf("mesh_dim,px,py,iters,threads,min_comm_time,max_comm_time,mean_comm_time,min_pack_time,max_pack_time,mean_pack_time,min_halox_time,max_halox_time,mean_halox_time\n");
        printf("%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", n, px, py, niters, nthreads, min_t_comm, max_t_comm, mean_t_comm, min_t_pack, max_t_pack, mean_t_pack, min_t_halox, max_t_halox, mean_t_halox);
#else
        printf("mesh_dim,px,py,iters,threads,time\n");
        printf("%d,%d,%d,%d,%d,%f\n", n, px, py, niters, nthreads, t2 - t1);
#endif
    }
 
#if FINE_TIME
    free(t_pack);
    free(t_comm);
    free(t_halox);
    free(t_comm_workers);
    free(t_pack_workers);
    free(t_halox_workers);
#endif
   
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_free(&north_south_comms_a[loop_i]);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_free(&north_south_comms_b[loop_i]);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_free(&east_west_comms_a[loop_i]);
    for (loop_i = 0; loop_i < Th_dim; loop_i++)
        MPI_Comm_free(&east_west_comms_b[loop_i]); 

    free(north_south_comms_a);
    free(north_south_comms_b);
    free(east_west_comms_a);
    free(east_west_comms_b);
    
    /* free working arrays and communication buffers */
    free(aold);
    free(anew);
    
    MPI_Finalize();
    return 0;
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
