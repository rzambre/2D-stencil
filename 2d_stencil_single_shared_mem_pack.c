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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* row-major order */
#define ind(col,row) ((row)*(bx+2)+(col))
#define COMPUTE 0
#define PACK 1
#define FINE_TIME 1

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *node_x_ptr, int *node_y_ptr, int *final_flag);

void alloc_bufs(size_t bx, size_t by,
                double **aold_ptr, double **anew_ptr,
                double **sbufnorth_ptr, double **sbufsouth_ptr,
                double **sbufeast_ptr, double **sbufwest_ptr,
                double **rbufnorth_ptr, double **rbufsouth_ptr,
                double **rbufeast_ptr, double **rbufwest_ptr);

int main(int argc, char **argv)
{
    int rank, size, rank_in_parray;
    int n, energy, niters, px, py, node_x, node_y;
    int node_i, node_j, in_node_i, in_node_j, ppn_x, ppn_y, ppn;

    int rx, ry;
    int north_in_parray, south_in_parray, west_in_parray, east_in_parray;
    int north, south, west, east;
    size_t bx, by;

    double *north_out, *south_out, *east_out, *west_out;
    double *north_in, *south_in, *east_in, *west_in;

#if FINE_TIME
    double t_pack_start, t_comm_start;
    double t_pack, t_comm;
    double max_t_pack, max_t_comm;
    double *t_comm_procs, *t_pack_procs;
#else
    double t1, t2;
#endif

    int iter;

    double *aold, *anew, *tmp;
    size_t i;
#if COMPUTE
    int j;
#endif
    
    MPI_Comm shm_comm;
    MPI_Group world_group, shared_group;
    int *rank_array, *global_to_shm_rank_map;
    int shm_rank, shm_size;

    int final_flag;

    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);

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

    /* create shared memory communicator */
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_size(shm_comm, &shm_size);
    MPI_Comm_rank(shm_comm, &shm_rank);

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Comm_group(shm_comm, &shared_group);

    rank_array = malloc(size * sizeof(int));
    global_to_shm_rank_map = malloc(size * sizeof(int));
    for (i = 0; i < size; i++)
        rank_array[i] = i;
    MPI_Group_translate_ranks(world_group, size, rank_array, shared_group, global_to_shm_rank_map);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &node_x, &node_y, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    ppn_x = (px / node_x);
    ppn_y = (py / node_y);
    ppn = ppn_x * ppn_y;
    /* find my rank in the processor array from my rank in COMM_WORLD */
    node_i = (rank / ppn) % node_x ;
    node_j = rank / (ppn * node_x);
    in_node_i = rank % ppn_x;
    in_node_j = (rank / ppn_x) % ppn_y;
    rank_in_parray = (node_j * ppn * node_x) + (in_node_j * px) + (node_i * ppn_x) + in_node_i;
    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank_in_parray % px;
    ry = rank_in_parray / px;

    /* determine my four neighbors */
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    else {
        north_in_parray = (ry - 1) * px + rx;
        node_i = (north_in_parray / ppn_x) % node_x;
        node_j = north_in_parray / (ppn * node_x);
        in_node_i = north_in_parray % ppn_x;
        in_node_j = (north_in_parray / px) % ppn_y;
        north = (node_j * ppn * node_x) + (node_i * ppn) + (in_node_j * ppn_x) + in_node_i;
        if (global_to_shm_rank_map[north] != MPI_UNDEFINED)
            north = MPI_PROC_NULL;
    }
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    else {
        south_in_parray = (ry + 1) * px + rx;
        node_i = (south_in_parray / ppn_x) % node_x;
        node_j = south_in_parray / (ppn * node_x);
        in_node_i = south_in_parray % ppn_x;
        in_node_j = (south_in_parray / px) % ppn_y;
        south = (node_j * ppn * node_x) + (node_i * ppn) + (in_node_j * ppn_x) + in_node_i;
        if (global_to_shm_rank_map[south] != MPI_UNDEFINED)
            south = MPI_PROC_NULL;
    }
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    else {
        west_in_parray = ry * px + rx - 1;
        node_i = (west_in_parray / ppn_x) % node_x;
        node_j = west_in_parray / (ppn * node_x);
        in_node_i = west_in_parray % ppn_x;
        in_node_j = (west_in_parray / px) % ppn_y;
        west = (node_j * ppn * node_x) + (node_i * ppn) + (in_node_j * ppn_x) + in_node_i;
        if (global_to_shm_rank_map[west] != MPI_UNDEFINED)
            west = MPI_PROC_NULL;
    }
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;
    else {
        east_in_parray = ry * px + rx + 1;
        node_i = (east_in_parray / ppn_x) % node_x;
        node_j = east_in_parray / (ppn * node_x);
        in_node_i = east_in_parray % ppn_x;
        in_node_j = (east_in_parray / px) % ppn_y;
        east = (node_j * ppn * node_x) + (node_i * ppn) + (in_node_j * ppn_x) + in_node_i;
        if (global_to_shm_rank_map[east] != MPI_UNDEFINED)
            east = MPI_PROC_NULL;
    }
    
    //printf("Rank %d: North %d, South %d, East %d, West %d\n", rank, north, south, east, west); exit(0);

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */

    
    /* TODO: allocate memory using Win_allocate_shared. Don't need it for communication only */
    /* allocate and initialize working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew,
               &north_out, &south_out, &east_out, &west_out,
               &north_in, &south_in, &east_in, &west_in);

    //printf("Rank %d: bx %d, by %d\n", rank, bx, by); exit(0);
    MPI_Barrier(MPI_COMM_WORLD);

#if FINE_TIME
    t_pack = t_comm = 0;
#else
    t1 = MPI_Wtime();   /* take time */
#endif

    //double t_init, t_progress;
    //double t_init_start, t_progress_start;

    //t_init = t_progress = 0;

    for (iter = 0; iter < niters; ++iter) {
        
        /* create request array */
        MPI_Request reqs[8];

#if PACK
#if FINE_TIME
        t_pack_start = MPI_Wtime();
#endif
        /* pack into buffers */
        if (north != MPI_PROC_NULL)
            for (i = 0; i < bx; ++i)
                north_out[i] = aold[ind(i + 1, 1)];
        if (south != MPI_PROC_NULL)
            for (i = 0; i < bx; ++i)
                south_out[i] = aold[ind(i + 1, by)];
        if (east != MPI_PROC_NULL)
            for (i = 0; i < by; ++i)
                east_out[i] = aold[ind(bx, i + 1)];
        if (west != MPI_PROC_NULL)    
            for (i = 0; i < by; ++i)
                west_out[i] = aold[ind(1, i + 1)];
#if FINE_TIME
        t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif
        /* prevent cost of load imbalance in communication time */
        MPI_Barrier(MPI_COMM_WORLD);
#if FINE_TIME
        t_comm_start = MPI_Wtime();
#endif
        //t_init_start = MPI_Wtime();
        /* Communication will not occur for neighbors that are MPI_PROC_NULL */
        MPI_Irecv(south_in, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(north_in, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(west_in, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(east_in, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[3]);
        MPI_Isend(south_out, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[4]);
        MPI_Isend(north_out, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[5]);
        MPI_Isend(west_out, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[6]);
        MPI_Isend(east_out, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[7]);
        //t_init += (MPI_Wtime() - t_init_start);

        //t_progress_start = MPI_Wtime();
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
        //t_progress += (MPI_Wtime() - t_progress_start);
#if FINE_TIME
        t_comm += (MPI_Wtime() - t_comm_start);
#endif
#if PACK
#if FINE_TIME
        t_pack_start = MPI_Wtime();
#endif
        /* unpack from buffers */
        if (north != MPI_PROC_NULL)
            for (i = 0; i < bx; ++i)
                aold[ind(i + 1, 0)] = north_in[i];
        if (south != MPI_PROC_NULL)
            for (i = 0; i < bx; ++i)
                aold[ind(i + 1, by + 1)] = south_in[i];
        if (east != MPI_PROC_NULL)
            for (i = 0; i < by; ++i)
                aold[ind(bx + 1, i + 1)] = east_out[i];
        if (west != MPI_PROC_NULL)    
            for (i = 0; i < by; ++i)
                aold[ind(0, i + 1)] = west_out[i];
#if FINE_TIME
        t_pack += (MPI_Wtime() - t_pack_start);
#endif
#endif

#if COMPUTE
        /* update grid */
        /* TODO: include computation that uses pointers obtained from MPI_Win_shared_query */
#endif
        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;
        
        MPI_Barrier(shm_comm);
    }

    //printf("Rank %d: init_time %f\t prog_time %f\n", rank, t_init, t_progress);
    
    MPI_Barrier(MPI_COMM_WORLD);
#if FINE_TIME
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
    } else
        t_pack_procs = t_comm_procs = NULL;
    MPI_Gather(&t_pack, 1, MPI_DOUBLE,
            t_pack_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_comm, 1, MPI_DOUBLE,
            t_comm_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    t2 = MPI_Wtime();
#endif

    MPI_Comm_free(&shm_comm);

    if (rank == 0) {
#if FINE_TIME
        int pi;
        int max_comm_rank, max_pack_rank;
        max_t_comm = t_comm_procs[0];
        max_t_pack = t_pack_procs[0];
        max_comm_rank = max_pack_rank = 0;
        for (pi = 0; pi < size; pi++) {
            if (max_t_comm < t_comm_procs[pi]) {
                max_t_comm = t_comm_procs[pi];
                max_comm_rank = pi;
            }
            if (max_t_pack < t_pack_procs[pi]) {
                max_t_pack = t_pack_procs[pi];
                max_pack_rank = pi;
            }
        }
        //printf("Rank %d had the max comm time and rank %d has the max pack time\n", max_comm_rank, max_pack_rank);
        printf("mesh_dim,px,py,iters,comm_time,pack_time\n");
        printf("%d,%d,%d,%d,%f,%f\n", n, px, py, niters, max_t_comm, max_t_pack);

#else
        printf("mesh_dim,px,py,iters,time\n");
        printf("%d,%d,%d,%d,%f\n", n, px, py, niters, t2 - t1);
#endif
    }

    /* free working arrays and communication buffers */
#if FINE_TIME
    free(t_comm_procs);
    free(t_pack_procs);
#endif
    
    free(rank_array);
    free(global_to_shm_rank_map);

    free(aold);
    free(anew);
    free(north_in);
    free(south_in);
    free(east_in);
    free(west_in);
    free(north_out);
    free(south_out);
    free(east_out);
    free(west_out);

    MPI_Finalize();
    return 0;
}

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
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *node_x_ptr, int *node_y_ptr, int *final_flag)
{
    int n, energy, niters, px, py, node_x, node_y;

    (*final_flag) = 0;

    if (argc < 6) {
        if (!rank)
            printf("usage: stencil_mpi <n> <energy> <niters> <px> <py> <node-x> <node-y>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */
    px = atoi(argv[4]); /* 1st dim processes */
    py = atoi(argv[5]); /* 2nd dim processes */
    node_x = atoi(argv[6]); /* 1st dim nodes */
    node_y = atoi(argv[7]); /* 2nd dim nodes */

    if (px * py != proc)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
    if (n % py != 0)
        MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
    if (n % px != 0)
        MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */
    if (px % node_x != 0)
        MPI_Abort(MPI_COMM_WORLD, 4);   /* abort node_x needs to divide px */
    if (py % node_y != 0)
        MPI_Abort(MPI_COMM_WORLD, 5);   /* abort node_y needs to divide py */

    (*n_ptr) = n;
    (*energy_ptr) = energy;
    (*niters_ptr) = niters;
    (*px_ptr) = px;
    (*py_ptr) = py;
    (*node_x_ptr) = node_x;
    (*node_y_ptr) = node_y;
}
