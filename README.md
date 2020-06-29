# 2D Stencil

This repository contains mini-apps for a 2D stencil solver
written in various ways. The mini-apps are extremely minimal
and primarily aim to capture the communication pattern of the
stencil pattern. The computation part may not be realistic.

# Versions

2d_stencil_single_pack.c
- MPI everywhere (a.k.a. MPI_THREAD_SINGLE)
- "pack" refers to packing of buffers (as opposed to using MPI datatypes)

2d_stencil_single_shared_mem_pack.c
- MPI everywhere + MPI Shared Memory

2d_stencil_funneled_pack.c
- MPI+OpenMP with MPI_THREAD_FUNNELED mode
- Serial packing of buffers

2d_stencil_funneled_parallel_pack.c
- MPI+OpenMP with MPI_THREAD_FUNNELED mode
- Parallel packing of buffers

2d_stencil_multiple_comm_world_reg_part_pack.c
- MPI+OpenMP with MPI_THREAD_MULTIPLE mode
- Expressing no logical parallelism to the MPI library
  - All threads use MPI_COMM_WORLD
- "reg_part" means that the domain is partitioned between threads the same way as in MPI everywhere
  - 2D partitioning of the domain

2d_stencil_multiple_multicomms_reg_part_pack.c
- MPI+OpenMP with MPI_THREAD_MULTIPLE mode
- Expressing logical parallelism to the MPI library
  - Threads issue operations in parallel on different communicators (design described in Section 6.1 of [1]).

[1] https://dl.acm.org/doi/abs/10.1145/3392717.3392773
