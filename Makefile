# -*- Mode: Makefile; -*-
CC=icc
CFLAGS= -g3 -O3 -Wall -lmpi
OMPFLAGS= -fopenmp
BINS=2d_stencil_multiple_multicomms_reg_part_pack 2d_stencil_multiple_comm_world_reg_part_pack
BINS+=2d_stencil_single_pack
BINS+=2d_stencil_single_shared_mem_pack
BINS+=2d_stencil_funneled_pack
BINS+=2d_stencil_funneled_parallel_pack

all: $(BINS)

2d_stencil_multiple_multicomms_reg_part_pack: 2d_stencil_multiple_multicomms_reg_part_pack.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ -lm

2d_stencil_multiple_comm_world_reg_part_pack: 2d_stencil_multiple_comm_world_reg_part_pack.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ -lm

2d_stencil_single_pack: 2d_stencil_single_pack.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

2d_stencil_single_shared_mem_pack: 2d_stencil_single_shared_mem_pack.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

2d_stencil_funneled_pack: 2d_stencil_funneled_pack.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ -lm

2d_stencil_funneled_parallel_pack: 2d_stencil_funneled_parallel_pack.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
