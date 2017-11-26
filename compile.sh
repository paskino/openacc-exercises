#!/bin/sh
export PATH=$PATH:/apps/pgi/linux86-64/17.4/bin
source=$1
gsize=$2

# pgaccelinfo -v

#pgcc -acc -Minfo -ta=tesla:cc20 -I/apps/pgi/linux86-64/17.4/include example$source.c -o gpu_accel$source
pgcc -acc -Minfo -ta=tesla:cc60 -DROWS=$gsize -DCOLS=$gsize example$source.c -o gpu_accel$source
pgcc -acc -Minfo -ta=multicore -DROWS=$gsize -DCOLS=$gsize example$source.c -o cpu_accel$source
pgcc  -Minfo -DROWS=$gsize -DCOLS=$gsize example$source.c -o serial$source
#/apps/pgi/linux86-64/17.4/bin/pgcc -I/apps/pgi/linux86-64/17.4/include example.c
#gcc -fopenacc -DROWS=$gsize -DCOLS=$gsize example$source.c -o gcc_gpu$source

