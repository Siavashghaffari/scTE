#!/bin/bash



ml samtools/1.18-GCC-6.3.0-2.28

# pre-filtering to remove the empty CB reads
samtools view possorted_genome_bam.bam -h | awk '/^@/ || /CB:/' | samtools view -h -b > possorted_genome_bam.clean.bam




