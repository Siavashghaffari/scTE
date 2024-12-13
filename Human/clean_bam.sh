#!/bin/bash



ml samtools/1.18-GCC-6.3.0-2.28

# pre-filtering to remove the empty CB reads
samtools view possorted_genome_bam.bam -h | awk '/^@/ || /CB:/' | samtools view -h -b > possorted_genome_bam.clean.bam

# Convert BAM to SAM
samtools view -h possorted_genome_bam.clean.bam > input.sam

# Replace all instances of 'GRCh38-2020-A_chr' with 'chr'
sed -i 's/GRCh38-2020-A_chr/chr/g' input.sam


# Convert SAM back to BAM
samtools view -bS input.sam > output.bam


# Index the new BAM file (optional)
samtools index output.bam


# delete sam file
rm input.sam

