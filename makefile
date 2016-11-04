all: 
	nvcc -o reverseArray_multiblock reverseArray_multiblock.cu -I./include
	nvcc -o reverseArray_multiblock_fast reverseArray_multiblock_fast.cu -I./include
	nvcc -o reverseArray_singleblock reverseArray_singleblock.cu -I./include
	qsub batch.node10
	qsub batch.node11
