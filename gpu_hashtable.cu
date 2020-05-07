#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	total_size = size;
	free_size = size;

	cudaMalloc (&bucket_1, size * sizeof (key_value_pair));
	cudaMemset (&bucket_1, 0, size * sizeof (key_value_pair));

	cudaMalloc (&bucket_2, size * sizeof (key_value_pair));
    cudaMemset (&bucket_2, 0, size * sizeof (key_value_pair));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(bucket_1);
	cudaFree(bucket_2);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	return false;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	return NULL;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return 0.f; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
