#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void insert_keys (int *new_keys, int *new_values, int numKeys, key_value_pair *bucket_1, key_value_pair *bucket_2, int bucket_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash, old;

	// Las sa ruleze un numar de threaduri egal cu numarul de perechi
	if (numKeys <= idx) {
        return;
    }

	// Incerc sa inserez in pozitia data de una dintre cele 3 functii de hash
	hash = hash1 (new_keys[idx], bucket_size);
	old = atomicCAS (&bucket_1[hash].key, KEY_INVALID, new_keys[idx]);
	if (old == KEY_INVALID) {
		atomicExch (&bucket_1[hash].value, new_values[idx]);
		return;
	}
	old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
    if (old == KEY_INVALID) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    hash = hash2 (new_keys[idx], bucket_size);
    old = atomicCAS (&bucket_1[hash].key, KEY_INVALID, new_keys[idx]);
    if (old == KEY_INVALID) {
        atomicExch (&bucket_1[hash].value, new_values[idx]);
        return;
    }
    old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
    if (old == KEY_INVALID) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    hash = hash3 (new_keys[idx], bucket_size);
    old = atomicCAS (&bucket_1[hash].key, KEY_INVALID, new_keys[idx]);
    if (old == KEY_INVALID) {
        atomicExch (&bucket_1[hash].value, new_values[idx]);
        return;
    }
    old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
    if (old == KEY_INVALID) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    // Daca nicio functie de hash nu a functionat, inserez in orice loc liber gasesc
    for (i = 0; i < bucket_size; i++) {
        old = atomicCAS (&bucket_1[i].key, KEY_INVALID, new_keys[idx]);
        if (old == KEY_INVALID) {
            atomicExch (&bucket_1[i].value, new_values[idx]);
            return;
        }

        old = atomicCAS (&bucket_2[i].key, KEY_INVALID, new_keys[idx]);
        if (old == KEY_INVALID) {
            atomicExch (&bucket_2[i].value, new_values[idx]);
            return;
        }
    }
}


/* MOVE DATA FROM OLD BUCKET TO NEW BIGGER BUCKET
*/
__global__ void move_bucket (key_value_pair *old_bucket, key_value_pair *new_bucket, int old_bucket_size, int new_bucket_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, old, hash;

	// Las sa ruleze un numar de threaduri egal cu marimea bucket-ului, doar pt intrarile cu valori in bucket
	if ((old_bucket_size <= idx) || (old_bucket[idx].key == KEY_INVALID)) {
		return;
	}

	// Incerc sa inserez in pozitia data de una dintre cele 3 functii de hash
	hash = hash1 (old_bucket[idx].key, new_bucket_size);
	old = atomicCAS (&new_bucket[hash].key, KEY_INVALID, old_bucket[idx].key);
	if (old == KEY_INVALID) {
        atomicExch (&new_bucket[hash].value, old_bucket[idx].value);
        return;
    }

	hash = hash2 (old_bucket[idx].key, new_bucket_size);
	old = atomicCAS (&new_bucket[hash].key, KEY_INVALID, old_bucket[idx].key);
	if (old == KEY_INVALID) {
        atomicExch (&new_bucket[hash].value, old_bucket[idx].value);
        return;
    }

	hash = hash3 (old_bucket[idx].key, new_bucket_size);
	old = atomicCAS (&new_bucket[hash].key, KEY_INVALID, old_bucket[idx].key);
	if (old == KEY_INVALID) {
        atomicExch (&new_bucket[hash].value, old_bucket[idx].value);
        return;
    }

	// Daca nicio functie de hash nu a functionat, inserez in orice loc liber gasesc
	for (i = 0; i < new_bucket_size; i++) {
		old = atomicCAS (&new_bucket[i].key, KEY_INVALID, old_bucket[idx].key);
		if (old == KEY_INVALID) {
            atomicExch (&new_bucket[i].value, old_bucket[idx].value);
            return;
        }
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

	// Sunt 2 bucketuri, deci in total avem 2 * size spatii pentru perechi cheie - valoare
	total_size = size * 2;
	free_size = size * 2;

	// Asa am vazut in lab
	bucket_1 = 0;
	bucket_2 = 0;

	// Aloc memorie pentru fiecare bucket si il setez la 0
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
	int blocks_number;

	// Verific daca marimea ceruta este valida
	if (numBucketsReshape <= total_size / 2) {
		return;
	}

	// Noile bucket-uri care le vor inlocui pe cele vechi
	key_value_pair *bucket_1_new;
	key_value_pair *bucket_2_new;

	bucket_1_new = 0;
	bucket_2_new = 0;

	// Aloc memorie pentru cele 2 noi bucket-uri
	cudaMalloc (&bucket_1_new, numBucketsReshape * sizeof (key_value_pair));
    cudaMemset (&bucket_2_new, 0, numBucketsReshape * sizeof (key_value_pair));

    cudaMalloc (&bucket_1_new, numBucketsReshape * sizeof (key_value_pair));
    cudaMemset (&bucket_2_new, 0, numBucketsReshape * sizeof (key_value_pair));

	// Calculez cate blocuri vor rula
    blocks_number = (total_size / 2) / THREADS_NUMBER + 1;

    // Trec datele din vechile bucket-uri in cele noi
    move_bucket <<<blocks_number, THREADS_NUMBER>>> (bucket_1, bucket_1_new, total_size / 2, numBucketsReshape);
    move_bucket <<<blocks_number, THREADS_NUMBER>>> (bucket_2, bucket_2_new, total_size / 2, numBucketsReshape);

    // Astept ca toate blocurile sa se termine
    cudaDeviceSynchronize();

    // Updatez metricile
    free_size += (numBucketsReshape * 2 - total_size);
    total_size = numBucketsReshape * 2;

    // Inlocuiesc vechile bucket-uri cu cele noi
    cudaFree (bucket_1);
    cudaFree (bucket_2);

    bucket_1 = bucket_1_new;
    bucket_2 = bucket_2_new;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int blocks_number;

	// Mut perechile cheie - valoare in memoria device-ului
	int *new_keys;
	int *new_values;

	int size = numKeys * sizeof (int);
	new_keys = 0;
	new_values = 0;

	cudaMalloc (&new_keys, size);
    cudaMalloc (&new_values, size);

    cudaMemcpy (new_keys, keys, size, cudaMemcpyHostToDevice);
    cudaMemcpy (new_values, values, size, cudaMemcpyHostToDevice);

    // Daca noile chei nu incap, fac reshape pentru a avea un load factor de 66% dupa adaugarea noilor chei
    if (free_size < numKeys) {
        reshape ((total_size - free_size + numKeys) * (150 / 100) / 2 + 2);
    }

	// Calculez cate blocuri vor rula
    blocks_number = numKeys / THREADS_NUMBER + 1;

    insert_keys <<<blocks_number, THREADS_NUMBER>>> (new_keys, new_values, numKeys, bucket_1, bucket_2, total_size / 2);

    // Astept ca toate blocurile sa se termine
    cudaDeviceSynchronize();

    free_size -= numKeys;

	// Sterg din memoria device-ului perechile cheie - valoare
    cudaFree (new_keys);
    cudaFree (new_values);

	return true;
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
	if (total_size == 0) {
		return 0.0f;
	}

	return (float)((total_size - free_size) / total_size);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
