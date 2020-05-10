#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void set_zero (key_value_pair *bucket, int bucket_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (bucket_size <= idx) {
        return;
    }

    bucket[idx].key = 0;
    bucket[idx].value = 0;
}


__global__ void get_keys (int *new_keys, int *new_values, int numKeys, key_value_pair *bucket_1, key_value_pair *bucket_2, int bucket_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i, hash;

    // Las sa ruleze un numar de threaduri egal cu numarul de perechi
    if (numKeys <= idx) {
        return;
    }

    // Incerc sa scot din pozitia data de una dintre cele 3 functii de hash
    hash = hash1 (new_keys[idx], bucket_size);

    if (new_keys[idx] == bucket_1[hash].key) {
        new_values[idx] = bucket_1[hash].value;
        return;
    }
    if (new_keys[idx] == bucket_2[hash].key) {
        new_values[idx] = bucket_2[hash].value;
        return;
    }

    hash = hash2 (new_keys[idx], bucket_size);

    if (new_keys[idx] == bucket_1[hash].key) {
        new_values[idx] = bucket_1[hash].value;
        return;
    }
    if (new_keys[idx] == bucket_2[hash].key) {
        new_values[idx] = bucket_2[hash].value;
        return;
    }

    hash = hash3 (new_keys[idx], bucket_size);

    if (new_keys[idx] == bucket_1[hash].key) {
        new_values[idx] = bucket_1[hash].value;
        return;
    }
    if (new_keys[idx] == bucket_2[hash].key) {
        new_values[idx] = bucket_2[hash].value;
        return;
    }

    // Daca nicio functie de hash nu a functionat
    for (i = 0; i < bucket_size; i++) {
        if (new_keys[idx] == bucket_1[i].key) {
            new_values[idx] = bucket_1[i].value;
            return;
        }
        if (new_keys[idx] == bucket_2[i].key) {
            new_values[idx] = bucket_2[i].value;
            return;
        }
    }
}

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
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
		atomicExch (&bucket_1[hash].value, new_values[idx]);
		return;
	}
	old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    hash = hash2 (new_keys[idx], bucket_size);
    old = atomicCAS (&bucket_1[hash].key, KEY_INVALID, new_keys[idx]);
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
        atomicExch (&bucket_1[hash].value, new_values[idx]);
        return;
    }
    old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    hash = hash3 (new_keys[idx], bucket_size);
    old = atomicCAS (&bucket_1[hash].key, KEY_INVALID, new_keys[idx]);
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
        atomicExch (&bucket_1[hash].value, new_values[idx]);
        return;
    }
    old = atomicCAS (&bucket_2[hash].key, KEY_INVALID, new_keys[idx]);
	if ((old == KEY_INVALID) || (old == new_keys[idx])) {
        atomicExch (&bucket_2[hash].value, new_values[idx]);
        return;
    }

    // Daca nicio functie de hash nu a functionat, inserez in orice loc liber gasesc
    for (i = 0; i < bucket_size; i++) {
        old = atomicCAS (&bucket_1[i].key, KEY_INVALID, new_keys[idx]);
		if ((old == KEY_INVALID) || (old == new_keys[idx])) {
            atomicExch (&bucket_1[i].value, new_values[idx]);
            return;
        }

        old = atomicCAS (&bucket_2[i].key, KEY_INVALID, new_keys[idx]);
		if ((old == KEY_INVALID) || (old == new_keys[idx])) {
            atomicExch (&bucket_2[i].value, new_values[idx]);
            return;
        }
    }

    bucket_1[idx].value = -19;
    bucket_1[idx].key = -19;
}


/* MOVE DATA FROM OLD BUCKET TO NEW BIGGER BUCKET
*/
__global__ void move_bucket (key_value_pair *old_bucket, key_value_pair *new_bucket1, key_value_pair *new_bucket2, int old_bucket_size, int new_bucket_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, old, hash;

	// Las sa ruleze un numar de threaduri egal cu marimea bucket-ului, doar pt intrarile cu valori in bucket
	if ((old_bucket_size <= idx) || (old_bucket[idx].key == KEY_INVALID)) {
		return;
	}

	// Incerc sa inserez in pozitia data de una dintre cele 3 functii de hash
	hash = hash1 (old_bucket[idx].key, new_bucket_size);

	old = atomicCAS (&new_bucket1[hash].key, KEY_INVALID, old_bucket[idx].key);
	if (old == KEY_INVALID) {
        atomicExch (&new_bucket1[hash].value, old_bucket[idx].value);
        return;
    }
    old = atomicCAS (&new_bucket2[hash].key, KEY_INVALID, old_bucket[idx].key);
    if (old == KEY_INVALID) {
        atomicExch (&new_bucket2[hash].value, old_bucket[idx].value);
        return;
    }

	hash = hash2 (old_bucket[idx].key, new_bucket_size);

	old = atomicCAS (&new_bucket1[hash].key, KEY_INVALID, old_bucket[idx].key);
    if (old == KEY_INVALID) {
        atomicExch (&new_bucket1[hash].value, old_bucket[idx].value);
        return;
    }
    old = atomicCAS (&new_bucket2[hash].key, KEY_INVALID, old_bucket[idx].key);
    if (old == KEY_INVALID) {
        atomicExch (&new_bucket2[hash].value, old_bucket[idx].value);
        return;
    }

	hash = hash3 (old_bucket[idx].key, new_bucket_size);

	old = atomicCAS (&new_bucket1[hash].key, KEY_INVALID, old_bucket[idx].key);
    if (old == KEY_INVALID) {
        atomicExch (&new_bucket1[hash].value, old_bucket[idx].value);
        return;
    }
    old = atomicCAS (&new_bucket2[hash].key, KEY_INVALID, old_bucket[idx].key);
    if (old == KEY_INVALID) {
        atomicExch (&new_bucket2[hash].value, old_bucket[idx].value);
        return;
    }

	// Daca nicio functie de hash nu a functionat, inserez in orice loc liber gasesc
	for (i = 0; i < new_bucket_size; i++) {
		old = atomicCAS (&new_bucket1[i].key, KEY_INVALID, old_bucket[idx].key);
		if (old == KEY_INVALID) {
            atomicExch (&new_bucket1[i].value, old_bucket[idx].value);
            return;
        }

        old = atomicCAS (&new_bucket2[i].key, KEY_INVALID, old_bucket[idx].key);
        if (old == KEY_INVALID) {
            atomicExch (&new_bucket2[i].value, old_bucket[idx].value);
            return;
        }
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	int rc;
	int blocks_number;

	// Sunt 2 bucketuri, deci avem (size / 2 + 1) spatii per bucket
	total_size = (size / 2 + 1) * 2;
	free_size = total_size;

	// Asa am vazut in lab
	bucket_1 = 0;
	bucket_2 = 0;

	// Aloc memorie pentru fiecare bucket si il setez la 0
	rc = cudaMalloc (&bucket_1, (total_size / 2) * sizeof (key_value_pair));
	DIE (rc != cudaSuccess, "Eroare in init la alocare bucket_1!");

	rc = cudaMalloc (&bucket_2, (total_size / 2) * sizeof (key_value_pair));
	DIE (rc != cudaSuccess, "Eroare in init la alocare bucket_2!");

    blocks_number = (total_size / 2) / THREADS_NUMBER + 1;
    set_zero <<<blocks_number, THREADS_NUMBER>>> (bucket_1, (total_size / 2));
    cudaDeviceSynchronize();
    set_zero <<<blocks_number, THREADS_NUMBER>>> (bucket_2, (total_size / 2));
    cudaDeviceSynchronize();
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
	int rc;
	int blocks_number;

	// Verific daca marimea ceruta este valida
	if (numBucketsReshape <= total_size) {
		return;
	}

	// Noile bucket-uri care le vor inlocui pe cele vechi
	key_value_pair *bucket_1_new;
	key_value_pair *bucket_2_new;

	bucket_1_new = 0;
	bucket_2_new = 0;

	// Aloc memorie pentru cele 2 noi bucket-uri
	rc = cudaMalloc (&bucket_1_new, (numBucketsReshape / 2 + 1) * sizeof (key_value_pair));
	DIE (rc != cudaSuccess, "Eroare in reshape la alocare bucket_1_new!");

    rc = cudaMalloc (&bucket_2_new, (numBucketsReshape / 2 + 1) * sizeof (key_value_pair));
    DIE (rc != cudaSuccess, "Eroare in reshape la alocare bucket_2_new!");

    blocks_number = (numBucketsReshape / 2 + 1) / THREADS_NUMBER + 1;
    set_zero <<<blocks_number, THREADS_NUMBER>>> (bucket_1_new, (numBucketsReshape / 2 + 1));
    cudaDeviceSynchronize();
    set_zero <<<blocks_number, THREADS_NUMBER>>> (bucket_2_new, (numBucketsReshape / 2 + 1));
    cudaDeviceSynchronize();

	// Calculez cate blocuri vor rula
    blocks_number = (total_size / 2) / THREADS_NUMBER + 1;

    // Trec datele din vechile bucket-uri in cele noi
    move_bucket <<<blocks_number, THREADS_NUMBER>>> (bucket_1, bucket_1_new, bucket_2_new, (total_size / 2), (numBucketsReshape / 2 + 1));
    cudaDeviceSynchronize();

    move_bucket <<<blocks_number, THREADS_NUMBER>>> (bucket_2, bucket_1_new, bucket_2_new, (total_size / 2), (numBucketsReshape / 2 + 1));
    cudaDeviceSynchronize();

    // Updatez metricile
    free_size += (((numBucketsReshape / 2 + 1) * 2) - total_size);
    total_size = (numBucketsReshape / 2 + 1) * 2;

    // Inlocuiesc vechile bucket-uri cu cele noi
    cudaFree (bucket_1);
    cudaFree (bucket_2);

    bucket_1 = bucket_1_new;
    bucket_2 = bucket_2_new;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int rc;
	int blocks_number;

	// Mut perechile cheie - valoare in memoria device-ului
	int *new_keys;
	int *new_values;

	new_keys = 0;
	new_values = 0;

	rc = cudaMalloc (&new_keys, numKeys * sizeof (int));
	DIE (rc != cudaSuccess, "Eroare in insertBatch la alocare new_keys!");
	cudaMemset (&new_keys, 0, numKeys * sizeof (int));

    rc = cudaMalloc (&new_values, numKeys * sizeof (int));
    DIE (rc != cudaSuccess, "Eroare in insertBatch la alocare new_values!");
    cudaMemset (&new_values, 0, numKeys * sizeof (int));

    cudaMemcpy (new_keys, keys, numKeys * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy (new_values, values, numKeys * sizeof (int), cudaMemcpyHostToDevice);

    // Daca cu noile chei se umple hashmapul mai mult de 75%, fac reshape pentru a avea un load factor de 50% dupa adaugarea noilor chei
    if ((total_size - free_size + numKeys) > ((float)((float)(95.00f / 100.00f) * (float)total_size))) {
        reshape ((total_size - free_size + numKeys) * 100 / 80);
    }

	// Calculez cate blocuri vor rula
    blocks_number = numKeys / THREADS_NUMBER + 1;

    insert_keys <<<blocks_number, THREADS_NUMBER>>> (new_keys, new_values, numKeys, bucket_1, bucket_2, (total_size / 2));

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
	int rc;
	int blocks_number;

	// Aloc memorie pentru chei si valori in VRAM
	int *new_keys;
	int *new_values;
	int *results;

	new_keys = 0;
	new_values = 0;

	rc = cudaMalloc (&new_keys, numKeys * sizeof (int));
    DIE (rc != cudaSuccess, "Eroare in getBatch la alocare new_keys!");
    cudaMemset (&new_keys, 0, numKeys * sizeof (int));

    cudaMalloc (&new_values, numKeys * sizeof (int));
    DIE (rc != cudaSuccess, "Eroare in getBatch la alocare new_values!");
    cudaMemset (&new_values, 0, numKeys * sizeof (int));

	// Copiez cheile in VRAM
    cudaMemcpy (new_keys, keys, numKeys * sizeof (int), cudaMemcpyHostToDevice);

    // Calculez cate blocuri vor rula
    blocks_number = numKeys / THREADS_NUMBER + 1;

	get_keys <<<blocks_number, THREADS_NUMBER>>> (new_keys, new_values, numKeys, bucket_1, bucket_2, (total_size / 2));

    // Astept ca toate blocurile sa se termine
	cudaDeviceSynchronize();

	// Copiez in memoria host-ului
	results = (int*) malloc (numKeys * sizeof (int));
    cudaMemcpy (results, new_values, numKeys * sizeof (int), cudaMemcpyDeviceToHost);

	// Sterg din memoria device-ului perechile cheie - valoare
    cudaFree (new_keys);
    cudaFree (new_values);

    return results;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	if (total_size == 0) {
		return 0.19f;
	}

	return (float)((float)((float)total_size - (float)free_size) / (float)total_size);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
