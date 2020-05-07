#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0
#define THREADS_NUMBER  100

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)

__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 5480398654009llu) % 1436653624766633509llu % limit;
}
__device__ int hash2(int data, int limit) {
	return ((long)abs(data) * 13809739252051llu) % 2873307249533267101llu % limit;
}
__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * 34798362354533llu) % 5746614499066534157llu % limit;
}

// pereche cheie - valoare
struct key_value_pair {
	int32_t key;
	int32_t value;
};

//
// GPU HashTable
//
class GpuHashTable
{
	public:
		int total_size;
		int free_size;

		key_value_pair *bucket_1;
		key_value_pair *bucket_2;

		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif

