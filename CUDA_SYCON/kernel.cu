// Created by Aslı Başak CİVEK by modifying the Ascon CUDA codes of Cihangir TEZCAN: https://github.com/cihangirtezcan/CUDA_ASCON
// Windows version

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include "rdrand.h"
#include <intrin.h>
#include <immintrin.h>
#include <string.h>

#define bit64 unsigned __int64
#define RDRAND_MASK 0x40000000
#define RETRY_LIMIT 10


// for random number generation //
int RdRand_cpuid() {
	int info[4] = { -1, -1, -1, -1 };
	__cpuid(info, 0);
	if (memcmp((void*)&info[1], (void*)"Genu", 4) != 0 ||
		memcmp((void*)&info[3], (void*)"ineI", 4) != 0 ||
		memcmp((void*)&info[2], (void*)"ntel", 4) != 0) {
		return 0;
	}
	__cpuid(info, 1);
	int ecx = info[2];
	if ((ecx & RDRAND_MASK) == RDRAND_MASK)
		return 1;
	else
		return 0;
}

int RdRand_isSupported() {
	static int supported = RDRAND_SUPPORT_UNKNOWN;
	if (supported == RDRAND_SUPPORT_UNKNOWN) {
		if (RdRand_cpuid())
			supported = RDRAND_SUPPORTED;
		else
			supported = RDRAND_UNSUPPORTED;
	}
	return (supported == RDRAND_SUPPORTED) ? 1 : 0;
}

int rdrand_64(uint64_t* x, int retry) {
	if (RdRand_isSupported()) {
		if (retry) {
			for (int i = 0; i < RETRY_LIMIT; i++) {
				if (_rdrand64_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else {
			if (_rdrand64_step(x))
				return RDRAND_SUCCESS;
			else
				return RDRAND_NOT_READY;
		}
	}
	else {
		return RDRAND_UNSUPPORTED;
	}
}

// for random number generation //

double PCFreq = 0.0;
__int64 CounterStart = 0;
void StartCounter() {
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		printf("QueryPerformanceFrequency failed!\n");

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

// #######################  SYCON EXPERIMENT - CPU CODE ######################
bit64 t[7];
bit64 state[5] = { 0x00000000000000001,0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
bit64 state2[5] = { 0 }; // pair
// to verify the test vectors:
bit64 test_state[5] = { 0x0000000000000000, 0x0000000000000000, 0x00000000aaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa };
// sycon constant
bit64 constant[12] = { 0x05aaaaaaaaaaaaaa,0x0aaaaaaaaaaaaaaa,0x0daaaaaaaaaaaaaa,0x0eaaaaaaaaaaaaaa,0x0faaaaaaaaaaaaaa,0x07aaaaaaaaaaaaaa,0x03aaaaaaaaaaaaaa,0x01aaaaaaaaaaaaaa,0x08aaaaaaaaaaaaaa,0x04aaaaaaaaaaaaaa,0x02aaaaaaaaaaaaaa,0x09aaaaaaaaaaaaaa };

void print_state(bit64 state[5]) {
	int i;
	for (i = 0; i < 5; i++) printf("%016llx\n", state[i]);
	printf("\n");
}

// SYCON SBOX
void substitution(bit64 x[5]) { // x0 as lsb
	t[0] = x[2] ^ x[4]; t[1] = t[0] ^ x[1]; t[2] = x[1] ^ x[3]; t[3] = x[0] ^ x[4]; t[4] = t[1] & x[3];
	t[5] = t[3] ^ t[4]; x[1] = ~x[1]; x[1] = x[1] & x[3]; t[6] = ~t[2]; t[6] = t[6] & x[0];
	x[1] = x[1] ^ t[1]; x[1] = x[1] ^ t[6]; t[3] = ~t[3]; t[6] = t[3] & x[2]; t[1] = t[6] ^ t[2];
	t[0] = ~t[0]; x[3] = t[0] & x[3]; x[3] = x[3] ^ x[0]; x[2] = ~x[2]; x[3] = x[3] ^ x[2];
	x[4] = ~x[4]; x[4] = x[4] & x[0]; x[4] = x[4] ^ t[2]; x[0] = t[5]; x[2] = t[1];

}

bit64 l_rotate(bit64 x, int l) {
	bit64 temp;
	temp = (x << l) ^ (x >> (64 - l));
	return temp;

}

void sycon_linear_layer(bit64 state[5]) {
	bit64 temp0, temp1, temp2;

	temp0 = l_rotate(state[0], 59);
	temp1 = l_rotate(state[0], 54);
	temp2 = state[0] ^ temp0 ^ temp1;
	state[0] = l_rotate(temp2, 40);

	temp0 = l_rotate(state[1], 55);
	temp1 = l_rotate(state[1], 46);
	temp2 = state[1] ^ temp0 ^ temp1;
	state[1] = l_rotate(temp2, 32);

	temp0 = l_rotate(state[2], 33);
	temp1 = l_rotate(state[2], 2);
	temp2 = state[2] ^ temp0 ^ temp1;
	state[2] = l_rotate(temp2, 16);

	temp0 = l_rotate(state[3], 21);
	temp1 = l_rotate(state[3], 42);
	temp2 = state[3] ^ temp0 ^ temp1;
	state[3] = l_rotate(temp2, 56);

	temp0 = l_rotate(state[4], 13);
	temp1 = l_rotate(state[4], 26);
	state[4] ^= temp0 ^ temp1;

}
void permutation(bit64 state[5], int round) {
	int i;
	for (i = 0; i < round; i++) {
		// SBox (SB), SubBlockDiffusion (SD), AddRound-Const(RC)
		substitution(state);
		sycon_linear_layer(state);
		//add round constant
		//state[2] = state[2] ^ constant[i]; // negligible for the experiment
	}
}

// to verify the test vectors
void test_permutation(bit64 state[5], int round) {
	int i;
	printf("Round: 0 STATE:\n");
	print_state(state);
	for (i = 0; i < round; i++) {
		// SBox (SB), SubBlockDiffusion (SD), AddRound-Const(RC)

		substitution(state);
		printf("Round: %d SBOX:\n", i + 1);
		print_state(state);

		sycon_linear_layer(state);
		printf("Round: %d LINEAR\n", i + 1);
		print_state(state);

		//add round constant 
		state[2] = state[2] ^ constant[i];
		printf("Round: %d ADD CONSTANT\n", i + 1);
		print_state(state);
	}
}

// hex xor calculation
int parity(unsigned long long v) { // 64-bit word 
	int a;
	v ^= v >> 1;
	v ^= v >> 2;
	v = (v & 0x1111111111111111UL) * 0x1111111111111111UL;
	a = (v >> 60) & 1;
	return a; //Parity of Xi
}

// it uses the permutation of the first state as a random state
int experiment(bit64 state[5], int round) {

	//flip some bits to get state2
	state2[0] = state[0];
	state2[1] = state[1] ^ 0x0000000000800000; // should be from the successful experiment
	state2[2] = state[2];
	state2[3] = state[3] ^ 0x0000000000800000; // should be from the successful experiment
	state2[4] = state[4];

	permutation(state, round);
	permutation(state2, round);

	// change it when needed:
	// 2r type-II linear approx - for 4 round experiment:
	//return parity((state[0] ^ state2[0]) & 0x66EEECDDD9BBB377);

	// 2r type-I linear approx - for 4 round experiment:
	//return parity(((state[1] ^ state2[1]) & 0xBB6ED9B76DDB76CD) ^ ((state[3] ^ state2[3]) & 0xFF000007FFFFFFFF) ^ ((state[4] ^ state2[4]) & 0xB6D6DB5B6F6DADB7));
	// approxes: 0x0000000004000000 for x0, x1, x2

	// 3r type-II linear approx - for 5 round experiment:
	//return parity((state[0] ^ state2[0]) & 0xB37766EECDDDD9BB);

	// 3r typeI for 5-round experiment
	return parity(((state[0] ^ state2[0]) & 0x576eaedd55faabb7) ^ ((state[1] ^ state2[1]) & 0x6c1b168d8362d9b4) ^ ((state[3] ^ state2[3]) & 0x001ffffffffff800));
}

bit64 cpu_single_experiment(int round, bit64 experiment_size) {
	bit64 a, s, counter;
	counter = 0;
	bit64 bias;

	// Run the experiment 1024*1024*32*trial times for r round 
	for (s = 0; s < experiment_size; s++) {
		a = experiment(state, round);
		if (a == 0) { counter = counter + 1; }
	}

	bias = (experiment_size) / 2 - counter;
	printf("Size: %lld, Counter: %lld, Bias: %lld, Time: %u seconds\n", s, counter, bias, clock() / CLOCKS_PER_SEC);
	return bias;
}


// #######################  SYCON EXPERIMENT - GPU CODE ######################
#define BLOCKS 32
#define THREADS 1024
#define TRIALS	1024 //*1024
__int64 trial = 1, trial_i = 0, repeat = 1; //should be 10 for calculating the average.
bit64* nonce, * nonce_d;

// device functions for rotation
__device__ bit64 l_rotate_d(bit64 x, int l) {
	bit64 temp;
	temp = (x << l) ^ (x >> (64 - l));
	return temp;

}

__global__ void gpu_single_experiment(bit64 nonce[], __int64 counter[], int round) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit64 initial0, initial1, initial2, initial3, initial4;
	bit64 pair0, pair1, pair2, pair3, pair4;
	bit64 t0, t1, t2, t3, t4, t5, t6;

	initial0 = 0x0000000000000000; // key 
	initial1 = 0x0000000000000000; // key
	initial2 = nonce[2 * threadIndex]; // nonce
	initial3 = nonce[2 * threadIndex + 1]; // nonce
	initial4 = 0x0000000000000000; // iv for sycon64-aead

	for (int c = 0; c < TRIALS; c++) {

		// for the 4-round experiment, change it to: 
		/*
		pair0 = initial0 ^ 0x0010000000000000;
		pair1 = initial1;
		pair2 = initial2 ^ 0x0010000000000000;
		pair3 = initial3;
		pair4 = initial4 ^ 0x0010000000000000;
		*/

		// for the 5-round experiment, change it to: 
		pair0 = initial0;
		pair1 = initial1 ^ 0x0000000000800000; // should be from the successful experiment
		pair2 = initial2;
		pair3 = initial3 ^ 0x0000000000800000; // should be from the successful experiment
		pair4 = initial4;

		for (int i = 0; i < round; i++) {

			// Sycon Sbox // initial
			t0 = initial2 ^ initial4; t1 = t0 ^ initial1; t2 = initial1 ^ initial3; t3 = initial0 ^ initial4; t4 = t1 & initial3;
			t5 = t3 ^ t4; initial1 = ~initial1; initial1 = initial1 & initial3; t6 = ~t2; t6 = t6 & initial0;
			initial1 = initial1 ^ t1; initial1 = initial1 ^ t6; t3 = ~t3; t6 = t3 & initial2; t1 = t6 ^ t2;
			t0 = ~t0; initial3 = t0 & initial3; initial3 = initial3 ^ initial0; initial2 = ~initial2; initial3 = initial3 ^ initial2;
			initial4 = ~initial4; initial4 = initial4 & initial0; initial4 = initial4 ^ t2; initial0 = t5; initial2 = t1;

			// Sycon Liner layer // initial
			initial0 = l_rotate_d(initial0 ^ l_rotate_d(initial0, 59) ^ l_rotate_d(initial0, 54), 40);
			initial1 = l_rotate_d(initial1 ^ l_rotate_d(initial1, 55) ^ l_rotate_d(initial1, 46), 32);
			initial2 = l_rotate_d(initial2 ^ l_rotate_d(initial2, 33) ^ l_rotate_d(initial2, 2), 16);
			initial3 = l_rotate_d(initial3 ^ l_rotate_d(initial3, 21) ^ l_rotate_d(initial3, 42), 56);
			initial4 = initial4 ^ l_rotate_d(initial4, 13) ^ l_rotate_d(initial4, 26);
		}

		for (int i = 0; i < round; i++) {
			// Sycon sbox // pair
			t0 = pair2 ^ pair4; t1 = t0 ^ pair1; t2 = pair1 ^ pair3; t3 = pair0 ^ pair4; t4 = t1 & pair3;
			t5 = t3 ^ t4; pair1 = ~pair1; pair1 = pair1 & pair3; t6 = ~t2; t6 = t6 & pair0;
			pair1 = pair1 ^ t1; pair1 = pair1 ^ t6; t3 = ~t3; t6 = t3 & pair2; t1 = t6 ^ t2;
			t0 = ~t0; pair3 = t0 & pair3; pair3 = pair3 ^ pair0; pair2 = ~pair2; pair3 = pair3 ^ pair2;
			pair4 = ~pair4; pair4 = pair4 & pair0; pair4 = pair4 ^ t2; pair0 = t5; pair2 = t1;

			// Liner layer // pair
			pair0 = l_rotate_d(pair0 ^ l_rotate_d(pair0, 59) ^ l_rotate_d(pair0, 54), 40);
			pair1 = l_rotate_d(pair1 ^ l_rotate_d(pair1, 55) ^ l_rotate_d(pair1, 46), 32);
			pair2 = l_rotate_d(pair2 ^ l_rotate_d(pair2, 33) ^ l_rotate_d(pair2, 2), 16);
			pair3 = l_rotate_d(pair3 ^ l_rotate_d(pair3, 21) ^ l_rotate_d(pair3, 42), 56);
			pair4 = pair4 ^ l_rotate_d(pair4, 13) ^ l_rotate_d(pair4, 26);
		}

		// for the 4-round experiment (type-II), change it to:
		/*
		t1 = 0;
		t0 = initial0 & 0x66EEECDDD9BBB377; //2 round type-II distinguisher for 4 rounds
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);

		t0 = pair0 & 0x66EEECDDD9BBB377; //2 round type-II distinguisher for 4 round
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);

		if (t1 == 0) counter[threadIndex]++;
		*/

		// for the 5-round experiment, change it to (TYPE-I):

		t1 = 0;
		t0 = initial0 & 0x576eaedd55faabb7;
		t0 = t0 ^ (pair0 & 0x576eaedd55faabb7);
		for (int i = 0; i < 64; i++) t1 ^= (t0 >> i);

		t0 = initial1 & 0x6c1b168d8362d9b4;
		t0 = t0 ^ (pair1 & 0x6c1b168d8362d9b4);
		for (int i = 0; i < 64; i++) t1 ^= (t0 >> i);

		t0 = initial3 & 0x001ffffffffff800;
		t0 = t0 ^ (pair3 & 0x001ffffffffff800);
		for (int i = 0; i < 64; i++) t1 ^= (t0 >> i);

		if ((t1 & 0x1) == 0) counter[threadIndex]++;


		// for the 5-round experiment, change it to (TYPE-II):
		/*
		t1 = 0;
		t0 = initial0 & 0xB37766EECDDDD9BB; //2 round type-II distinguisher for 4 rounds
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);

		t0 = pair0 & 0xB37766EECDDDD9BB; //2 round type-II distinguisher for 4 round
		for (int i = 0; i < 64; i++) t1 ^= ((t0 >> i) & 0x1);

		if (t1 == 0) counter[threadIndex]++;
		*/
	}
}

__global__ void gpu_rotate(bit64 key[], bit64 nonce[], int key_choice, __int64 counter[], int rotation, int round) {
	// x0-x1: key, x2-x3: nonce, x4: IV
	// IV: Sycon-AEAD-96: 0x5980A92AFC5D9D2C
	// IV: Sycon-AEAD-64: 0x0000000000000000

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit64 initial0, initial1, initial2, initial3, initial4 = 0x0000000000000000;
	bit64 pair0, pair1, pair2, pair3, pair4 = 0x0000000000000000;
	bit64 t0, t1, t2, t3, t4, t5, t6;

	initial2 = nonce[2 * threadIndex]; // nonce
	initial3 = nonce[2 * threadIndex + 1]; // nonce

	for (int c = 0; c < TRIALS; c++) {
		t0 = ((bit64)0x7FFFFFFFFFFFFFFF >> rotation) ^ ((bit64)0x7FFFFFFFFFFFFFFF << (64 - rotation));
		t1 = ((bit64)0x8000000000000000 >> rotation) ^ ((bit64)0x8000000000000000 << (64 - rotation));

		initial4 = 0x0000000000000000; // IV
		pair4 = 0x0000000000000000; // IV

		initial0 = key[0] & t0; if (key_choice == 2 || key_choice == 4) initial0 ^= t1; // key
		initial1 = key[1] & t0; if (key_choice == 3 || key_choice == 4) initial1 ^= t1; // key

		//1= (0, 0), 2= (0, 1), 3= (1, 0), 4= (1, 1)

		// the difference is on the nonce (x2-x3) for key recovery 
		/*
		pair0 = initial0;
		pair1 = initial1;
		pair2 = initial2 ^ t1;
		pair3 = initial3^ t1;
		*/

		// the difference is on the key (x0-x1) for related key attacks 
		/*
		pair0 = initial0^t1;
		pair1 = initial1^ t1;
		pair2 = initial2;
		pair3 = initial3;
		// pair4 = initial4;
		*/

		// the difference is on the nonce (x2-x3) and IV (x4) 

		pair0 = initial0;
		pair1 = initial1;
		pair2 = initial2 ^ t1;
		pair3 = initial3 ^ t1;
		pair4 = pair4 ^ t1;


		for (int i = 0; i < round; i++) {

			// Sycon Sbox // initial
			t0 = initial2 ^ initial4; t1 = t0 ^ initial1; t2 = initial1 ^ initial3; t3 = initial0 ^ initial4; t4 = t1 & initial3;
			t5 = t3 ^ t4; initial1 = ~initial1; initial1 = initial1 & initial3; t6 = ~t2; t6 = t6 & initial0;
			initial1 = initial1 ^ t1; initial1 = initial1 ^ t6; t3 = ~t3; t6 = t3 & initial2; t1 = t6 ^ t2;
			t0 = ~t0; initial3 = t0 & initial3; initial3 = initial3 ^ initial0; initial2 = ~initial2; initial3 = initial3 ^ initial2;
			initial4 = ~initial4; initial4 = initial4 & initial0; initial4 = initial4 ^ t2; initial0 = t5; initial2 = t1;

			// Sycon Liner layer // initial

			initial0 = l_rotate_d(initial0 ^ l_rotate_d(initial0, 59) ^ l_rotate_d(initial0, 54), 40);
			initial1 = l_rotate_d(initial1 ^ l_rotate_d(initial1, 55) ^ l_rotate_d(initial1, 46), 32);
			initial2 = l_rotate_d(initial2 ^ l_rotate_d(initial2, 33) ^ l_rotate_d(initial2, 2), 16);
			initial3 = l_rotate_d(initial3 ^ l_rotate_d(initial3, 21) ^ l_rotate_d(initial3, 42), 56);
			initial4 = initial4 ^ l_rotate_d(initial4, 13) ^ l_rotate_d(initial4, 26);
		}

		for (int i = 0; i < round; i++) {
			// Sycon sbox // pair
			t0 = pair2 ^ pair4; t1 = t0 ^ pair1; t2 = pair1 ^ pair3; t3 = pair0 ^ pair4; t4 = t1 & pair3;
			t5 = t3 ^ t4; pair1 = ~pair1; pair1 = pair1 & pair3; t6 = ~t2; t6 = t6 & pair0;
			pair1 = pair1 ^ t1; pair1 = pair1 ^ t6; t3 = ~t3; t6 = t3 & pair2; t1 = t6 ^ t2;
			t0 = ~t0; pair3 = t0 & pair3; pair3 = pair3 ^ pair0; pair2 = ~pair2; pair3 = pair3 ^ pair2;
			pair4 = ~pair4; pair4 = pair4 & pair0; pair4 = pair4 ^ t2; pair0 = t5; pair2 = t1;

			// Liner layer // pair
			pair0 = l_rotate_d(pair0 ^ l_rotate_d(pair0, 59) ^ l_rotate_d(pair0, 54), 40);
			pair1 = l_rotate_d(pair1 ^ l_rotate_d(pair1, 55) ^ l_rotate_d(pair1, 46), 32);
			pair2 = l_rotate_d(pair2 ^ l_rotate_d(pair2, 33) ^ l_rotate_d(pair2, 2), 16);
			pair3 = l_rotate_d(pair3 ^ l_rotate_d(pair3, 21) ^ l_rotate_d(pair3, 42), 56);
			pair4 = pair4 ^ l_rotate_d(pair4, 13) ^ l_rotate_d(pair4, 26);

		}
		// type-II - 3r
		t1 = 0;
		t0 = initial0 & 0xB37766EECDDDD9BB;
		t0 = t0 ^ (pair0 & 0xB37766EECDDDD9BB);
		for (int i = 0; i < 64; i++) t1 ^= (t0 >> i);
		if ((t1 & 0x1) == 0) counter[threadIndex]++;

		// type-II - 2r
		/*
		t1 = 0;
		t0 = initial0 & 0x66EEECDDD9BBB377;
		t0 = t0 ^ (pair0 & 0x66EEECDDD9BBB377);
		for (int i = 0; i < 64; i++) t1 ^= (t0 >> i);
		if ((t1 & 0x1) == 0) counter[threadIndex]++;
		*/

		initial2 += initial0;
		initial3 += initial1;
		// nonce    // key
	}
}

// #######################  SYCON EXPERIMENT - MAIN ######################

void show_menu() {
	printf(">>> SYCON Distinguisher Finder <<<\n\n"
		"(0) Test Vectors\n"
		"(1) CPU version\n"
		"(2) GPU version\n"
		"(3) GPU rotation version\n"
		"(4) Clear screen\n"
		"(5) Exit\n\n"
		"Choice: ");
}

void main() {
	int round = 0;
	bit64 experiment_size;
	int choice = 0;
	nonce = (bit64*)calloc(BLOCKS * THREADS * 2, sizeof(bit64));

	while (1) {

		show_menu();
		scanf_s("%d", &choice);

		if (choice == 0) { // (0) Test Vectors
			// to verify the test vectors
			test_permutation(test_state, 12);
		}

		if (choice == 1) { // (1) CPU version

			printf("Trial = 2^25 +  ");
			scanf_s("%I64d", &trial_i);
			trial = pow(2, trial_i);

			printf("For how many rounds: ");
			scanf_s("%d", &round);


			experiment_size = 1024 * 1024 * 32 * trial;
			printf("Running the experiment with %lld (2** %lld) data\n", experiment_size, trial_i + 25);
			bit64 ten_total = 0;
			for (int t = 0; t < repeat; t++) {
				ten_total += cpu_single_experiment(round, experiment_size);
			}
			printf("\nAverage bias: %lld\n", llabs(ten_total) / repeat);
		}

		if (choice == 2) { // (2) GPU version


			printf("Trial = 2^25 +  ");
			scanf_s("%I64d", &trial_i);
			trial = pow(2, trial_i);

			printf("For how many rounds: ");
			scanf_s("%d", &round);

			__int64* counter_d, * counter, total_counter = 0, bias, average_bias = 0;
			bit64 total_bias;
			// thread 32, block 1024, trials 1024 = 2**25 
			experiment_size = trial * TRIALS * THREADS * BLOCKS; // trial * 2**25
			printf("Running the experiment with %lld (2** %lld) data\n", experiment_size, trial_i + 25);

			for (int m = 0; m < repeat; m++) { // same experiment for "repeat" times
				counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
				total_counter = 0;
				cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
				cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
				cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);


				// Create cuda events for measuring the time
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				// Start the timer
				cudaEventRecord(start, 0);

				for (int i = 0; i < trial; i++) {
					for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); }
					cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
					cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
					cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
					gpu_single_experiment << <BLOCKS, THREADS >> > (nonce_d, counter_d, round);

				}

				// stop the timer
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);

				// calculate the elapsed time
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);


				cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
				for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
				bias = (experiment_size) / 2 - total_counter;
				printf("%03d: Total counter: %I64d Bias: %I64d Elapsed Time: %f second\n", m, total_counter, bias, elapsedTime / 1000.0f);
				total_bias = total_bias + llabs(bias);

				// destroy the events and free memory
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				cudaFree(nonce_d); cudaFree(counter_d);
			}
			printf("\nAverage Bias: %I64d \n", total_bias / repeat);

		}

		if (choice == 3) { // (3) GPU rotation version
			FILE* fp;
			int shift = 0, flag = 0, key_choice = 0;
			__int64* counter, * counter_d, total_counter = 0, bias, average_bias = 0, experiment;
			__int64 trial = 1, keys = 1; //10
			bit64 key[2], * key_d;
			printf("For how many rounds: ");
			scanf_s("%d", &round);

			printf("Pairs (2 ^ 25 + ?): ");
			scanf_s("%d", &shift);

			printf("Select key (1-4): ");
			scanf_s("%d", &key_choice);

			trial = 1;		trial = trial << shift;
			experiment = trial * TRIALS * THREADS * BLOCKS; // trial * 2**25
			if (key_choice == 1) fopen_s(&fp, "Automatic_search_key1.txt", "w");
			if (key_choice == 2) fopen_s(&fp, "Automatic_search_key2.txt", "w");
			if (key_choice == 3) fopen_s(&fp, "Automatic_search_key3.txt", "w");
			if (key_choice == 4) fopen_s(&fp, "Automatic_search_key4.txt", "w");
			printf("Key Choice: %d\n", key_choice); fprintf(fp, "Key Choice: %d\n", key_choice);
			printf("Pairs: 2 ^ %d\n", shift + 25); fprintf(fp, "Pairs: 2 ^ %d\n", shift + 25);
			printf("Experiment: %I64d\n", experiment); fprintf(fp, "Experiment: %I64d\n", experiment);
			for (int rotation = 0; rotation < 64; rotation++) {
				total_counter = 0; bias = 0; average_bias = 0; flag = 0;
				printf("Rotation: %d\n", rotation); fprintf(fp, "Rotation: %d\n", rotation);
				for (int m = 0; m < keys; m++) {
					counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
					total_counter = 0;
					cudaMalloc((void**)&key_d, 2 * sizeof(bit64));
					cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 2 * sizeof(bit64));
					cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
					rdrand_64(key, 0);
					rdrand_64(key + 1, 0);
					StartCounter();
					cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
					for (int i = 0; i < trial; i++) {
						for (int j = 0; j < THREADS * BLOCKS * 2; j++) { rdrand_64(nonce + j, 0); }
						cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
						cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
						cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 2 * sizeof(bit64), cudaMemcpyHostToDevice);
						cudaMemcpy(key_d, key, 2 * sizeof(bit64), cudaMemcpyHostToDevice);
						gpu_rotate << <BLOCKS, THREADS >> > (key_d, nonce_d, key_choice, counter_d, rotation, round);

					}
					cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
					for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
					bias = (experiment) / 2 - total_counter;
					printf("%03d: Total counter: %I64d Bias: %I64d\n", m, total_counter, bias);
					fprintf(fp, "%03d: Total counter: %I64d Bias: %I64d\n", m, total_counter, bias);
					average_bias += bias;
					cudaFree(key_d); cudaFree(nonce_d); cudaFree(counter_d);
					if (bias > 0 && flag < 0) m = keys + 1;
					else if (bias < 0 && flag > 0) m = keys + 1;
					if (bias > 0) flag = 1;
					else if (bias < 0) flag = -11;
				}
				average_bias /= keys;
				printf("Average bias: %I64d\n", llabs(average_bias)); fprintf(fp, "Average bias: %I64d\n", average_bias);
			}
			fclose(fp);
		}
		if (choice == 4) {
#ifdef _WIN32
			system("cls");
#else
			system("clear");
#endif
		}
		if (choice == 5) {
			printf("Exiting the program...\n");
			exit(0);
		}
	}
	system("PAUSE");
}