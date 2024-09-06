/* Implementation of Sycon-2 permutation
   Adapted from Sycon V1.0 implementation of Kalikinkar Mandal <kmandal@uwaterloo.ca>
   Adapted by: Aslı Başak Civek

   We provided this for POC purposes. Right bit-rotation produces the incorrect test vectors, 
   while left bit-rotation produces the correct test vectors.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define NUMROUNDS 12 //Number of Rounds 
#define STATEBYTES 40 //SIZE of state in bytes


static const unsigned char RC[12] = { 0x05,0x0a,0x0d,0x0e,0x0f,0x07,0x03,0x01,0x08,0x04,0x02,0x09 };

// Right bit-rotation that produces incorrect test vectors
unsigned char rotl8(const unsigned char x, const unsigned char y, const unsigned char shift)
{
    return ((x >> shift) | (y << (8 - shift)));
}

// Left bit-rotation that produces correct test vectors
/*
unsigned char rotl8(const unsigned char x, const unsigned char y, const unsigned char shift)
{
    return ((x << shift) | (y >> (8 - shift)));
}
*/

/***********************************************************
  ******* Sycon permutation implementation *****************
  *********************************************************/

void print_sycon_state(const unsigned char* state)
{
    unsigned char i;
    for (i = 0; i < STATEBYTES; i++)
    {
        printf("%02X", state[i]);
    }
    printf("\n");
}

void left_cyclic_shift64(unsigned char* tmp, unsigned char rot_const)
{
    unsigned char i, u, v, t[8];
    u = rot_const / 8;
    v = rot_const % 8;

    for (i = 0; i < 8; i++)
        t[i] = tmp[(i + u) % 8];
    if (v != 0)
    {
        for (i = 0; i < 8; i++)
            tmp[i] = rotl8(t[i], t[(i + 1) % 8], v);
    }
    else
    {
        for (i = 0; i < 8; i++)
            tmp[i] = t[i];
    }

}

void sycon_perm(unsigned char* input)
{
    unsigned char i, j;
    unsigned char t[5][8];

    for (i = 0; i < NUMROUNDS; i++)
    {
        printf("\nSBOX LAYER - Round %d\n", i+1);
        //Sbox layer
        for (j = 0; j < 8; j++)
            t[0][j] = input[8 * 2 + j] ^ input[8 * 4 + j];
        for (j = 0; j < 8; j++)
            t[1][j] = t[0][j] ^ input[8 * 1 + j];
        for (j = 0; j < 8; j++)
            t[2][j] = input[8 * 1 + j] ^ input[8 * 3 + j];
        for (j = 0; j < 8; j++)
            t[3][j] = input[j] ^ input[8 * 4 + j];
        for (j = 0; j < 8; j++)
            t[4][j] = t[3][j] ^ (t[1][j] & input[8 * 3 + j]);
        for (j = 0; j < 8; j++)
            input[8 * 1 + j] = ((~input[8 * 1 + j]) & input[8 * 3 + j]) ^ t[1][j] ^ ((~t[2][j]) & input[j]);
        for (j = 0; j < 8; j++)
            t[1][j] = ((~t[3][j]) & input[8 * 2 + j]) ^ t[2][j];
        for (j = 0; j < 8; j++)
            input[8 * 3 + j] = ((~t[0][j]) & input[8 * 3 + j]) ^ input[j] ^ (~input[8 * 2 + j]);
        for (j = 0; j < 8; j++)
            input[8 * 4 + j] = ((~input[8 * 4 + j]) & input[j]) ^ t[2][j];
        for (j = 0; j < 8; j++)
            input[j] = t[4][j];
        for (j = 0; j < 8; j++)
            input[8 * 2 + j] = t[1][j];

        print_sycon_state(input);
        printf("\nDIFFUSION LAYER - Round %d\n", i + 1);
        //SubBlockDiffusion Layer

        // x0
        for (j = 0; j < 8; j++)
            t[0][j] = input[j];
        for (j = 0; j < 8; j++)
            t[1][j] = input[j];
        left_cyclic_shift64(t[0], 59);
        left_cyclic_shift64(t[1], 54);

        for (j = 0; j < 8; j++)
            t[2][j] = input[j];
        for (j = 0; j < 8; j++)
            t[2][j] = t[2][j] ^ t[0][j] ^ t[1][j];
        left_cyclic_shift64(t[2], 40);
        for (j = 0; j < 8; j++)
            input[j] = t[2][j];

        // x1
        for (j = 0; j < 8; j++)
            t[0][j] = input[8 * 1 + j];
        for (j = 0; j < 8; j++)
            t[1][j] = input[8 * 1 + j];
        left_cyclic_shift64(t[0], 55);
        left_cyclic_shift64(t[1], 46);

        for (j = 0; j < 8; j++)
            t[2][j] = input[8 * 1 + j];
        for (j = 0; j < 8; j++)
            t[2][j] = t[2][j] ^ t[0][j] ^ t[1][j];
        left_cyclic_shift64(t[2], 32);
        for (j = 0; j < 8; j++)
            input[8 * 1 + j] = t[2][j];

        // x2
        for (j = 0; j < 8; j++)
            t[0][j] = input[8 * 2 + j];
        for (j = 0; j < 8; j++)
            t[1][j] = input[8 * 2 + j];
        left_cyclic_shift64(t[0], 33);
        left_cyclic_shift64(t[1], 2);

        for (j = 0; j < 8; j++)
            t[2][j] = input[8 * 2 + j];
        for (j = 0; j < 8; j++)
            t[2][j] = t[2][j] ^ t[0][j] ^ t[1][j];
        left_cyclic_shift64(t[2], 16);
        for (j = 0; j < 8; j++)
            input[8 * 2 + j] = t[2][j];

        // x3
        for (j = 0; j < 8; j++)
            t[0][j] = input[8 * 3 + j];
        for (j = 0; j < 8; j++)
            t[1][j] = input[8 * 3 + j];
        left_cyclic_shift64(t[0], 21);
        left_cyclic_shift64(t[1], 42);

        for (j = 0; j < 8; j++)
            t[2][j] = input[8 * 3 + j];
        for (j = 0; j < 8; j++)
            t[2][j] = t[2][j] ^ t[0][j] ^ t[1][j];
        left_cyclic_shift64(t[2], 56);
        for (j = 0; j < 8; j++)
            input[8 * 3 + j] = t[2][j];

        // x4
        for (j = 0; j < 8; j++)
            t[0][j] = input[8 * 4 + j];
        for (j = 0; j < 8; j++)
            t[1][j] = input[8 * 4 + j];
        left_cyclic_shift64(t[0], 13);
        left_cyclic_shift64(t[1], 26);

        for (j = 0; j < 8; j++)
            input[8 * 4 + j] = input[8 * 4 + j] ^ t[0][j] ^ t[1][j];

        print_sycon_state(input);
        printf("\nADD ROUND CONSTANT - Round %d\n", i + 1);
        input[8 * 2] = input[8 * 2] ^ RC[i];
        for (j = 1; j < 8; j++)
            input[8 * 2 + j] = input[8 * 2 + j] ^ (0xaa);

        print_sycon_state(input);
    }
    return;
}

int main() {
    unsigned char mystate[40] = { // Test Vectors
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA
    };
    printf("INITIAL STATE\n");
    print_sycon_state(mystate);
    sycon_perm(mystate);
    return 0;
}