// Sycon version-2 bitwise implementation that produces the incorrect test vectors.
// We provided this to demonstrate the effect of the wrong implementation on the linear layer.
// Created by Aslı Başak Civek

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#define bit64 unsigned __int64

bit64 t[7];
// test vectors:
bit64 state[5] = { 0x0000000000000000, 0x0000000000000000, 0x00000000aaaaaaaa, 0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa };

// sycon constant
bit64 constant[12] = { 0x05aaaaaaaaaaaaaa,0x0aaaaaaaaaaaaaaa,0x0daaaaaaaaaaaaaa,0x0eaaaaaaaaaaaaaa,0x0faaaaaaaaaaaaaa,0x07aaaaaaaaaaaaaa,0x03aaaaaaaaaaaaaa,0x01aaaaaaaaaaaaaa,0x08aaaaaaaaaaaaaa,0x04aaaaaaaaaaaaaa,0x02aaaaaaaaaaaaaa,0x09aaaaaaaaaaaaaa };

void print_state(bit64 state[5]) {
    int i;
    for (i = 0; i < 5; i++) printf("%016llx\n", state[i]);
    printf("\n");
}

bit64 r_rotate(bit64 x, int l) {
    return (x >> l) | (x << (64 - l));
}

bit64 rotater(bit64 x, int i, int j, int k) {
    bit64 temp0, temp1, temp2;

    temp0 = r_rotate(x, i);
    temp1 = r_rotate(x, j);
    temp2 = r_rotate(x, k);
    x = temp0 ^ temp1 ^ temp2;
    return x;
}

bit64 x0(bit64 x) {
    bit64 a, b, c;
    a = x & 0x3838383838383838;
    a = rotater(a, 24, 35, 30);

    b = x & 0xc0c0c0c0c0c0c0c0;
    b = rotater(b, 24, 35, 46);

    c = x & 0x0707070707070707;
    c = rotater(c, 24, 19, 30);

    x = a ^ b ^ c;
    return x;
}

bit64 x1(bit64 x) {
    bit64 a, b, c;
    a = x & 0x8080808080808080;
    a = rotater(a, 32, 55, 62);

    b = x & 0x4040404040404040;
    b = rotater(b, 32, 39, 62);

    c = x & 0x3f3f3f3f3f3f3f3f;
    c = rotater(c, 32, 39, 46);

    x = a ^ b ^ c;
    return x;
}

bit64 x2(bit64 x) {
    bit64 a, b, c;
    a = x & 0xfcfcfcfcfcfcfcfc;
    a = rotater(a, 17, 48, 50);

    b = x & 0x0202020202020202;
    b = rotater(b, 17, 48, 34);

    c = x & 0x0101010101010101;
    c = rotater(c, 1, 48, 34);

    x = a ^ b ^ c;
    return x;
}

bit64 x3(bit64 x) {
    bit64 a, b, c;
    a = x & 0xe0e0e0e0e0e0e0e0;
    a = rotater(a, 8, 34, 61);

    b = x & 0x1c1c1c1c1c1c1c1c;
    b = rotater(b, 8, 34, 45);

    c = x & 0x0303030303030303;
    c = rotater(c, 8, 18, 45);

    x = a ^ b ^ c;
    return x;
}

bit64 x4(bit64 x) {
    bit64 a, b, c;
    a = x & 0xe0e0e0e0e0e0e0e0;
    a = rotater(a, 0, 42, 61);

    b = x & 0x1c1c1c1c1c1c1c1c;
    b = rotater(b, 0, 42, 45);

    c = x & 0x0303030303030303;
    c = rotater(c, 0, 26, 45);

    x = a ^ b ^ c;
    return x;
}


// Sycon SBox Layer
void substitution(bit64 x[5]) {
    t[0] = x[2] ^ x[4]; t[1] = t[0] ^ x[1]; t[2] = x[1] ^ x[3]; t[3] = x[0] ^ x[4]; t[4] = t[1] & x[3];
    t[5] = t[3] ^ t[4]; x[1] = ~x[1]; x[1] = x[1] & x[3]; t[6] = ~t[2]; t[6] = t[6] & x[0];
    x[1] = x[1] ^ t[1]; x[1] = x[1] ^ t[6]; t[3] = ~t[3]; t[6] = t[3] & x[2]; t[1] = t[6] ^ t[2];
    t[0] = ~t[0]; x[3] = t[0] & x[3]; x[3] = x[3] ^ x[0]; x[2] = ~x[2]; x[3] = x[3] ^ x[2];
    x[4] = ~x[4]; x[4] = x[4] & x[0]; x[4] = x[4] ^ t[2]; x[0] = t[5]; x[2] = t[1];
}

// Sycon SubBlockDiffusion Layer
void sycon_linear_layer(bit64 state[5]) {
    state[0] = x0(state[0]);
    state[1] = x1(state[1]);
    state[2] = x2(state[2]);
    state[3] = x3(state[3]);
    state[4] = x4(state[4]);
}

// Sycon Permutation
void permutation(bit64 state[5], int round) {
    int i;
    printf("Round: 0 STATE:\n");
    print_state(state);
    for (i = 0; i < round; i++) {
        // SBox (SB), SubBlockDiffusion (SD), AddRound-Const(RC)
        substitution(state);
        printf("Round: %d SBOX:\n", i+1);
        print_state(state);

        sycon_linear_layer(state);
        printf("Round: %d LINEAR\n", i+1);
        print_state(state);

        //add round constant 
        state[2] = state[2] ^ constant[i];
        printf("Round: %d ADD CONSTANT\n", i+1);
        print_state(state);
    }
}

int main() {
    permutation(state, 12);
    return 0;

}