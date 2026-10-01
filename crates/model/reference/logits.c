// Reference logits from upstream llama2.c run.c (karpathy/llama2.c @ 350e04f).
// Calls run.c's own build_transformer and forward; prints each logit's
// float bits so expected values are exact.
//
//   gcc -O2 -I<llama2.c checkout> -o logits logits.c -lm
//   ./logits checkpoint.bin TOKEN...
#define TESTING
#include "run.c"
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: logits checkpoint token...\n"); return 2; }
    Transformer t;
    build_transformer(&t, argv[1]);
    for (int pos = 0; pos + 2 < argc; pos++) {
        float *logits = forward(&t, atoi(argv[pos + 2]), pos);
        printf("%d:", pos);
        for (int i = 0; i < t.config.vocab_size; i++) {
            uint32_t bits;
            memcpy(&bits, &logits[i], 4);
            printf(" 0x%08x", bits);
        }
        printf("\n");
    }
    free_transformer(&t);
    return 0;
}
