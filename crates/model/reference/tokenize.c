// Reference tokenization from upstream llama2.c run.c (karpathy/llama2.c @ 350e04f).
// Calls run.c's own build_tokenizer, encode (with BOS, without EOS) and decode.
//
//   gcc -O2 -I<llama2.c checkout> -o tokenize tokenize.c -lm
//   ./tokenize tokenizer.bin VOCAB_SIZE HEXTEXT...     encode each hex-encoded string
//   ./tokenize tokenizer.bin VOCAB_SIZE d:PREV:TOKEN... decode, printing the bytes as hex
#define TESTING
#include "run.c"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: tokenize tokenizer.bin vocab_size arg...\n"); return 2; }
    Tokenizer t;
    build_tokenizer(&t, argv[1], atoi(argv[2]));
    for (int a = 3; a < argc; a++) {
        if (argv[a][0] == 'd' && argv[a][1] == ':') {
            int prev, token;
            sscanf(argv[a] + 2, "%d:%d", &prev, &token);
            char *piece = decode(&t, prev, token);
            printf("%s:", argv[a]);
            for (char *c = piece; *c; c++) printf(" %02x", (unsigned char)*c);
            printf("\n");
            continue;
        }
        size_t n = strlen(argv[a]) / 2;
        char *text = malloc(n + 1);
        for (size_t i = 0; i < n; i++) { unsigned v; sscanf(argv[a] + 2 * i, "%2x", &v); text[i] = (char)v; }
        text[n] = '\0';
        int *tokens = malloc((n + 3) * sizeof(int));
        int count;
        encode(&t, text, 1, 0, tokens, &count);
        printf("%s:", argv[a]);
        for (int i = 0; i < count; i++) printf(" %d", tokens[i]);
        printf("\n");
        free(tokens);
        free(text);
    }
    free_tokenizer(&t);
    return 0;
}
