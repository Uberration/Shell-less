//! Expected values from the external reference: upstream llama2.c `run.c`
//! (karpathy/llama2.c @ 350e04fe35433e6d2941dce5a1f53308f87058eb, sha256
//! 9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd),
//! compiled with `gcc -O2` (GCC 13.3, x86-64, no -ffast-math, no OpenMP)
//! through the harnesses in `crates/model/reference/`. Generated; see the
//! README there for the procedure. Do not edit by hand.

/// `logits fixture_a.bin 1 7 7 30 2`: float bits per position.
pub(super) const LOGITS_A: &[&[u32]] = &[
    &[
        0x405b950d, 0x407de3e8, 0xc02112e8, 0x3ffa3e50, 0xc09c8813, 0x3eae2a28, 0xc092a8d0, 0x3e85a8bd, 0xbf5ecd9c,
        0xbf9c6f23, 0x40340b08, 0x4054309f, 0xbe8bcb31, 0x3f93afbc, 0xbef5015a, 0x40d525c1, 0xbd1ef8b8, 0x3eaaeab6,
        0xc0a0d2df, 0xbf6ebd26, 0x40591d13, 0x3f0932e5, 0x3e3b8750, 0xbf3eef44, 0x3fd82a45, 0xbfb9e254, 0x3f519e80,
        0x40a8f84f, 0xbf57c992, 0xbf4b5860, 0xc0868dc1, 0xbf4bdcb5,
    ],
    &[
        0xbf138e11, 0x3f6c76cd, 0x3f61fdbc, 0x3fa29c99, 0x40062e95, 0x400451f8, 0x3eaede31, 0x40a99ce9, 0xbf243c98,
        0xc0448793, 0x400ec2ff, 0x3eaab5de, 0x3e5bbcfc, 0xbda084f5, 0x3f825e87, 0x3fc7d025, 0x3e235064, 0xbdaa09b0,
        0xbf2c07fe, 0x3fb87f0e, 0x3f66c6c8, 0x3f247ea9, 0xbf91fe7b, 0x403bebbd, 0x3fb0e378, 0x4093cd55, 0x408f9d2d,
        0x3dedda7c, 0xbf927c0a, 0x403b81b1, 0xc079eb13, 0xc007843f,
    ],
    &[
        0xbf892c90, 0x3dd67d2b, 0x3f9667cf, 0x3fb928e9, 0x4048ef21, 0x3fe0d9f8, 0x3f61f012, 0x40988e8d, 0xbfb1c9e3,
        0xc035e7be, 0x3dfc10c4, 0xbf12af63, 0x3f8948d4, 0x3ed24d8d, 0x3f7dffc4, 0x3f104b60, 0xbe9bcb6c, 0xbec10095,
        0x3e854e6d, 0x3f9f8f53, 0xbee5b304, 0x3e96bb28, 0x3e26985e, 0x406050ab, 0x3ec66d24, 0x409f4d5b, 0x4074a77e,
        0xbfe416e4, 0xbf6a6d28, 0x3fd44143, 0xc06ab46e, 0xc008d720,
    ],
    &[
        0x3f9b6f28, 0x3f6feb3f, 0x3f81fbf4, 0x400d0154, 0x3ed28fc0, 0xbe290df4, 0x3fb3df4c, 0xbfa01ba9, 0xbf4b0b54,
        0x3f466eb5, 0xbffae7eb, 0xbf4db86c, 0xbfb76258, 0xc00ea0b0, 0xbc469140, 0xbf2a9c5f, 0xc09a23b7, 0xbfec0769,
        0x3f90b091, 0x3f253d45, 0xc00b71b7, 0x40825607, 0xbfc4018f, 0x4002ee32, 0x3f76a424, 0x3ef05380, 0xc0036452,
        0x3f1412c8, 0x402364da, 0xc00b200d, 0x4093f8e1, 0xc0594e75,
    ],
    &[
        0x3eab83a2, 0xc00c6b75, 0x40b2e245, 0xbf9e299e, 0xbfa74e7a, 0x3fdc2b0e, 0x3f61ea68, 0xbe90e0b0, 0x403de146,
        0xc06c47a9, 0x4080d33a, 0xbfaf1270, 0xc028893b, 0xbee89c76, 0x4029fb97, 0xc0237b6d, 0x3ffebecd, 0x3e915dca,
        0x401f4c98, 0x4010d381, 0x4041c31c, 0xbfff4dda, 0x3f7327fc, 0xbfb412a8, 0xbf6c875f, 0x3f3524a7, 0x406597d9,
        0xbded72d4, 0x3faa801b, 0x4028b22d, 0xc0662553, 0xbfa693f4,
    ],
];

/// `logits fixture_b.bin 0 5 23 5`: float bits per position.
pub(super) const LOGITS_B: &[&[u32]] = &[
    &[
        0x3d101e90, 0xbf5edd0c, 0xc0424110, 0x3e7002b8, 0xc04882c9, 0x3f1d0db4, 0x3f88276e, 0x3f97eea8, 0xbf802bc5,
        0xbcd01200, 0x4027e30f, 0x3f23325e, 0x3e979fe2, 0xbf4d294f, 0x3fdb0c4e, 0x3f76ea72, 0xbdbeaa30, 0x3ffca3c3,
        0xbf763b87, 0xbf752828, 0x3fc5442a, 0x3f5ef46e, 0xc02552a0, 0xbfe8035a,
    ],
    &[
        0x3f599f1a, 0xbf6784a4, 0x4011658c, 0x3f5174d4, 0xc019013e, 0xbf931f27, 0xbe09786e, 0x3f143cca, 0x40260673,
        0xbfb16766, 0x3f3a3bf1, 0xbf51a46e, 0xc03a1d8b, 0x3fc99381, 0xbfea516d, 0xbd2c1448, 0xbfa330a4, 0x403e2299,
        0xbea1bf40, 0xbff6e3f3, 0x40c09e43, 0x3f13ab36, 0x3f83a30e, 0xbfb09408,
    ],
    &[
        0x3e7b14f8, 0x4006ca2f, 0x402c3ab6, 0xbe95e66e, 0x3fac8514, 0xbefa03ba, 0xc008dc1c, 0xbf89ff2e, 0xbe92f1ae,
        0xbf89f677, 0xc02f653f, 0xbf86de29, 0x3e9d00ea, 0x401fb82d, 0xbfd11b14, 0xc01979b2, 0xbf96f89f, 0x4013787a,
        0x3d973bdc, 0x3f365f24, 0xbf29c72a, 0x3ef2f211, 0xbf931fef, 0xbf39e294,
    ],
    &[
        0x401a7abf, 0xbf9d44f7, 0x4033e7b5, 0x3f6a43c5, 0xbfc816b1, 0xbfe13a08, 0x3ee3d4d5, 0x3f019395, 0x406356fe,
        0xbfccf9ed, 0xbf0309f5, 0xbfa08ed7, 0xc027ed02, 0x3fa9b407, 0xbf89b253, 0xbf5593be, 0xbf83a0ec, 0x40715146,
        0x3e817111, 0xbff98250, 0x40c1a12c, 0x3f107f53, 0x3ffe8c7b, 0xbf9d95f6,
    ],
];

/// `tokenize fixture_tokenizer.bin 296 ...`: upstream encodings (BOS, no EOS).
pub(super) const FIXTURE_ENCODINGS: &[(&str, &[u32])] = &[
    ("", &[1]),
    ("hello world", &[1, 271, 275]),
    (" hello", &[1, 259, 271]),
    ("hello  world", &[1, 271, 259, 275]),
    ("\n", &[1, 259, 278]),
    ("é", &[1, 259, 277]),
    ("日", &[1, 259, 233, 154, 168]),
    ("é日", &[1, 259, 277, 233, 154, 168]),
    ("q", &[1, 259, 116]),
    ("\n<s>\n", &[1, 259, 1]),
    ("<s>", &[1, 259, 284]),
    ("<0x41>", &[1, 259, 68]),
    ("a a", &[1, 295, 295]),
    ("hellohello", &[1, 271, 270]),
];

/// `tokenize tokenizer.bin 32000 ...` with the upstream 32,000-piece
/// `tokenizer.bin` (sha256 50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361).
pub(super) const REAL_ENCODINGS: &[(&str, &[u32])] = &[
    ("Once upon a time", &[1, 9038, 2501, 263, 931]),
    ("Hello, world!", &[1, 15043, 29892, 3186, 29991]),
    ("  two leading spaces", &[1, 259, 1023, 8236, 8162]),
    ("trailing ", &[1, 25053, 29871]),
    ("line\nbreak", &[1, 1196, 13, 8690]),
    ("naïve café", &[1, 1055, 30085, 345, 274, 28059]),
    ("日本語", &[1, 29871, 30325, 30346, 30968]),
    ("emoji 🙂", &[1, 953, 29877, 2397, 29871, 243, 162, 156, 133]),
    ("<s>", &[1, 529, 29879, 29958]),
    ("\n<s>\n", &[1, 29871, 13, 29966, 29879, 29958, 13]),
    ("</s>", &[1, 1533, 29879, 29958]),
    ("<0x41>", &[1, 529, 29900, 29916, 29946, 29896, 29958]),
    ("a\tb", &[1, 263, 12, 29890]),
    ("Lily and Ben went to the park.", &[1, 365, 2354, 322, 4111, 3512, 304, 278, 14089, 29889]),
];
