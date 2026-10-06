/* Runtime implementation. See recomp.h. */

#include "recomp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t *g_mem = NULL;
int sr_hit_hle = 0;

/* Out-of-range (wild guest pointer) access recorder. Bounds-safe MEM_* drops these; with
 * SR_OORLOG set, log the first distinct store addresses so a dropped game flag-write can be
 * found (a masked divergence). Reads are counted only. */
unsigned long g_oor_reads = 0, g_oor_writes = 0;
void sr_oor(uint32_t a, uint32_t v, int store) {
    static int logging = -1;
    if (logging < 0) logging = getenv("SR_OORLOG") ? 1 : 0;
    if (store) {
        g_oor_writes++;
        if (logging) {
            static uint32_t seen[64]; static int n = 0; int known = 0;
            for (int i = 0; i < n; i++) if (seen[i] == a) { known = 1; break; }
            if (!known && n < 64) { seen[n++] = a;
                fprintf(stderr, "OOR store [0x%08x] = 0x%08x\n", a, v); }
        }
    } else g_oor_reads++;
}

#define SR_VRAM_LOW  0x04000000u  /* guest VRAM/eDRAM base, below user RAM */
#define SR_RAM_SIZE  0x04000000u  /* 64 MB user RAM at 0x08000000 */

void sr_mem_init(void) {
    if (!g_mem) {
        /* One arena covering VRAM (0x04000000..) and user RAM (0x08000000..). g_mem points at
         * the user-RAM base so SR_RAM_BASE stays 0x08000000; the VRAM span sits below it. */
        uint32_t below = 0x08000000u - SR_VRAM_LOW;          /* 64 MB of VRAM/IO span */
        uint8_t *arena = (uint8_t *)calloc((size_t)below + SR_RAM_SIZE, 1);
        if (!arena) {
            fprintf(stderr, "sr_mem_init: out of memory\n");
            abort();
        }
        g_mem = arena + below;                                /* g_mem == guest 0x08000000 */
    }
}

static uint32_t g_loaded_end = 0;   /* highest guest address written by the loader */

uint32_t sr_loaded_end(void) { return g_loaded_end; }

void sr_load_segment(uint32_t vaddr, const void *data, uint32_t len) {
    sr_mem_init();
    memcpy(SR_HOST(vaddr), data, len);
    if (vaddr + len > g_loaded_end) g_loaded_end = vaddr + len;
}

/* Unaligned word access. Little-endian merge, identical to PPSSPP's interpreter. */
uint32_t sr_lwl(uint32_t rtv, uint32_t addr) {
    uint32_t shift = (addr & 3) * 8;
    uint32_t mem = MEM_R32(addr & ~3u);
    return (rtv & (0x00ffffffu >> shift)) | (mem << (24 - shift));
}
uint32_t sr_lwr(uint32_t rtv, uint32_t addr) {
    uint32_t shift = (addr & 3) * 8;
    uint32_t mem = MEM_R32(addr & ~3u);
    return (rtv & (0xffffff00u << (24 - shift))) | (mem >> shift);
}
void sr_swl(uint32_t addr, uint32_t rtv) {
    uint32_t shift = (addr & 3) * 8;
    uint32_t mem = MEM_R32(addr & ~3u);
    MEM_W32(addr & ~3u, (rtv >> (24 - shift)) | (mem & (0xffffff00u << shift)));
}
void sr_swr(uint32_t addr, uint32_t rtv) {
    uint32_t shift = (addr & 3) * 8;
    uint32_t mem = MEM_R32(addr & ~3u);
    MEM_W32(addr & ~3u, (rtv << shift) | (mem & (0x00ffffffu >> (24 - shift))));
}

/* ---- VFPU transcendental kernels (exact ports of PPSSPP's table-based vfpu_rcp/vfpu_rsqrt) ---- */

#include <math.h>

static int8_t (*sr_rcp_lut)[2] = NULL;
static uint32_t *sr_sin_lut8192 = NULL;
static int8_t (*sr_sin_lut_delta)[2] = NULL;
static int16_t *sr_sin_lut_interval_delta = NULL;
static uint8_t *sr_sin_lut_exceptions = NULL;
static uint32_t *sr_exp2_lut65536 = NULL;
static uint8_t (*sr_exp2_lut)[2] = NULL;
static int sr_vfpu_loaded = 0;

static void *sr_load_raw(const char *name, size_t expected) {
    char path[512];
    const char *base = getenv("PSP_VFPU_TABLES");
    snprintf(path, sizeof(path), "%s/%s.dat", base ? base : "assets/vfpu", name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sr_vfpu: cannot open %s\n", path); abort(); }
    void *t = malloc(expected);
    if (!t || fread(t, 1, expected, f) != expected) { fprintf(stderr, "sr_vfpu: bad table %s\n", path); abort(); }
    fclose(f);
    return t;
}

static void sr_vfpu_load(void) {
    if (sr_vfpu_loaded) return;
    sr_rcp_lut = (int8_t (*)[2])sr_load_raw("vfpu_rcp_lut", 262144);
    sr_sin_lut8192 = (uint32_t *)sr_load_raw("vfpu_sin_lut8192", 4100);
    sr_sin_lut_delta = (int8_t (*)[2])sr_load_raw("vfpu_sin_lut_delta", 262144);
    sr_sin_lut_interval_delta = (int16_t *)sr_load_raw("vfpu_sin_lut_interval_delta", 131074);
    sr_sin_lut_exceptions = (uint8_t *)sr_load_raw("vfpu_sin_lut_exceptions", 86938);
    sr_exp2_lut65536 = (uint32_t *)sr_load_raw("vfpu_exp2_lut65536", 512);
    sr_exp2_lut = (uint8_t (*)[2])sr_load_raw("vfpu_exp2_lut", 262144);
    sr_vfpu_loaded = 1;
}

static uint32_t sr_exp2_approx(uint32_t x) {
    if (x == 0x00800000u) return 0x00800000u;
    uint32_t a = sr_exp2_lut65536[x >> 16];
    x &= 0x0000FFFFu;
    uint32_t b = (uint32_t)(((2977151143ull * x) >> 23) + ((1032119999ull * (x * x)) >> 46));
    return (a + (uint32_t)(((uint64_t)(a + (1u << 23)) * (uint64_t)b) >> 32)) & 0xFFFFFFFCu;
}

static uint32_t sr_exp2_fixed(uint32_t x) {
    if (x == 0u) return 0u;
    if (x == 0x00800000u) return 0x00800000u;
    uint32_t A = sr_exp2_approx(x & 0xFFFFFFC0u);
    uint32_t B = sr_exp2_approx((x + 64) & 0xFFFFFFC0u);
    uint64_t a = ((uint64_t)A << 4) + (uint64_t)sr_exp2_lut[x >> 6][0] - 64u;
    uint64_t b = ((uint64_t)B << 4) + (uint64_t)sr_exp2_lut[x >> 6][1] - 64u;
    uint32_t y = (uint32_t)((a + (((b - a) * (x & 63)) >> 6)) >> 4);
    y &= 0xFFFFFFFCu;
    return y;
}

float sr_vfpu_exp2(float x) {
    sr_vfpu_load();
    int32_t bits;
    memcpy(&bits, &x, 4);
    if ((bits & 0x7FFFFFFF) <= 0x007FFFFF) return 1.0f;
    if (x != x) { bits = 0x7F800001; memcpy(&x, &bits, 4); return x; }
    if (x <= -126.0f) return 0.0f;
    if (x >= 128.0f) { bits = 0x7F800000; memcpy(&x, &bits, 4); return x; }
    bits = (int32_t)(x * 8388608.0f);  /* 0x1p23 */
    if (x < 0.0f) --bits;
    bits = (int32_t)0x3F800000 + (bits & (int32_t)0xFF800000) + (int32_t)sr_exp2_fixed(bits & 0x007FFFFF);
    memcpy(&x, &bits, 4);
    return x;
}

static uint32_t sr_sin_quantum(uint32_t x) {
    return x < (1u << 22) ? 1u : 1u << (32 - 22 - (uint32_t)__builtin_clz(x));
}
static uint32_t sr_sin_truncate_bits(uint32_t x) {
    return x & (0u - sr_sin_quantum(x));
}

static uint32_t sr_sin_fixed(uint32_t arg) {
    if (arg == 0u) return 0u;
    if (arg == 0x00800000u) return 0x10000000u;
    uint32_t L = sr_sin_lut8192[(arg >> 13) + 0];
    uint32_t H = sr_sin_lut8192[(arg >> 13) + 1];
    uint32_t A = L + (((H - L) * (((arg >> 6) & 127) + 0)) >> 7);
    uint32_t B = L + (((H - L) * (((arg >> 6) & 127) + 1)) >> 7);
    uint64_t a = ((uint64_t)A << 5) + (uint64_t)sr_sin_lut_delta[arg >> 6][0] * sr_sin_quantum(A);
    uint64_t b = ((uint64_t)B << 5) + (uint64_t)sr_sin_lut_delta[arg >> 6][1] * sr_sin_quantum(B);
    uint32_t v = (uint32_t)(((a * (64 - (arg & 63)) + b * (arg & 63)) >> 6) >> 5);
    v = sr_sin_truncate_bits(v);
    uint32_t lo = ((169u * ((arg >> 7) + 0)) >> 7) + (uint32_t)sr_sin_lut_interval_delta[(arg >> 7) + 0] + 16384u;
    uint32_t hi = ((169u * ((arg >> 7) + 1)) >> 7) + (uint32_t)sr_sin_lut_interval_delta[(arg >> 7) + 1] + 16384u;
    while (lo < hi) {
        uint32_t m = (lo + hi) / 2;
        uint32_t bb = sr_sin_lut_exceptions[m];
        uint32_t e = (arg & 0xFFFFFF80u) + (bb & 127u);
        if (e == arg) { v += sr_sin_quantum(v) * ((bb >> 7) ? 0xFFFFFFFFu : 1u); break; }
        else if (e < arg) lo = m + 1;
        else hi = m;
    }
    return v;
}

float sr_vfpu_sin(float x) {
    sr_vfpu_load();
    uint32_t bits;
    memcpy(&bits, &x, 4);
    uint32_t sign = bits & 0x80000000u;
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t significand = (bits & 0x007FFFFFu) | 0x00800000u;
    if (exponent == 0xFFu) { bits = sign ^ 0x7F800001u; memcpy(&x, &bits, 4); return x; }
    if (exponent < 0x7Fu) {
        if (exponent < 0x7Fu - 23u) significand = 0u;
        else significand >>= (0x7F - exponent);
    } else if (exponent > 0x7Fu) {
        if (exponent - 0x7Fu >= 25u && exponent - 0x7Fu < 32u) significand = 0u;
        else if ((exponent & 0x9Fu) == 0x9Fu) significand = 0u;
        else significand <<= ((exponent - 0x7Fu) & 31);
    }
    sign ^= ((significand << 7) & 0x80000000u);
    significand &= 0x00FFFFFFu;
    if (significand > 0x00800000u) significand = 0x01000000u - significand;
    uint32_t ret = sr_sin_fixed(significand);
    return (sign ? -1.0f : 1.0f) * (float)(int32_t)ret * 3.7252903e-09f;
}

float sr_vfpu_cos(float x) {
    sr_vfpu_load();
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits &= 0x7FFFFFFFu;
    uint32_t sign = 0u;
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t significand = (bits & 0x007FFFFFu) | 0x00800000u;
    if (exponent == 0xFFu) { bits = sign ^ 0x7F800001u; memcpy(&x, &bits, 4); return x; }
    if (exponent < 0x7Fu) {
        if (exponent < 0x7Fu - 23u) significand = 0u;
        else significand >>= (0x7F - exponent);
    } else if (exponent > 0x7Fu) {
        if (exponent - 0x7Fu >= 25u && exponent - 0x7Fu < 32u) significand = 0u;
        else if ((exponent & 0x9Fu) == 0x9Fu) significand = 0u;
        else significand <<= ((exponent - 0x7Fu) & 31);
    }
    sign ^= ((significand << 7) & 0x80000000u);
    significand &= 0x00FFFFFFu;
    if (significand >= 0x00800000u) { significand = 0x01000000u - significand; sign ^= 0x80000000u; }
    uint32_t ret = sr_sin_fixed(0x00800000u - significand);
    return (sign ? -1.0f : 1.0f) * (float)(int32_t)ret * 3.7252903e-09f;
}

static uint32_t sr_rcp_approx(uint32_t i) {
    return 0x3E800000u + ((uint32_t)((1ull << 47) / ((1ull << 23) + i)) & 0xFFFFFFFCu);
}

float sr_vfpu_rcp(float x) {
    sr_vfpu_load();
    uint32_t bits;
    memcpy(&bits, &x, 4);
    uint32_t s = bits & 0x80000000u, e = bits & 0x7F800000u, i = bits & 0x007FFFFFu;
    if ((bits & 0x7FFFFFFFu) > 0x7E800000u) {
        bits = (e == 0x7F800000u && i) ? (s ^ 0x7F800001u) : s;
        memcpy(&x, &bits, 4); return x;
    }
    if (e == 0u) { bits = s ^ 0x7F800000u; memcpy(&x, &bits, 4); return x; }
    uint32_t A = sr_rcp_approx(i & 0xFFFFFFC0u);
    uint32_t B = sr_rcp_approx((i + 64) & 0xFFFFFFC0u);
    uint64_t a = ((uint64_t)A << 6) + (uint64_t)sr_rcp_lut[i >> 6][0] * 4u;
    uint64_t b = ((uint64_t)B << 6) + (uint64_t)sr_rcp_lut[i >> 6][1] * 4u;
    uint32_t v = (uint32_t)((a + (((b - a) * (i & 63)) >> 6)) >> 6);
    v &= 0xFFFFFFFCu;
    bits = s + (0x3F800000u - e) + v;
    memcpy(&x, &bits, 4); return x;
}

/* PPSSPP's USE_VFPU_SQRT is false, so vrsq is plain single-precision 1.0f / sqrtf(x)
 * (sqrtf then reciprocal), which matches the reference trace bit-for-bit. */
float sr_vfpu_rsqrt(float x) {
    return 1.0f / sqrtf(x);
}

/* vsqrt with USE_VFPU_SQRT false is fabsf(sqrtf(x)). */
float sr_vfpu_sqrt(float x) {
    return fabsf(sqrtf(x));
}

/* vasin: asin(x)*2/pi bit-exato, como o hardware: a curva é aproximada por 128 segmentos
 * quadráticos em ponto fixo (port de vfpu_asin, PPSSPP MIPSVFPUUtils.cpp). */
typedef struct { int32_t c0, m; int16_t n, e; } SrVfpuSeg;
static const SrVfpuSeg sr_asin_seg[128] = {
    { 0x0000000,  0x145F3E0,  0x000, 0x77 }, { 0x0A2F978,  0x145F8F8,  0x03C, 0x77 },
    { 0x0A2FAD2,  0x0A30194,  0x032, 0x78 }, { 0x0F47B72,  0x0A3093C,  0x047, 0x78 },
    { 0x0A2FFF5,  0x05189B4,  0x02D, 0x79 }, { 0x0CBC4BC,  0x0519018,  0x037, 0x79 },
    { 0x0F48CB2,  0x05197C4,  0x042, 0x79 }, { 0x08EAC42,  0x028D058,  0x025, 0x7A },
    { 0x0A31463,  0x028D57C,  0x02B, 0x7A }, { 0x0B77F17,  0x028DB40,  0x030, 0x7A },
    { 0x0CBECAD,  0x028E1AC,  0x035, 0x7A }, { 0x0E05D7A,  0x028E8C0,  0x03A, 0x7A },
    { 0x0F4D1D1,  0x028F07C,  0x03F, 0x7A }, { 0x084A503,  0x0147C74,  0x022, 0x7B },
    { 0x08EE339,  0x01480FC,  0x024, 0x7B }, { 0x09923B2,  0x01485DC,  0x027, 0x7B },
    { 0x0A3669B,  0x0148B14,  0x02A, 0x7B }, { 0x0ADAC20,  0x01490A0,  0x02D, 0x7B },
    { 0x0B7F46B,  0x014968C,  0x030, 0x7B }, { 0x0C23FAB,  0x0149CCC,  0x033, 0x7B },
    { 0x0CC8E0B,  0x014A368,  0x036, 0x7B }, { 0x0D6DFBB,  0x014AA5C,  0x038, 0x7B },
    { 0x0E134E4,  0x014B1B0,  0x03B, 0x7B }, { 0x0EB8DB6,  0x014B960,  0x03E, 0x7B },
    { 0x0F5EA61,  0x014C170,  0x041, 0x7B }, { 0x080258A,  0x00A64F0,  0x022, 0x7C },
    { 0x0855801,  0x00A6958,  0x023, 0x7C }, { 0x08A8CA9,  0x00A6DF4,  0x025, 0x7C },
    { 0x08FC39F,  0x00A72C0,  0x027, 0x7C }, { 0x094FCFD,  0x00A77BC,  0x028, 0x7C },
    { 0x09A38D8,  0x00A7CF0,  0x02A, 0x7C }, { 0x09F774E,  0x00A8258,  0x02B, 0x7C },
    { 0x0A4B876,  0x00A87F0,  0x02D, 0x7C }, { 0x0A9FC6B,  0x00A8DC8,  0x02F, 0x7C },
    { 0x0AF434D,  0x00A93D0,  0x030, 0x7C }, { 0x0B48D31,  0x00A9A14,  0x032, 0x7C },
    { 0x0B9DA37,  0x00AA08C,  0x034, 0x7C }, { 0x0BF2A7A,  0x00AA744,  0x036, 0x7C },
    { 0x0C47E19,  0x00AAE38,  0x038, 0x7C }, { 0x0C9D531,  0x00AB564,  0x03A, 0x7C },
    { 0x0CF2FE0,  0x00ABCD4,  0x03C, 0x7C }, { 0x0D48E48,  0x00AC480,  0x03D, 0x7C },
    { 0x0D9F083,  0x00ACC70,  0x040, 0x7C }, { 0x0DF56B7,  0x00AD4A0,  0x042, 0x7C },
    { 0x0E4C103,  0x00ADD14,  0x044, 0x7C }, { 0x0EA2F89,  0x00AE5D0,  0x046, 0x7C },
    { 0x0EFA26D,  0x00AEED4,  0x048, 0x7C }, { 0x0F519D1,  0x00AF81C,  0x04B, 0x7C },
    { 0x0FA95DC,  0x00B01B4,  0x04D, 0x7C }, { 0x0800B5A,  0x00585CC,  0x027, 0x7D },
    { 0x082CE3F,  0x0058AE4,  0x028, 0x7D }, { 0x08593AE,  0x0059028,  0x02A, 0x7D },
    { 0x0885BC1,  0x0059594,  0x02B, 0x7D }, { 0x08B2688,  0x0059B2C,  0x02D, 0x7D },
    { 0x08DF41C,  0x005A0F0,  0x02E, 0x7D }, { 0x090C493,  0x005A6E4,  0x02F, 0x7D },
    { 0x0939801,  0x005AD04,  0x031, 0x7D }, { 0x0966E82,  0x005B354,  0x032, 0x7D },
    { 0x0994829,  0x005B9D8,  0x034, 0x7D }, { 0x09C2514,  0x005C090,  0x035, 0x7D },
    { 0x09F0559,  0x005C780,  0x037, 0x7D }, { 0x0A1E915,  0x005CEA4,  0x039, 0x7D },
    { 0x0A4D064,  0x005D604,  0x03B, 0x7D }, { 0x0A7BB62,  0x005DD9C,  0x03D, 0x7D },
    { 0x0AAAA2D,  0x005E574,  0x03F, 0x7D }, { 0x0AD9CE3,  0x005ED8C,  0x041, 0x7D },
    { 0x0B093A5,  0x005F5E8,  0x043, 0x7D }, { 0x0B38E95,  0x005FE84,  0x045, 0x7D },
    { 0x0B68DD1,  0x006076C,  0x048, 0x7D }, { 0x0B99183,  0x0061098,  0x04A, 0x7D },
    { 0x0BC99CC,  0x0061A1C,  0x04C, 0x7D }, { 0x0BFA6D4,  0x00623E8,  0x04F, 0x7D },
    { 0x0C2B8C4,  0x0062E0C,  0x051, 0x7D }, { 0x0C5CFC3,  0x0063888,  0x055, 0x7D },
    { 0x0C8EC01,  0x0064360,  0x058, 0x7D }, { 0x0CC0DAD,  0x0064E94,  0x05A, 0x7D },
    { 0x0CF34F0,  0x0065A34,  0x05E, 0x7D }, { 0x0D26204,  0x0066638,  0x061, 0x7D },
    { 0x0D59519,  0x00672AC,  0x065, 0x7D }, { 0x0D8CE6A,  0x0067F94,  0x068, 0x7D },
    { 0x0DC0E2D,  0x0068CF8,  0x06C, 0x7D }, { 0x0DF54A1,  0x0069ADC,  0x070, 0x7D },
    { 0x0E2A205,  0x006A944,  0x075, 0x7D }, { 0x0E5F6A0,  0x006B840,  0x079, 0x7D },
    { 0x0E952B6,  0x006C7D0,  0x07E, 0x7D }, { 0x0ECB694,  0x006D7FC,  0x083, 0x7D },
    { 0x0F02287,  0x006E8D8,  0x089, 0x7D }, { 0x0F396E9,  0x006FA64,  0x08E, 0x7D },
    { 0x0F7140F,  0x0070CAC,  0x094, 0x7D }, { 0x0FA9A58,  0x0071FC4,  0x09B, 0x7D },
    { 0x0FE2A2C,  0x00733B0,  0x0A2, 0x7D }, { 0x080E1FC,  0x003A440,  0x054, 0x7E },
    { 0x082B415,  0x003AF24,  0x058, 0x7E }, { 0x0848BA0,  0x003BA90,  0x05C, 0x7E },
    { 0x08668DE,  0x003C688,  0x061, 0x7E }, { 0x0884C18,  0x003D314,  0x066, 0x7E },
    { 0x08A3599,  0x003E048,  0x06B, 0x7E }, { 0x08C25B2,  0x003EE30,  0x071, 0x7E },
    { 0x08E1CBD,  0x003FCD8,  0x078, 0x7E }, { 0x0901B1D,  0x0040C50,  0x07E, 0x7E },
    { 0x0922135,  0x0041CB0,  0x086, 0x7E }, { 0x0942F7D,  0x0042E04,  0x08E, 0x7E },
    { 0x096466E,  0x0044070,  0x097, 0x7E }, { 0x0986692,  0x0045408,  0x0A1, 0x7E },
    { 0x09A9080,  0x00468F0,  0x0AC, 0x7E }, { 0x09CC4E0,  0x0047F44,  0x0B8, 0x7E },
    { 0x09F0469,  0x004973C,  0x0C5, 0x7E }, { 0x0A14FE8,  0x004B104,  0x0D5, 0x7E },
    { 0x0A3A848,  0x004CCD4,  0x0E6, 0x7E }, { 0x0A60E8A,  0x004EAF4,  0x0FA, 0x7E },
    { 0x0A883D6,  0x0050BB8,  0x111, 0x7E }, { 0x0AB097F,  0x0052F84,  0x12B, 0x7E },
    { 0x0ADA106,  0x00556D8,  0x149, 0x7E }, { 0x0B04C2B,  0x005824C,  0x16D, 0x7E },
    { 0x0B30CFB,  0x005B29C,  0x198, 0x7E }, { 0x0B5E5E5,  0x005E8C0,  0x1CA, 0x7E },
    { 0x0B8E1C8,  0x00625FC, -0x1F7, 0x7E }, { 0x0BBF42A,  0x0066BF4, -0x1A9, 0x7E },
    { 0x0BF295F,  0x006BCE8, -0x146, 0x7E }, { 0x0C286D1,  0x0071C04, -0x0C5, 0x7E },
    { 0x0C6137A,  0x0078DE4, -0x018, 0x7E }, { 0x0C9D887,  0x008198C,  0x0DB, 0x7E },
    { 0x0CDEA7F,  0x008CA58, -0x1BE, 0x7E }, { 0x0D24B35,  0x009B428,  0x07D, 0x7E },
    { 0x0D72566,  0x00AFE0C,  0x06F, 0x7E }, { 0x0DCA3BF,  0x00D035C,  0x0C6, 0x7E },
    { 0x0E32341,  0x010ECE0,  0x1DC, 0x7E }, { 0x0EB9DD1,  0x028C52C, -0x034, 0x7E },
};

static int32_t sr_seg_square(uint32_t x2) {
    int32_t t = abs((int32_t)(x2 >> 6) - 512);   /* o quadrador só vê os 10 bits altos */
    return (t * t + 255) >> 8;
}

static int32_t sr_seg_interp(const SrVfpuSeg *g, uint32_t x2) {
    return g->c0 + (int32_t)(((int64_t)g->m * x2) >> 17) + ((g->n * sr_seg_square(x2)) >> 9);
}

static uint32_t sr_fixed_to_bits(uint64_t v, int frac, int neg) {
    uint32_t sign = neg ? 0x80000000u : 0u;
    if (v == 0) return sign;
    int top = 63 - __builtin_clzll(v);
    uint64_t sig = top <= 23 ? v << (23 - top) : v >> (top - 23);
    return sign + ((uint32_t)(top - frac + 127) << 23) + ((uint32_t)sig & 0x007FFFFFu);
}

static uint32_t sr_asin_fixed(uint32_t x) {
    if (x == 0u) return 0u;
    if (x == 1u << 23) return 1u << 30;
    const SrVfpuSeg *g = &sr_asin_seg[x >> 16];
    /* O primeiro segmento é linear e mantém o passo de 2^-30 da saída. */
    uint32_t v = (uint32_t)sr_seg_interp(g, x & 0xFFFF) & ((x >> 16) == 0 ? ~1u : ~3u);
    return g->e >= 120 ? v << (g->e - 120) : v >> (120 - g->e);
}

float sr_vfpu_asin(float x) {
    uint32_t bits, r;
    memcpy(&bits, &x, 4);
    uint32_t sign = bits & 0x80000000u, a = bits & 0x7FFFFFFFu;
    if (a > 0x3F800000u) {
        r = 0x7F800001u ^ sign;                  /* |x| > 1 e NaN: NaN sinalizante */
    } else {
        /* |x| * 2^23 truncado; abaixo de 2^-23 (denormais inclusos) o resultado é 0. */
        int e = (int)(a >> 23);
        uint32_t sig = (a & 0x007FFFFFu) | 0x00800000u;
        uint32_t fixed = e < 104 ? 0u : sig >> (127 - e);
        r = sr_fixed_to_bits(sr_asin_fixed(fixed), 30, sign != 0);
    }
    memcpy(&x, &r, 4);
    return x;
}

/* ---- VFPU prefix application (ports of PPSSPP ApplyPrefixST / ApplyPrefixD) ---- */

static const float SR_VFPU_CONST[8] = {0.0f, 1.0f, 2.0f, 0.5f, 3.0f, 1.0f / 3.0f, 0.25f, 1.0f / 6.0f};

void sr_vread(float *r, const CpuState *s, const uint8_t *idx, int n, uint32_t prefix) {
    for (int i = 0; i < n; i++)
        r[i] = s->v[idx[i]];
    if (prefix == 0xe4)  /* identity */
        return;
    float orig[4] = {0.0f, 0.0f, 0.0f, 0.0f};  /* swizzle lanes >= n read as 0 (PPSSPP) */
    for (int i = 0; i < n; i++)
        orig[i] = r[i];
    for (int i = 0; i < n; i++) {
        int regnum = (prefix >> (i * 2)) & 3;
        int absbit = (prefix >> (8 + i)) & 1;
        int constant = (prefix >> (12 + i)) & 1;
        int negate = (prefix >> (16 + i)) & 1;
        if (!constant) {
            r[i] = orig[regnum];
            if (absbit)
                ((uint32_t *)r)[i] &= 0x7FFFFFFF;
        } else {
            r[i] = SR_VFPU_CONST[regnum + (absbit << 2)];
        }
        if (negate)
            ((uint32_t *)r)[i] ^= 0x80000000;
    }
}

static float sr_clamp(float x, float lo, float hi) {
    /* Como vfpu_clamp do PPSSPP: NaN é preservado e -0.0 vira +0.0 quando lo = +0.0. */
    return x >= hi ? hi : (x <= lo ? lo : x);
}

void sr_vwrite(CpuState *s, const uint8_t *idx, float *d, int n, uint32_t dprefix) {
    if (dprefix) {
        for (int i = 0; i < n; i++) {
            int sat = (dprefix >> (i * 2)) & 3;
            if (sat == 1)
                d[i] = sr_clamp(d[i], 0.0f, 1.0f);
            else if (sat == 3)
                d[i] = sr_clamp(d[i], -1.0f, 1.0f);
        }
    }
    for (int i = 0; i < n; i++)
        if (!((dprefix >> (8 + i)) & 1))  /* write mask */
            s->v[idx[i]] = d[i];
}

/* ---- dispatch table: guest address -> native function (open-addressing hash) ---- */

#define SR_DISP_CAP 65536  /* power of two; homebrew has < a few thousand functions */
static uint32_t s_disp_addr[SR_DISP_CAP];
static RecompFn s_disp_fn[SR_DISP_CAP];
static int s_disp_init = 0;

static void disp_init(void) {
    memset(s_disp_addr, 0, sizeof(s_disp_addr));
    memset(s_disp_fn, 0, sizeof(s_disp_fn));
    s_disp_init = 1;
}

void sr_register(uint32_t addr, RecompFn fn) {
    if (!s_disp_init) disp_init();
    uint32_t i = (addr >> 2) & (SR_DISP_CAP - 1);
    for (uint32_t n = 0; n < SR_DISP_CAP; n++) {
        if (s_disp_fn[i] == NULL || s_disp_addr[i] == addr) {
            s_disp_addr[i] = addr;
            s_disp_fn[i] = fn;
            return;
        }
        i = (i + 1) & (SR_DISP_CAP - 1);
    }
    fprintf(stderr, "sr_register: dispatch table full\n");
    abort();
}

RecompFn sr_lookup(uint32_t addr) {
    if (!s_disp_init) return NULL;
    uint32_t i = (addr >> 2) & (SR_DISP_CAP - 1);
    for (uint32_t n = 0; n < SR_DISP_CAP; n++) {
        if (s_disp_fn[i] == NULL) return NULL;
        if (s_disp_addr[i] == addr) return s_disp_fn[i];
        i = (i + 1) & (SR_DISP_CAP - 1);
    }
    return NULL;
}

void dispatch(CpuState *s, uint32_t target) {
    if (getenv("SR_DISPLOG")) {
        /* Log first invocation of boot/state-machine handlers, to see which actually run. */
        static uint32_t seen[256]; static int n = 0; int known = 0;
        if (target == 0x0880b398 || target == 0x0880b84c || target == 0x0880c584 ||
            target == 0x0880abac || target == 0x089dc638 || target == 0x0880c454 ||
            (target >= 0x0880b000 && target < 0x0880d000)) {
            for (int i = 0; i < n; i++) if (seen[i] == target) { known = 1; break; }
            if (!known && n < 256) { seen[n++] = target;
                fprintf(stderr, "DISPATCH boot handler 0x%08x (pc=0x%08x)\n", target, s->pc); }
        }
    }
    RecompFn fn = sr_lookup(target);
    if (fn) {
        fn(s);
        return;
    }
    /* No recompiled target -- an indirect jump / jump table / unanalyzed code the analyzer
     * never discovered as a function. By default this is a hard error (no silent no-op). For
     * bring-up triage during the renderer work, SR_DISPATCHLOG reports each distinct missing
     * target once, and SR_DISPATCH_PERMISSIVE additionally treats the miss as a call that did
     * nothing and returned 0 (v0=0) so a render session survives the gap instead of aborting.
     * Like SR_PERMISSIVE for HLE NIDs, this returns fake success and is therefore opt-in.
     * See RENDERING.md section 7. */
    {
        static int permissive = -1, logging = -1;
        if (permissive < 0) permissive = getenv("SR_DISPATCH_PERMISSIVE") ? 1 : 0;
        if (logging < 0)    logging = (permissive || getenv("SR_DISPATCHLOG")) ? 1 : 0;
        if (logging) {
            static uint32_t seen[256]; static int n = 0; int known = 0;
            for (int i = 0; i < n; i++) if (seen[i] == target) { known = 1; break; }
            if (!known) {
                if (n < 256) seen[n++] = target;
                fprintf(stderr, "dispatch: no recompiled function at 0x%08x (pc=0x%08x)%s\n",
                        target, s->pc, permissive ? " [permissive: no-op return]" : "");
            }
        }
        if (permissive) { s->r[2] = 0; return; }
    }
    fprintf(stderr, "dispatch: no recompiled function at 0x%08x (pc=0x%08x)\n", target, s->pc);
    fprintf(stderr, "dispatch: ra=0x%08x a0=0x%08x a1=0x%08x a2=0x%08x a3=0x%08x sp=0x%08x\n",
            s->r[31], s->r[4], s->r[5], s->r[6], s->r[7], s->r[29]);
    /* Dump user RAM so the missing code (runtime-loaded overlay / module) can be identified
     * and recompiled offline: tools/analyze.py + codegen.py accept a flat image + base. */
    {
        FILE *f = fopen("dispatch_dump.bin", "wb");
        if (f) {
            fwrite(SR_HOST(0x08800000u), 1, 0x01800000u, f);   /* 0x08800000..0x0A000000 */
            fclose(f);
            fprintf(stderr, "dispatch: wrote dispatch_dump.bin (RAM 0x08800000..0x0A000000)\n");
        }
    }
    abort();
}

jmp_buf g_hle_jmp;

void sr_hle_call(CpuState *s, uint32_t nid) {
    (void)nid;
    sr_hit_hle = 1;
    s->pc = s->r[31];  /* as if the import stub returned */
    longjmp(g_hle_jmp, 1);
}

void sr_unimplemented(uint32_t pc, const char *reason) {
    fprintf(stderr, "sr_unimplemented: function 0x%08x: %s\n", pc, reason);
    abort();
}

/* ---- tracing (TRACE_FORMAT.md), mirroring the PPSSPP instrumentation ---- */

static FILE *s_fp = NULL;
static uint32_t s_r[32], s_fi[32], s_vi[128], s_hi, s_lo, s_fcr31, s_pc, s_op;
static unsigned long long s_step = 0;

int sr_trace_open(const char *path, const char *target, uint32_t start_pc) {
    s_fp = fopen(path, "wb");
    if (!s_fp) return -1;
    fprintf(s_fp, "# psp-recomp trace v1 oracle=recomp target=%s start_pc=0x%08x\n",
            target ? target : "unknown", start_pc);
    s_step = 0;
    return 0;
}

void sr_trace_close(void) {
    if (s_fp) { fflush(s_fp); fclose(s_fp); s_fp = NULL; }
}

/* SR_WATCH=<hexaddr>: poll a guest halfword before every instruction and report the PC of the
 * instruction after which it changed (plus ra for caller context). Polling here instead of in
 * the MEM_W* path means no rebuild of the big recompiled object and catches every writer that
 * executes as guest code (HLE-side writes attribute to the syscall site). */
static uint32_t s_watch_addr = 0;
static int s_watch_on = -1;
void sr_begin(const CpuState *s, uint32_t pc, uint32_t op) {
    if (s_watch_on < 0) {
        const char *w = getenv("SR_WATCH");
        s_watch_addr = w ? (uint32_t)strtoul(w, NULL, 16) : 0;
        s_watch_on = s_watch_addr ? 1 : 0;
    }
    if (s_watch_on) {
        static uint16_t last_val; static int primed = 0; static uint32_t last_pc; static int hits = 0;
        uint16_t v = sr_r16(s_watch_addr);
        if (!primed) { last_val = v; primed = 1; }
        else if (v != last_val && hits < 400) {
            hits++;
            fprintf(stderr, "WATCH [0x%08x] 0x%04x -> 0x%04x after pc=0x%08x ra=0x%08x\n",
                    s_watch_addr, last_val, v, last_pc, s->r[31]);
            last_val = v;
        } else if (v != last_val) last_val = v;
        last_pc = pc;
    }
    if (!s_fp) return;
    memcpy(s_r, s->r, sizeof(s_r));
    memcpy(s_fi, s->fi, sizeof(s_fi));
    memcpy(s_vi, s->vi, sizeof(s_vi));
    s_hi = s->hi; s_lo = s->lo; s_fcr31 = s->fcr31;
    s_pc = pc; s_op = op;
}

void sr_end(const CpuState *s, uint32_t mem_addr, int mem_size) {
    if (!s_fp) return;
    char line[4096];
    int n = snprintf(line, sizeof(line), "%llu pc=0x%08x op=0x%08x", s_step, s_pc, s_op);
    for (int i = 1; i < 32; i++)
        if (s->r[i] != s_r[i]) n += snprintf(line + n, sizeof(line) - n, " r%d=0x%08x", i, s->r[i]);
    if (s->hi != s_hi) n += snprintf(line + n, sizeof(line) - n, " hi=0x%08x", s->hi);
    if (s->lo != s_lo) n += snprintf(line + n, sizeof(line) - n, " lo=0x%08x", s->lo);
    for (int i = 0; i < 32; i++)
        if (s->fi[i] != s_fi[i]) n += snprintf(line + n, sizeof(line) - n, " f%d=0x%08x", i, s->fi[i]);
    if (s->fcr31 != s_fcr31) n += snprintf(line + n, sizeof(line) - n, " fcr31=0x%08x", s->fcr31);
    for (int i = 0; i < 128; i++)
        if (s->vi[i] != s_vi[i]) n += snprintf(line + n, sizeof(line) - n, " v%d=0x%08x", i, s->vi[i]);
    if (mem_size == 1)
        n += snprintf(line + n, sizeof(line) - n, " m8[0x%08x]=0x%02x", mem_addr, MEM_R8(mem_addr));
    else if (mem_size == 2)
        n += snprintf(line + n, sizeof(line) - n, " m16[0x%08x]=0x%04x", mem_addr, MEM_R16(mem_addr));
    else if (mem_size == 4)
        n += snprintf(line + n, sizeof(line) - n, " m32[0x%08x]=0x%08x", mem_addr, MEM_R32(mem_addr));
    else if (mem_size == 16)
        for (int w = 0; w < 4; w++)
            n += snprintf(line + n, sizeof(line) - n, " m32[0x%08x]=0x%08x", mem_addr + w * 4, MEM_R32(mem_addr + w * 4));
    line[n] = '\n';
    fwrite(line, 1, n + 1, s_fp);
    s_step++;
    /* Safety: a correct run stops at the HLE boundary. An unbounded trace means the
     * generated code is looping (a codegen bug) -- fail loudly instead of filling the disk. */
    if (s_step > 50000000ULL) {
        fprintf(stderr, "sr_end: trace exceeded 50M steps (likely a codegen infinite loop)\n");
        sr_trace_close();
        abort();
    }
}
