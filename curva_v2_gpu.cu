#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <sched.h>
#include <gmp.h>
#include <openssl/evp.h>
#include <cuda_runtime.h>


#define LOTE                512
#define DISPLAY_MS          200
#define WINDOW_W            6
#define WINDOW_N            (1 << WINDOW_W)
#define GBASE_N             256
#define SOMENTE_COMPRIMIDO  1
#define MAX_TASKS           1048576
#define CHECKPOINT_FILE     "checkpoint.dat"
#define CHECKPOINT_INTERVAL_S 30
#define MAX_WORKERS         256
#define MAX_STEP_IDX        16777216
#define VRAM_BUDGET_BYTES   ((size_t)(13.9 * 1024.0 * 1024.0 * 1024.0))
#define BLOCKS_PER_WORKER   2097152
#define GPU_THREADS         32
/* GPU_THREADS=32: threads que COOPERAM em cada worker (1 bloco = 1 worker),
 * mantendo o WARP CHEIO (32 lanes uteis). Registradores sao alocados em
 * granularidade de WARP (32 slots), entao GPU_THREADS<32 NAO economiza regs
 * e so desperdica lanes (GPU_THREADS=8 -> 1/4 das lanes -> ~50 Mkey/s).
 * O footprint do kernel (254 regs, 0 spill com BATCH_SIZE=4) limita a
 * ocupacao a 65536/(254*32) = 8 blocos/SM -> 288 blocos residentes.
 * Ter 24 blocos/SM (todos os 864) exigiria <=85 regs -> spill obrigatorio.
 * Sao mutuamente exclusivos: "864 residentes" XOR "0 spill". Aqui priorizamos
 * 0 spill + warp cheio (maior throughput por warp). Cobertura segue 100%. */
#define THREADS_PER_WORKER  (BLOCKS_PER_WORKER * GPU_THREADS)
#define BATCH_SIZE          8
/* BATCH_SIZE=8: pontos por inversao de Montgomery. EMPIRICAMENTE mais rapido
 * que 4 (240-270 vs 170-180 Mkey/s) porque faz 1 inversao a cada 8 chaves em
 * vez de 1 a cada 4. Com __launch_bounds__(32,8) o compilador tem orcamento
 * total de regs (255) e o ptxas encaixa os 4 arrays (batch_X/Y/Z/prefix) em
 * 254 regs com 0 spill via reuso entre os passos forward/backward (a conta
 * ingenua 4*8*8=256 e so um limite superior; ptxas faz liveness e cabe).
 * Cobertura 100% (uniao sobre tid de [0,M), independente de BATCH_SIZE). */
#define BATCH_SIZE_LOG2     3       /* log2(BATCH_SIZE). BATCH_SIZE DEVE ser potencia de 2 (4->2, 8->3, 16->4).
                                     * Usado para bigstride = (BATCH_SIZE*GPU_THREADS)*step via dobramentos. */
#define GPU_MACRO_ITERS     4096
#define BLOCKS_PER_SM       8       /* DERIVA num_workers: persistent_blocks = sm_count * BLOCKS_PER_SM = 36*8 = 288.
                                     * 8 e o MAXIMO de blocos/SM que cabe sem spill: 8 blocos * 32 lanes * 254 regs
                                     * = 65024 <= 65536 (register file/SM). Logo TODOS os 288 ficam residentes:
                                     * 288 ativos de 288 (sem ondas, sem blocos idle). A range e dividida em
                                     * 288 sub-faixas, cada uma totalmente coberta por 1 bloco com warp cheio. */
#define LB_BLOCKS_PER_SM    8       /* Hint de ocupacao para __launch_bounds__ = blocos/SM residentes (= BLOCKS_PER_SM aqui).
                                     * launch_bounds(32, 8) -> teto de regs = 65536/(32*8) = 256 -> 255 >= 254 usados -> 0 spill,
                                     * e minBlocks=8 garante os 8 blocos/SM residentes. */

/* Status: MICRO_K=0 (manter desabilitado). */

#define MICRO_K              0
#define MICRO_MAX_ITER       4096LL

#if GPU_THREADS == 32
  #define GPU_THREADS_LOG2 5
#elif GPU_THREADS == 64
  #define GPU_THREADS_LOG2 6
#elif GPU_THREADS == 128
  #define GPU_THREADS_LOG2 7
#elif GPU_THREADS == 256
  #define GPU_THREADS_LOG2 8
#elif GPU_THREADS == 16
  #define GPU_THREADS_LOG2 4
#elif GPU_THREADS == 512
  #define GPU_THREADS_LOG2 9
#elif GPU_THREADS == 1024
  #define GPU_THREADS_LOG2 10
#elif GPU_THREADS == 8
  #define GPU_THREADS_LOG2 3
#elif GPU_THREADS == 4
  #define GPU_THREADS_LOG2 2
#elif GPU_THREADS == 2
  #define GPU_THREADS_LOG2 1
#elif GPU_THREADS == 1
  #define GPU_THREADS_LOG2 0
#endif

static constexpr int compile_time_log2(unsigned long long n) {
    return (n <= 1) ? 0 : 1 + compile_time_log2(n / 2);
}

#define _INIT_NBITS_RAW compile_time_log2(THREADS_PER_WORKER)
#define _CT_MAX(a,b) ((a)>(b)?(a):(b))
#define INIT_NBITS _CT_MAX(_INIT_NBITS_RAW, GPU_THREADS_LOG2 + 1)
static int g_target_steps = 200000; 

/* ════════════════════════════════════════════════════════════════════
 * CONFIG DA GERAÇÃO DE STEPS (saltos / pulos) — editável aqui no topo
 *
 * ATENÇÃO: estes definem a QUANTIDADE DE TAMANHOS DE SALTO distintos
 * (quantos steps diferentes existem), NÃO quantas iterações cada salto faz.
 * O nº de iterações de cada salto é DERIVADO automaticamente como
 *   ceil(range_do_worker / valor_do_salto)
 * no preenchimento de d_gpu_max_iter (cobre início→fim do range do worker).
 * ════════════════════════════════════════════════════════════════════ */
#define STEPS_PER_MAGNITUDE  13878   /* nº de saltos distintos gerados POR magnitude (2^k) */
#define SEQ_STEPS_COUNT      22464   /* nº de saltos sequenciais/lineares (S=1..N) no final */

/* MODO DE VARREDURA:
 *   0 = usa TODOS os steps normalmente (maior→menor, cobertura logarítmica completa).
 *   1 = usa APENAS o último step (S=1, exaustivo) — varre todo o range do worker
 *       chave-por-chave, ignorando os saltos maiores.
 * O último step (si = num_steps-1) tem valor 1 por construção; max_iter dele já é
 * ceil(range_do_worker / 1) = range inteiro do worker. Cobertura 100% em ambos. */
#define STEP_ONLY_EXHAUSTIVE 1

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA erro %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUDA_SOFT(call, label) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[Worker] CUDA erro %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        goto label; \
    } \
} while(0)

#define ADDR_TYPE_P2PKH      0   /* Legacy 1... (compressed) */
#define ADDR_TYPE_P2PKH_U    1   /* Legacy 1... (uncompressed) */
#define ADDR_TYPE_P2SH       2   /* Nested SegWit 3... */
#define ADDR_TYPE_P2WPKH     3   /* Native SegWit bc1q... */
#define ADDR_TYPE_P2TR       4   /* Taproot bc1p... */

#define GPU_MODE_HASH160     0   /* P2PKH / P2WPKH: hash160(sha256(compressed_pub)) */
#define GPU_MODE_P2SH        1   /* P2SH-P2WPKH: hash160(0x0014 || hash160(...)) */
#define GPU_MODE_P2TR        2   /* Taproot: x-only pubkey comparison (32 bytes) */
#define GPU_MODE_UNCOMP      3   /* Uncompressed P2PKH */


static const char P_HEX[]   = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
static const char G_x_HEX[] = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static const char G_y_HEX[] = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

static mpz_t P_val, G_x_val, G_y_val, P_minus_2;

typedef struct { mpz_t X, Y, Z; } JacPoint;
typedef struct { mpz_t x, y; int valid; } AffPoint;

typedef struct {
    mpz_t d_ysq, d_xysq, d_S, d_xx, d_M, d_ysq2, d_X2, d_Y2, d_Z2, d_tmp;
    mpz_t a_Z1sq, a_Z2sq, a_U1, a_U2, a_S1, a_S2, a_H, a_R,
          a_H2, a_H3, a_U1H2, a_X3, a_Y3, a_Z3, a_tmp;
    mpz_t z_Z1sq, z_U2, z_S2, z_H, z_R, z_H2, z_H3,
          z_U1H2, z_X3, z_Y3, z_Z3, z_tmp, z_dummy;
} JacWorkspace;

static void jac_workspace_init(JacWorkspace *ws) {
    mpz_inits(ws->d_ysq, ws->d_xysq, ws->d_S, ws->d_xx, ws->d_M,
              ws->d_ysq2, ws->d_X2, ws->d_Y2, ws->d_Z2, ws->d_tmp,
              ws->a_Z1sq, ws->a_Z2sq, ws->a_U1, ws->a_U2, ws->a_S1,
              ws->a_S2, ws->a_H, ws->a_R, ws->a_H2, ws->a_H3,
              ws->a_U1H2, ws->a_X3, ws->a_Y3, ws->a_Z3, ws->a_tmp,
              ws->z_Z1sq, ws->z_U2, ws->z_S2, ws->z_H, ws->z_R,
              ws->z_H2, ws->z_H3, ws->z_U1H2, ws->z_X3, ws->z_Y3,
              ws->z_Z3, ws->z_tmp, ws->z_dummy, NULL);
}
static void jac_workspace_clear(JacWorkspace *ws) {
    mpz_clears(ws->d_ysq, ws->d_xysq, ws->d_S, ws->d_xx, ws->d_M,
               ws->d_ysq2, ws->d_X2, ws->d_Y2, ws->d_Z2, ws->d_tmp,
               ws->a_Z1sq, ws->a_Z2sq, ws->a_U1, ws->a_U2, ws->a_S1,
               ws->a_S2, ws->a_H, ws->a_R, ws->a_H2, ws->a_H3,
               ws->a_U1H2, ws->a_X3, ws->a_Y3, ws->a_Z3, ws->a_tmp,
               ws->z_Z1sq, ws->z_U2, ws->z_S2, ws->z_H, ws->z_R,
               ws->z_H2, ws->z_H3, ws->z_U1H2, ws->z_X3, ws->z_Y3,
               ws->z_Z3, ws->z_tmp, ws->z_dummy, NULL);
}

static AffPoint G_PRECOMP[GBASE_N];
static AffPoint G_WINDOW_PRECOMP[WINDOW_N];

static const uint8_t _RMD_R1[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
static const uint8_t _RMD_S1[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
static const uint8_t _RMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
static const uint8_t _RMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

static const char B58_ALPHA[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int _B58_MAP[256];

static const char BECH32_CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static int _BECH32_MAP[128];

static void bech32_init_map(void) {
    memset(_BECH32_MAP, -1, sizeof(_BECH32_MAP));
    for (int i = 0; BECH32_CHARSET[i]; i++)
        _BECH32_MAP[(unsigned char)BECH32_CHARSET[i]] = i;
}

static uint32_t bech32_polymod(const uint8_t *values, int len) {
    static const uint32_t GEN[] = {0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    uint32_t chk = 1;
    for (int i = 0; i < len; i++) {
        uint8_t top = chk >> 25;
        chk = ((chk & 0x1FFFFFF) << 5) ^ values[i];
        for (int j = 0; j < 5; j++)
            if ((top >> j) & 1) chk ^= GEN[j];
    }
    return chk;
}

static int bech32_decode_witness(const char *addr, uint8_t *program, int *prog_len) {
    int addr_len = strlen(addr);
    int sep = -1;
    for (int i = addr_len - 1; i >= 0; i--)
        if (addr[i] == '1') { sep = i; break; }
    if (sep < 1 || sep + 7 > addr_len) return -1;

    int data_len = addr_len - sep - 1;
    uint8_t data[90];
    if (data_len > 90 || data_len < 6) return -1;
    for (int i = 0; i < data_len; i++) {
        char c = addr[sep + 1 + i];
        if (c >= 'A' && c <= 'Z') c += 32; /* lowercase */
        int val = (c >= 0 && c < 128) ? _BECH32_MAP[(unsigned char)c] : -1;
        if (val < 0) return -1;
        data[i] = (uint8_t)val;
    }

    uint8_t hrp_expand[20];
    int hrp_len = sep;
    for (int i = 0; i < hrp_len; i++) {
        char c = addr[i]; if (c >= 'A' && c <= 'Z') c += 32;
        hrp_expand[i] = (uint8_t)(c >> 5);
    }
    hrp_expand[hrp_len] = 0;
    int exp_len = hrp_len + 1;
    uint8_t verify_buf[120];
    for (int i = 0; i < exp_len; i++) verify_buf[i] = hrp_expand[i];
    for (int i = 0; i < hrp_len; i++) {
        char c = addr[i]; if (c >= 'A' && c <= 'Z') c += 32;
        verify_buf[exp_len + i] = (uint8_t)(c & 0x1f);
    }
    int vb_len = exp_len + hrp_len;
    for (int i = 0; i < data_len; i++) verify_buf[vb_len + i] = data[i];
    vb_len += data_len;

    uint32_t chk = bech32_polymod(verify_buf, vb_len);
    int witness_version = data[0];
    if (witness_version == 0 && chk != 1) return -1;
    if (witness_version >= 1 && chk != 0x2bc830a3) return -1;

    int five_len = data_len - 6; 
    uint8_t *five = data + 1;
    int acc = 0, bits = 0, out_idx = 0;
    for (int i = 0; i < five_len; i++) {
        acc = (acc << 5) | five[i];
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            program[out_idx++] = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    *prog_len = out_idx;

    return witness_version;
}

typedef struct {
    char inicio_hex[68];
    char fim_hex[68];
    char alvo[80];
    unsigned long long salto;       /* mantido para compat — 0 se step > 64 bits */
    unsigned long long step_valor;
    char salto_hex_full[68];       /* v4.0: step completo em hex (até 256 bits) */
    int step_idx;
    int worker_idx;
    /* v2.0: pré-computados em main() — elimina CPU work do hot path */
    uint64_t pre_x[4];     /* fe_t do ponto afim X do inicio */
    uint64_t pre_y[4];     /* fe_t do ponto afim Y do inicio */
    long long max_iter_pre; /* (fim - inicio) / salto */
    int precomputed;        /* 1 = pré-computado */
} Tarefa;

typedef struct {
    long long max_iter;    /* iterações totais para esta task */
    int step_idx;          /* índice na tabela de potências */
    int worker_idx;        /* índice nos pontos iniciais */
    int k;                 /* nível de potência para init (0..7) */
    int orig_task_idx;     /* índice original em task_queue (para key recovery) */
} GPUTaskInfo;

#define MAX_PERSISTENT_BLOCKS 4096
#define STATUS_UPDATE_INTERVAL 256  /* batches entre atualizações de status */

typedef struct {
    volatile long long task_id;    /* task atual (-1 = idle/done) — 64-bit para tasks > 2G */
    volatile int    step_idx;      /* step_idx da task atual */
    volatile int    worker_idx;    /* worker_idx da task atual */
    volatile int    actual_N;      /* threads ativas neste bloco */
    volatile int    _pad;          /* alinha o struct em 8 bytes */
    volatile long long loops_done; /* progresso dentro da task */
    volatile long long max_iter;   /* total de iterações da task */
    volatile long long keys_total; /* chaves verificadas acumuladas */
    volatile long long resume_m;   /* v10.17: rodada m onde retomar dentro do step exaustivo */
} GPUBlockStatus;

static long long task_count = 0; /* = num_steps × num_workers */

/* Contador de iterações de 128 bits: walk exaustivo percorre até 2^128-1 iterações
 * por step (vs 2^63 antes). ITERS_NONFINITE = valor de saturação que o host grava
 * quando ceil(range/salto) excede 2^128-1 — o kernel trata como "não-finito" e
 * jamais marca o step como concluído. */
#define ITERS_NONFINITE  (~(unsigned __int128)0)   /* 2^128 - 1 */
__host__ __device__ __forceinline__ long long clamp_u128_to_i64(unsigned __int128 v) {
    /* satura para LLONG_MAX ao gravar nos campos long long do display — preserva
     * o layout/formatos do display sem truncar com lixo para ranges enormes. */
    const unsigned __int128 LIM = (unsigned __int128)0x7FFFFFFFFFFFFFFFLL;
    return (v > LIM) ? 0x7FFFFFFFFFFFFFFFLL : (long long)v;
}

static pthread_mutex_t stdout_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t found_mutex  = PTHREAD_MUTEX_INITIALIZER;

static volatile int evento_conclusao = 0;

static int g_display_offset = 0;  /* bloco inicial da janela de display */

static int g_addr_type = ADDR_TYPE_P2PKH;
static int g_gpu_mode  = GPU_MODE_HASH160;

/* ── MULTI-ALVO v11.0: arrays dinâmicos + prefix table p/ lookup O(1) ─
 *
 * v10.13 tinha 3 gargalos hard-cap em 256 alvos:
 *   1) __constant__ d_alvos_u32[256][5] — 64KB total disponível
 *   2) __device__   d_alvos_ativos[256] — array estático
 *   3) for-loop linear no hot path      — O(N) por hash check
 *
 * v11.0 resolve os 3:
 *   1) → cudaMalloc em global memory (sem limite prático)
 *   2) → cudaMalloc dimensionado em runtime
 *   3) → prefix table de 16M entradas indexada pelos 24 bits superiores
 *        de h0; bucket size médio ≤1.25 alvo p/ 20M alvos → O(1) lookup.
 *
 * Custo VRAM extra (20M alvos): 64MB prefix + 480MB sorted + 80MB ativos
 *                              = ~624MB (dentro do 16GB do 5060 Ti).
 * ──────────────────────────────────────────────────────────────────── */

/* Cap só de sanidade — não tem mais ligação com hardware. */
#define MAX_ALVOS_HARD_CAP   100000000u   /* 100M */
#define ALVOS_PREFIX_BITS    24u
#define ALVOS_PREFIX_SIZE    (1u << ALVOS_PREFIX_BITS)  /* 16,777,216 buckets */

/* Entry compactada (24 bytes) para o array sorted no device.
 * orig_idx mapeia de volta pro índice na lista de alvos do host. */
typedef struct __align__(8) {
    uint32_t h0, h1, h2, h3, h4;
    uint32_t orig_idx;
} AlvoEntry;

static char     (*g_alvos_lista)[80]   = NULL;  /* [g_num_alvos] strings */
static uint8_t  (*g_alvos_hash)[32]    = NULL;  /* [g_num_alvos] hash160 bytes */
static int       *g_alvos_ativos_host  = NULL;  /* [g_num_alvos] mirror host */
static AlvoEntry *g_alvos_sorted       = NULL;  /* [g_num_alvos] sorted by h0 */
static uint32_t  *g_alvos_prefix       = NULL;  /* [ALVOS_PREFIX_SIZE+1] bucket bounds */
static int        g_num_alvos          = 0;
static int        g_alvos_capacity     = 0;     /* tamanho alocado p/ realloc */

/* Ponteiros device guardados no host (retornados por cudaMalloc).
 * Usados para cudaMemcpy ao atualizar (ex: marcar alvo como encontrado). */
static AlvoEntry *g_d_alvos_sorted_ptr = NULL;
static uint32_t  *g_d_alvos_prefix_ptr = NULL;
static int       *g_d_alvos_ativos_ptr = NULL;

typedef uint64_t fe_t_fwd[4];  /* forward decl — mesmo layout que fe_t */
typedef char     worker_hex_t[260];
static worker_hex_t *g_worker_inicio_hex = NULL;  /* [NUM_WORKERS] */
static worker_hex_t *g_worker_fim_hex = NULL;
static fe_t_fwd *g_worker_start_x = NULL;   /* [NUM_WORKERS] */
static fe_t_fwd *g_worker_start_y = NULL;

static unsigned __int128 *g_step_max_iter = NULL; /* [num_steps]: iters per step (128 bits: walk exaustivo até 2^128) */

static int            num_workers_global = 0;
static int            num_steps_total = 0;

static long long      g_ckpt_tasks_done = 0;  /* checkpoint: how many tasks completed */

static pthread_mutex_t ckpt_mutex = PTHREAD_MUTEX_INITIALIZER;

/* v10.14 forward decls: definidas mais abaixo, usadas em save/load_checkpoint */
extern int *g_h_worker_next_si;
extern int g_persistent_blocks;

static void set_evento(void) {
    __sync_synchronize();
    evento_conclusao = 1;
    __sync_synchronize();
}
static int get_evento(void) {
    __sync_synchronize();
    return evento_conclusao;
}

static void mpz_export_32be(const mpz_t n, uint8_t *buf32) {
    size_t count = (mpz_sizeinbase(n, 2) + 7) / 8;
    if (count > 32) count = 32;
    size_t off = 32 - count;
    if (off) memset(buf32, 0, off);
    size_t actual;
    mpz_export(buf32 + off, &actual, 1, 1, 1, 0, n);
    if (actual < count) {
        memmove(buf32 + 32 - actual, buf32 + off, actual);
        memset(buf32, 0, 32 - actual);
    }
}

static void save_checkpoint(void) {
    pthread_mutex_lock(&ckpt_mutex);

    /* Header em texto: parâmetros + soma de progresso (para display) */
    FILE *f = fopen(CHECKPOINT_FILE ".tmp", "w");
    if (!f) { pthread_mutex_unlock(&ckpt_mutex); return; }
    fprintf(f, "num_workers %d\n", num_workers_global);
    fprintf(f, "num_steps %d\n", num_steps_total);
    fprintf(f, "task_count %lld\n", task_count);
    fprintf(f, "tasks_done %lld\n", g_ckpt_tasks_done);
    /* v10.14: progresso per-worker. Salva o array inteiro como tabela ASCII.
     * Cada linha: "worker_next_si <bid> <next_si>" */
    if (g_h_worker_next_si && g_persistent_blocks > 0) {
        fprintf(f, "worker_next_si_count %d\n", g_persistent_blocks);
        for (int bid = 0; bid < g_persistent_blocks; bid++) {
            fprintf(f, "wns %d %d\n", bid, g_h_worker_next_si[bid]);
        }
    }
    fclose(f);
    rename(CHECKPOINT_FILE ".tmp", CHECKPOINT_FILE);
    pthread_mutex_unlock(&ckpt_mutex);

    /* Calcula soma e min/max para display informativo */
    long long sum = 0;
    int min_si = INT_MAX, max_si = INT_MIN;
    if (g_h_worker_next_si) {
        for (int bid = 0; bid < g_persistent_blocks; bid++) {
            sum += g_h_worker_next_si[bid];
            if (g_h_worker_next_si[bid] < min_si) min_si = g_h_worker_next_si[bid];
            if (g_h_worker_next_si[bid] > max_si) max_si = g_h_worker_next_si[bid];
        }
    }
    printf("[Checkpoint] Salvo: %lld tasks concluidas, workers next_si min=%d max=%d (de %d steps).\n",
           sum, min_si == INT_MAX ? 0 : min_si, max_si == INT_MIN ? 0 : max_si, num_steps_total);
}

static int load_checkpoint(void) {
    FILE *f = fopen(CHECKPOINT_FILE, "r");
    if (!f) return 0;
    char line[512];
    int saved_nw = 0, saved_ns = 0;
    long long saved_tc = 0, saved_done = 0;
    int saved_wns_count = 0;
    /* Buffer temporário para próximos_si lidos (até persistent_blocks) */
    int wns_temp[8192]; /* MAX_PERSISTENT_BLOCKS hard cap */
    for (int i = 0; i < 8192; i++) wns_temp[i] = 0;
    int wns_loaded = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "num_workers ", 12) == 0) saved_nw = atoi(line + 12);
        else if (strncmp(line, "num_steps ", 10) == 0) saved_ns = atoi(line + 10);
        else if (strncmp(line, "task_count ", 11) == 0) saved_tc = atoll(line + 11);
        else if (strncmp(line, "tasks_done ", 11) == 0) saved_done = atoll(line + 11);
        else if (strncmp(line, "worker_next_si_count ", 21) == 0) saved_wns_count = atoi(line + 21);
        else if (strncmp(line, "wns ", 4) == 0) {
            int bid, nsi;
            if (sscanf(line + 4, "%d %d", &bid, &nsi) == 2) {
                if (bid >= 0 && bid < 8192) {
                    wns_temp[bid] = nsi;
                    if (bid >= wns_loaded) wns_loaded = bid + 1;
                }
            }
        }
    }
    fclose(f);
    if (saved_nw != num_workers_global || saved_ns != num_steps_total ||
        saved_tc != task_count) {
        printf("[Checkpoint] Parâmetros diferentes — iniciando do zero.\n");
        return 0;
    }
    g_ckpt_tasks_done = saved_done;

    /* Copia para o array per-worker, se já alocado */
    if (g_h_worker_next_si && saved_wns_count > 0) {
        int n = (saved_wns_count < g_persistent_blocks) ? saved_wns_count : g_persistent_blocks;
        for (int bid = 0; bid < n; bid++) {
            g_h_worker_next_si[bid] = wns_temp[bid];
        }
        /* Workers extras (se persistent_blocks > saved_wns_count): inicia do zero */
        for (int bid = saved_wns_count; bid < g_persistent_blocks; bid++) {
            g_h_worker_next_si[bid] = 0;
        }
        int min_si = INT_MAX, max_si = INT_MIN;
        for (int bid = 0; bid < g_persistent_blocks; bid++) {
            if (g_h_worker_next_si[bid] < min_si) min_si = g_h_worker_next_si[bid];
            if (g_h_worker_next_si[bid] > max_si) max_si = g_h_worker_next_si[bid];
        }
        printf("[Checkpoint] Carregado: %d workers, next_si min=%d max=%d (de %d steps).\n",
               g_persistent_blocks, min_si, max_si, num_steps_total);
    } else {
        printf("[Checkpoint] Carregado (legado, sem per-worker): %lld/%lld tasks.\n", saved_done, task_count);
    }
    return 1;
}

static void *checkpoint_thread_func(void *arg) {
    (void)arg;
    struct timespec last_save;
    clock_gettime(CLOCK_MONOTONIC, &last_save);
    while (!get_evento()) {
        sched_yield();
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((int)(now.tv_sec - last_save.tv_sec) >= CHECKPOINT_INTERVAL_S) {
            save_checkpoint(); clock_gettime(CLOCK_MONOTONIC, &last_save);
        }
    }
    return NULL;
}

static void format_commas(long long n, char *buf) {
    if (n < 0) { buf[0] = '-'; format_commas(-n, buf + 1); return; }
    char tmp[64]; int len = snprintf(tmp, sizeof(tmp), "%lld", n);
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) buf[j++] = ',';
        buf[j++] = tmp[i];
    }
    buf[j] = '\0';
}

static void mpz_to_hex_zfill(const mpz_t n, char *buf, int width) {
    char tmp[300]; mpz_get_str(tmp, 16, n);
    int len = (int)strlen(tmp); int pad = width - len;
    if (pad > 0) { memset(buf, '0', pad); memcpy(buf + pad, tmp, len + 1); }
    else          { memcpy(buf, tmp, len + 1); }
}

static void sha256(const uint8_t *dados_bytes, size_t len, uint8_t *out32) {
    unsigned int md_len = 32;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, dados_bytes, len);
    EVP_DigestFinal_ex(ctx, out32, &md_len);
    EVP_MD_CTX_free(ctx);
}

static const uint32_t _SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
__constant__ uint32_t d_SHA256_K[64];

#define _S256_ROTR(x,n) (((uint32_t)(x)>>(n))|((uint32_t)(x)<<(32-(n))))
#define _S256_CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define _S256_MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define _S256_EP0(a) (_S256_ROTR(a,2) ^_S256_ROTR(a,13)^_S256_ROTR(a,22))
#define _S256_EP1(e) (_S256_ROTR(e,6) ^_S256_ROTR(e,11)^_S256_ROTR(e,25))
#define _S256_SG0(x) (_S256_ROTR(x,7) ^_S256_ROTR(x,18)^((x)>>3))
#define _S256_SG1(x) (_S256_ROTR(x,17)^_S256_ROTR(x,19)^((x)>>10))

#define _S256_R(a,b,c,d,e,f,g,h,k,w) do{ \
    uint32_t _t1=(h)+_S256_EP1(e)+_S256_CH(e,f,g)+(k)+(w); \
    uint32_t _t2=_S256_EP0(a)+_S256_MAJ(a,b,c); \
    (d)+=_t1; (h)=_t1+_t2; }while(0)

#define _S256_R8(i,W) \
    _S256_R(a,b,c,d,e,f,g,h,_SHA256_K[(i)+0],(W)[(i)+0]); \
    _S256_R(h,a,b,c,d,e,f,g,_SHA256_K[(i)+1],(W)[(i)+1]); \
    _S256_R(g,h,a,b,c,d,e,f,_SHA256_K[(i)+2],(W)[(i)+2]); \
    _S256_R(f,g,h,a,b,c,d,e,_SHA256_K[(i)+3],(W)[(i)+3]); \
    _S256_R(e,f,g,h,a,b,c,d,_SHA256_K[(i)+4],(W)[(i)+4]); \
    _S256_R(d,e,f,g,h,a,b,c,_SHA256_K[(i)+5],(W)[(i)+5]); \
    _S256_R(c,d,e,f,g,h,a,b,_SHA256_K[(i)+6],(W)[(i)+6]); \
    _S256_R(b,c,d,e,f,g,h,a,_SHA256_K[(i)+7],(W)[(i)+7]);

static void sha256_33b(const uint8_t *pk33, uint8_t *out32) {
    uint32_t W[64];
    W[0]=((uint32_t)pk33[0]<<24)|((uint32_t)pk33[1]<<16)|((uint32_t)pk33[2]<<8)|(uint32_t)pk33[3];
    W[1]=((uint32_t)pk33[4]<<24)|((uint32_t)pk33[5]<<16)|((uint32_t)pk33[6]<<8)|(uint32_t)pk33[7];
    W[2]=((uint32_t)pk33[8]<<24)|((uint32_t)pk33[9]<<16)|((uint32_t)pk33[10]<<8)|(uint32_t)pk33[11];
    W[3]=((uint32_t)pk33[12]<<24)|((uint32_t)pk33[13]<<16)|((uint32_t)pk33[14]<<8)|(uint32_t)pk33[15];
    W[4]=((uint32_t)pk33[16]<<24)|((uint32_t)pk33[17]<<16)|((uint32_t)pk33[18]<<8)|(uint32_t)pk33[19];
    W[5]=((uint32_t)pk33[20]<<24)|((uint32_t)pk33[21]<<16)|((uint32_t)pk33[22]<<8)|(uint32_t)pk33[23];
    W[6]=((uint32_t)pk33[24]<<24)|((uint32_t)pk33[25]<<16)|((uint32_t)pk33[26]<<8)|(uint32_t)pk33[27];
    W[7]=((uint32_t)pk33[28]<<24)|((uint32_t)pk33[29]<<16)|((uint32_t)pk33[30]<<8)|(uint32_t)pk33[31];
    W[8]=((uint32_t)pk33[32]<<24)|0x00800000u;
    W[9]=W[10]=W[11]=W[12]=W[13]=W[14]=0u; W[15]=264u;
    for(int i=16;i<64;i++) W[i]=_S256_SG1(W[i-2])+W[i-7]+_S256_SG0(W[i-15])+W[i-16];
    uint32_t a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au,
             e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    _S256_R8(0,W);_S256_R8(8,W);_S256_R8(16,W);_S256_R8(24,W);
    _S256_R8(32,W);_S256_R8(40,W);_S256_R8(48,W);_S256_R8(56,W);
    uint32_t H[8]={0x6a09e667u+a,0xbb67ae85u+b,0x3c6ef372u+c,0xa54ff53au+d,
                   0x510e527fu+e,0x9b05688cu+f,0x1f83d9abu+g,0x5be0cd19u+h};
    for(int i=0;i<8;i++){
        out32[i*4]=(uint8_t)(H[i]>>24);out32[i*4+1]=(uint8_t)(H[i]>>16);
        out32[i*4+2]=(uint8_t)(H[i]>>8);out32[i*4+3]=(uint8_t)(H[i]);
    }
}

#define ROTL32(x,n) (((uint32_t)(x)<<(n))|((uint32_t)(x)>>(32-(n))))

#define RL1(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+((B)^(C)^(D))+X[xi],(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RL2(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)&(C))|(~(B)&(D)))+X[xi]+0x5A827999u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RL3(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)|(~(C)))^(D))+X[xi]+0x6ED9EBA1u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RL4(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)&(D))|((C)&~(D)))+X[xi]+0x8F1BBCDCu,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RL5(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+((B)^((C)|(~(D))))+X[xi]+0xA953FD4Eu,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RR1(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+((B)^((C)|(~(D))))+X[xi]+0x50A28BE6u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RR2(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)&(D))|((C)&~(D)))+X[xi]+0x5C4DD124u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RR3(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)|(~(C)))^(D))+X[xi]+0x6D703EF3u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RR4(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+(((B)&(C))|(~(B)&(D)))+X[xi]+0x7A6D76E9u,(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)
#define RR5(A,B,C,D,E,xi,si) do{ uint32_t _T=ROTL32((A)+((B)^(C)^(D))+X[xi],(si))+(E); (A)=(E);(E)=(D);(D)=ROTL32((C),10);(C)=(B);(B)=_T; }while(0)

#define RMD160_COMPRESS_BLOCK(h0,h1,h2,h3,h4,X) do{ \
    uint32_t A=(h0),B=(h1),C=(h2),D=(h3),E=(h4); \
    uint32_t Ap=(h0),Bp=(h1),Cp=(h2),Dp=(h3),Ep=(h4); \
    RL1(A,B,C,D,E, 0,11); RR1(Ap,Bp,Cp,Dp,Ep, 5, 8); RL1(A,B,C,D,E, 1,14); RR1(Ap,Bp,Cp,Dp,Ep,14, 9); \
    RL1(A,B,C,D,E, 2,15); RR1(Ap,Bp,Cp,Dp,Ep, 7, 9); RL1(A,B,C,D,E, 3,12); RR1(Ap,Bp,Cp,Dp,Ep, 0,11); \
    RL1(A,B,C,D,E, 4, 5); RR1(Ap,Bp,Cp,Dp,Ep, 9,13); RL1(A,B,C,D,E, 5, 8); RR1(Ap,Bp,Cp,Dp,Ep, 2,15); \
    RL1(A,B,C,D,E, 6, 7); RR1(Ap,Bp,Cp,Dp,Ep,11,15); RL1(A,B,C,D,E, 7, 9); RR1(Ap,Bp,Cp,Dp,Ep, 4, 5); \
    RL1(A,B,C,D,E, 8,11); RR1(Ap,Bp,Cp,Dp,Ep,13, 7); RL1(A,B,C,D,E, 9,13); RR1(Ap,Bp,Cp,Dp,Ep, 6, 7); \
    RL1(A,B,C,D,E,10,14); RR1(Ap,Bp,Cp,Dp,Ep,15, 8); RL1(A,B,C,D,E,11,15); RR1(Ap,Bp,Cp,Dp,Ep, 8,11); \
    RL1(A,B,C,D,E,12, 6); RR1(Ap,Bp,Cp,Dp,Ep, 1,14); RL1(A,B,C,D,E,13, 7); RR1(Ap,Bp,Cp,Dp,Ep,10,14); \
    RL1(A,B,C,D,E,14, 9); RR1(Ap,Bp,Cp,Dp,Ep, 3,12); RL1(A,B,C,D,E,15, 8); RR1(Ap,Bp,Cp,Dp,Ep,12, 6); \
    RL2(A,B,C,D,E, 7, 7); RR2(Ap,Bp,Cp,Dp,Ep, 6, 9); RL2(A,B,C,D,E, 4, 6); RR2(Ap,Bp,Cp,Dp,Ep,11,13); \
    RL2(A,B,C,D,E,13, 8); RR2(Ap,Bp,Cp,Dp,Ep, 3,15); RL2(A,B,C,D,E, 1,13); RR2(Ap,Bp,Cp,Dp,Ep, 7, 7); \
    RL2(A,B,C,D,E,10,11); RR2(Ap,Bp,Cp,Dp,Ep, 0,12); RL2(A,B,C,D,E, 6, 9); RR2(Ap,Bp,Cp,Dp,Ep,13, 8); \
    RL2(A,B,C,D,E,15, 7); RR2(Ap,Bp,Cp,Dp,Ep, 5, 9); RL2(A,B,C,D,E, 3,15); RR2(Ap,Bp,Cp,Dp,Ep,10,11); \
    RL2(A,B,C,D,E,12, 7); RR2(Ap,Bp,Cp,Dp,Ep,14, 7); RL2(A,B,C,D,E, 0,12); RR2(Ap,Bp,Cp,Dp,Ep,15, 7); \
    RL2(A,B,C,D,E, 9,15); RR2(Ap,Bp,Cp,Dp,Ep, 8,12); RL2(A,B,C,D,E, 5, 9); RR2(Ap,Bp,Cp,Dp,Ep,12, 7); \
    RL2(A,B,C,D,E, 2,11); RR2(Ap,Bp,Cp,Dp,Ep, 4, 6); RL2(A,B,C,D,E,14, 7); RR2(Ap,Bp,Cp,Dp,Ep, 9,15); \
    RL2(A,B,C,D,E,11,13); RR2(Ap,Bp,Cp,Dp,Ep, 1,13); RL2(A,B,C,D,E, 8,12); RR2(Ap,Bp,Cp,Dp,Ep, 2,11); \
    RL3(A,B,C,D,E, 3,11); RR3(Ap,Bp,Cp,Dp,Ep,15, 9); RL3(A,B,C,D,E,10,13); RR3(Ap,Bp,Cp,Dp,Ep, 5, 7); \
    RL3(A,B,C,D,E,14, 6); RR3(Ap,Bp,Cp,Dp,Ep, 1,15); RL3(A,B,C,D,E, 4, 7); RR3(Ap,Bp,Cp,Dp,Ep, 3,11); \
    RL3(A,B,C,D,E, 9,14); RR3(Ap,Bp,Cp,Dp,Ep, 7, 8); RL3(A,B,C,D,E,15, 9); RR3(Ap,Bp,Cp,Dp,Ep,14, 6); \
    RL3(A,B,C,D,E, 8,13); RR3(Ap,Bp,Cp,Dp,Ep, 6, 6); RL3(A,B,C,D,E, 1,15); RR3(Ap,Bp,Cp,Dp,Ep, 9,14); \
    RL3(A,B,C,D,E, 2,14); RR3(Ap,Bp,Cp,Dp,Ep,11,12); RL3(A,B,C,D,E, 7, 8); RR3(Ap,Bp,Cp,Dp,Ep, 8,13); \
    RL3(A,B,C,D,E, 0,13); RR3(Ap,Bp,Cp,Dp,Ep,12, 5); RL3(A,B,C,D,E, 6, 6); RR3(Ap,Bp,Cp,Dp,Ep, 2,14); \
    RL3(A,B,C,D,E,13, 5); RR3(Ap,Bp,Cp,Dp,Ep,10,13); RL3(A,B,C,D,E,11,12); RR3(Ap,Bp,Cp,Dp,Ep, 0,13); \
    RL3(A,B,C,D,E, 5, 7); RR3(Ap,Bp,Cp,Dp,Ep, 4, 7); RL3(A,B,C,D,E,12, 5); RR3(Ap,Bp,Cp,Dp,Ep,13, 5); \
    RL4(A,B,C,D,E, 1,11); RR4(Ap,Bp,Cp,Dp,Ep, 8,15); RL4(A,B,C,D,E, 9,12); RR4(Ap,Bp,Cp,Dp,Ep, 6, 5); \
    RL4(A,B,C,D,E,11,14); RR4(Ap,Bp,Cp,Dp,Ep, 4, 8); RL4(A,B,C,D,E,10,15); RR4(Ap,Bp,Cp,Dp,Ep, 1,11); \
    RL4(A,B,C,D,E, 0,14); RR4(Ap,Bp,Cp,Dp,Ep, 3,14); RL4(A,B,C,D,E, 8,15); RR4(Ap,Bp,Cp,Dp,Ep,11,14); \
    RL4(A,B,C,D,E,12, 9); RR4(Ap,Bp,Cp,Dp,Ep,15, 6); RL4(A,B,C,D,E, 4, 8); RR4(Ap,Bp,Cp,Dp,Ep, 0,14); \
    RL4(A,B,C,D,E,13, 9); RR4(Ap,Bp,Cp,Dp,Ep, 5, 6); RL4(A,B,C,D,E, 3,14); RR4(Ap,Bp,Cp,Dp,Ep,12, 9); \
    RL4(A,B,C,D,E, 7, 5); RR4(Ap,Bp,Cp,Dp,Ep, 2,12); RL4(A,B,C,D,E,15, 6); RR4(Ap,Bp,Cp,Dp,Ep,13, 9); \
    RL4(A,B,C,D,E,14, 8); RR4(Ap,Bp,Cp,Dp,Ep, 9,12); RL4(A,B,C,D,E, 5, 6); RR4(Ap,Bp,Cp,Dp,Ep, 7, 5); \
    RL4(A,B,C,D,E, 6, 5); RR4(Ap,Bp,Cp,Dp,Ep,10,15); RL4(A,B,C,D,E, 2,12); RR4(Ap,Bp,Cp,Dp,Ep,14, 8); \
    RL5(A,B,C,D,E, 4, 9); RR5(Ap,Bp,Cp,Dp,Ep,12, 8); RL5(A,B,C,D,E, 0,15); RR5(Ap,Bp,Cp,Dp,Ep,15, 5); \
    RL5(A,B,C,D,E, 5, 5); RR5(Ap,Bp,Cp,Dp,Ep,10,12); RL5(A,B,C,D,E, 9,11); RR5(Ap,Bp,Cp,Dp,Ep, 4, 9); \
    RL5(A,B,C,D,E, 7, 6); RR5(Ap,Bp,Cp,Dp,Ep, 1,12); RL5(A,B,C,D,E,12, 8); RR5(Ap,Bp,Cp,Dp,Ep, 5, 5); \
    RL5(A,B,C,D,E, 2,13); RR5(Ap,Bp,Cp,Dp,Ep, 8,14); RL5(A,B,C,D,E,10,12); RR5(Ap,Bp,Cp,Dp,Ep, 7, 6); \
    RL5(A,B,C,D,E,14, 5); RR5(Ap,Bp,Cp,Dp,Ep, 6, 8); RL5(A,B,C,D,E, 1,12); RR5(Ap,Bp,Cp,Dp,Ep, 2,13); \
    RL5(A,B,C,D,E, 3,13); RR5(Ap,Bp,Cp,Dp,Ep,13, 6); RL5(A,B,C,D,E, 8,14); RR5(Ap,Bp,Cp,Dp,Ep,14, 5); \
    RL5(A,B,C,D,E,11,11); RR5(Ap,Bp,Cp,Dp,Ep, 0,15); RL5(A,B,C,D,E, 6, 8); RR5(Ap,Bp,Cp,Dp,Ep, 3,13); \
    RL5(A,B,C,D,E,15, 5); RR5(Ap,Bp,Cp,Dp,Ep, 9,11); RL5(A,B,C,D,E,13, 6); RR5(Ap,Bp,Cp,Dp,Ep,11,11); \
    uint32_t _tt=(h1)+C+Dp; (h1)=(h2)+D+Ep; (h2)=(h3)+E+Ap; (h3)=(h4)+A+Bp; (h4)=(h0)+B+Cp; (h0)=_tt; \
}while(0)

static void ripemd160(const uint8_t *msg, size_t len, uint8_t *out20) {
    uint32_t h0=0x67452301u,h1=0xEFCDAB89u,h2=0x98BADCFEu,h3=0x10325476u,h4=0xC3D2E1F0u;
    uint32_t X[16];
    if (len == 32) {
        X[0]=(uint32_t)msg[0]|((uint32_t)msg[1]<<8)|((uint32_t)msg[2]<<16)|((uint32_t)msg[3]<<24);
        X[1]=(uint32_t)msg[4]|((uint32_t)msg[5]<<8)|((uint32_t)msg[6]<<16)|((uint32_t)msg[7]<<24);
        X[2]=(uint32_t)msg[8]|((uint32_t)msg[9]<<8)|((uint32_t)msg[10]<<16)|((uint32_t)msg[11]<<24);
        X[3]=(uint32_t)msg[12]|((uint32_t)msg[13]<<8)|((uint32_t)msg[14]<<16)|((uint32_t)msg[15]<<24);
        X[4]=(uint32_t)msg[16]|((uint32_t)msg[17]<<8)|((uint32_t)msg[18]<<16)|((uint32_t)msg[19]<<24);
        X[5]=(uint32_t)msg[20]|((uint32_t)msg[21]<<8)|((uint32_t)msg[22]<<16)|((uint32_t)msg[23]<<24);
        X[6]=(uint32_t)msg[24]|((uint32_t)msg[25]<<8)|((uint32_t)msg[26]<<16)|((uint32_t)msg[27]<<24);
        X[7]=(uint32_t)msg[28]|((uint32_t)msg[29]<<8)|((uint32_t)msg[30]<<16)|((uint32_t)msg[31]<<24);
        X[8]=0x00000080u;X[9]=0;X[10]=0;X[11]=0;X[12]=0;X[13]=0;X[14]=256u;X[15]=0;
        RMD160_COMPRESS_BLOCK(h0,h1,h2,h3,h4,X);
        uint32_t _out[5]={h0,h1,h2,h3,h4}; memcpy(out20,_out,20); return;
    }
    uint8_t buf[128]; uint64_t bits=(uint64_t)len*8;
    size_t padded=len+1; while(padded%64!=56) padded++; padded+=8;
    memset(buf,0,padded); memcpy(buf,msg,len); buf[len]=0x80;
    for(int i=0;i<8;i++) buf[padded-8+i]=(uint8_t)((bits>>(8*i))&0xFF);
    size_t num_blocks=padded/64;
    for(size_t blk=0;blk<num_blocks;blk++){
        const uint8_t *b=buf+blk*64;
        for(int j=0;j<16;j++)
            X[j]=(uint32_t)b[j*4]|((uint32_t)b[j*4+1]<<8)|((uint32_t)b[j*4+2]<<16)|((uint32_t)b[j*4+3]<<24);
        RMD160_COMPRESS_BLOCK(h0,h1,h2,h3,h4,X);
    }
    uint32_t _out[5]={h0,h1,h2,h3,h4}; memcpy(out20,_out,20);
}

static void _jac_double(JacPoint *result, const JacPoint *pt, JacWorkspace *ws) {
    if(mpz_sgn(pt->Y)==0||mpz_sgn(pt->Z)==0){mpz_set_ui(result->X,0);mpz_set_ui(result->Y,1);mpz_set_ui(result->Z,0);return;}
    mpz_t *ysq=&ws->d_ysq,*xysq=&ws->d_xysq,*S=&ws->d_S,*xx=&ws->d_xx,
          *M=&ws->d_M,*ysq2=&ws->d_ysq2,*X2=&ws->d_X2,*Y2=&ws->d_Y2,*Z2=&ws->d_Z2,*tmp=&ws->d_tmp;
    mpz_mul(*ysq,pt->Y,pt->Y);mpz_mod(*ysq,*ysq,P_val);
    mpz_mul(*xysq,pt->X,*ysq);mpz_mod(*xysq,*xysq,P_val);
    mpz_mul_2exp(*S,*xysq,2);mpz_mod(*S,*S,P_val);
    mpz_mul(*xx,pt->X,pt->X);mpz_mod(*xx,*xx,P_val);
    mpz_mul_ui(*M,*xx,3);mpz_mod(*M,*M,P_val);
    mpz_mul(*ysq2,*ysq,*ysq);mpz_mod(*ysq2,*ysq2,P_val);
    mpz_mul(*X2,*M,*M);mpz_submul_ui(*X2,*S,2);mpz_mod(*X2,*X2,P_val);
    mpz_sub(*tmp,*S,*X2);mpz_mul(*Y2,*M,*tmp);mpz_submul_ui(*Y2,*ysq2,8);mpz_mod(*Y2,*Y2,P_val);
    mpz_mul(*Z2,pt->Y,pt->Z);mpz_mul_2exp(*Z2,*Z2,1);mpz_mod(*Z2,*Z2,P_val);
    mpz_set(result->X,*X2);mpz_set(result->Y,*Y2);mpz_set(result->Z,*Z2);
}

static void _jac_add_z2_one(JacPoint *result, const JacPoint *p1,
                             const mpz_t X2, const mpz_t Y2, JacWorkspace *ws) {
    if(mpz_sgn(p1->Z)==0){mpz_set(result->X,X2);mpz_set(result->Y,Y2);mpz_set_ui(result->Z,1);return;}
    mpz_t *Z1sq=&ws->z_Z1sq,*U2=&ws->z_U2,*S2=&ws->z_S2,*H=&ws->z_H,*R=&ws->z_R,
          *H2=&ws->z_H2,*H3=&ws->z_H3,*U1H2=&ws->z_U1H2,*X3=&ws->z_X3,*Y3=&ws->z_Y3,*Z3=&ws->z_Z3,*tmp=&ws->z_tmp;
    mpz_mul(*Z1sq,p1->Z,p1->Z);mpz_mod(*Z1sq,*Z1sq,P_val);
    mpz_mul(*U2,X2,*Z1sq);mpz_mod(*U2,*U2,P_val);
    mpz_mul(*S2,Y2,*Z1sq);mpz_mul(*S2,*S2,p1->Z);mpz_mod(*S2,*S2,P_val);
    mpz_sub(*H,*U2,p1->X);if(mpz_sgn(*H)<0)mpz_add(*H,*H,P_val);
    mpz_sub(*R,*S2,p1->Y);if(mpz_sgn(*R)<0)mpz_add(*R,*R,P_val);
    if(mpz_sgn(*H)==0){
        if(mpz_sgn(*R)==0){JacPoint tp;mpz_init_set(tp.X,X2);mpz_init_set(tp.Y,Y2);mpz_init_set_ui(tp.Z,1);
            _jac_double(result,&tp,ws);mpz_clears(tp.X,tp.Y,tp.Z,NULL);}
        else{mpz_set_ui(result->X,0);mpz_set_ui(result->Y,1);mpz_set_ui(result->Z,0);}return;}
    mpz_mul(*H2,*H,*H);mpz_mod(*H2,*H2,P_val);
    mpz_mul(*H3,*H,*H2);mpz_mod(*H3,*H3,P_val);
    mpz_mul(*U1H2,p1->X,*H2);mpz_mod(*U1H2,*U1H2,P_val);
    mpz_mul(*X3,*R,*R);mpz_sub(*X3,*X3,*H3);mpz_submul_ui(*X3,*U1H2,2);mpz_mod(*X3,*X3,P_val);
    mpz_sub(*tmp,*U1H2,*X3);mpz_mul(*Y3,*R,*tmp);mpz_submul(*Y3,p1->Y,*H3);mpz_mod(*Y3,*Y3,P_val);
    mpz_mul(*Z3,*H,p1->Z);mpz_mod(*Z3,*Z3,P_val);
    mpz_set(result->X,*X3);mpz_set(result->Y,*Y3);mpz_set(result->Z,*Z3);
}

static void _precompute_gbase(void) {
    JacWorkspace ws; jac_workspace_init(&ws);
    JacPoint *pts_j=(JacPoint*)malloc(GBASE_N*sizeof(JacPoint));
    for(int i=0;i<GBASE_N;i++){mpz_init(pts_j[i].X);mpz_init(pts_j[i].Y);mpz_init(pts_j[i].Z);}
    mpz_set(pts_j[0].X,G_x_val);mpz_set(pts_j[0].Y,G_y_val);mpz_set_ui(pts_j[0].Z,1);
    for(int i=1;i<GBASE_N;i++) _jac_double(&pts_j[i],&pts_j[i-1],&ws);
    mpz_t *pre=( mpz_t*)malloc(GBASE_N*sizeof(mpz_t)),*zinv=(mpz_t*)malloc(GBASE_N*sizeof(mpz_t));
    for(int i=0;i<GBASE_N;i++){mpz_init(pre[i]);mpz_init(zinv[i]);}
    mpz_set_ui(pre[0],1);
    for(int i=1;i<GBASE_N;i++){mpz_mul(pre[i],pre[i-1],pts_j[i-1].Z);mpz_mod(pre[i],pre[i],P_val);}
    mpz_t inv,tmp; mpz_init(inv);mpz_init(tmp);
    mpz_mul(tmp,pre[GBASE_N-1],pts_j[GBASE_N-1].Z);mpz_mod(tmp,tmp,P_val);
    mpz_powm(inv,tmp,P_minus_2,P_val);
    for(int i=GBASE_N-1;i>=0;i--){mpz_mul(zinv[i],inv,pre[i]);mpz_mod(zinv[i],zinv[i],P_val);mpz_mul(inv,inv,pts_j[i].Z);mpz_mod(inv,inv,P_val);}
    mpz_clears(inv,tmp,NULL);
    for(int i=0;i<GBASE_N;i++){
        mpz_t zi,zi2;mpz_init_set(zi,zinv[i]);mpz_init(zi2);
        mpz_mul(zi2,zi,zi);mpz_mod(zi2,zi2,P_val);
        mpz_init(G_PRECOMP[i].x);mpz_init(G_PRECOMP[i].y);
        mpz_mul(G_PRECOMP[i].x,pts_j[i].X,zi2);mpz_mod(G_PRECOMP[i].x,G_PRECOMP[i].x,P_val);
        mpz_mul(G_PRECOMP[i].y,pts_j[i].Y,zi2);mpz_mul(G_PRECOMP[i].y,G_PRECOMP[i].y,zi);
        mpz_mod(G_PRECOMP[i].y,G_PRECOMP[i].y,P_val);G_PRECOMP[i].valid=1;
        mpz_clears(zi,zi2,NULL);
    }
    for(int i=0;i<GBASE_N;i++){mpz_clear(pre[i]);mpz_clear(zinv[i]);}
    free(pre);free(zinv);
    for(int i=0;i<GBASE_N;i++){mpz_clear(pts_j[i].X);mpz_clear(pts_j[i].Y);mpz_clear(pts_j[i].Z);}
    free(pts_j);jac_workspace_clear(&ws);
}

static void _precompute_window_gbase(void) {
    JacWorkspace ws;jac_workspace_init(&ws);
    int n_pts=(1<<WINDOW_W)-1;
    JacPoint *pts_j=(JacPoint*)malloc(n_pts*sizeof(JacPoint));
    for(int i=0;i<n_pts;i++){mpz_init(pts_j[i].X);mpz_init(pts_j[i].Y);mpz_init(pts_j[i].Z);}
    mpz_set(pts_j[0].X,G_x_val);mpz_set(pts_j[0].Y,G_y_val);mpz_set_ui(pts_j[0].Z,1);
    for(int i=1;i<n_pts;i++) _jac_add_z2_one(&pts_j[i],&pts_j[i-1],G_x_val,G_y_val,&ws);
    mpz_t *pre=(mpz_t*)malloc(n_pts*sizeof(mpz_t)),*zinv=(mpz_t*)malloc(n_pts*sizeof(mpz_t));
    for(int i=0;i<n_pts;i++){mpz_init(pre[i]);mpz_init(zinv[i]);}
    mpz_set_ui(pre[0],1);
    for(int i=1;i<n_pts;i++){mpz_mul(pre[i],pre[i-1],pts_j[i-1].Z);mpz_mod(pre[i],pre[i],P_val);}
    mpz_t inv,tmp2;mpz_init(inv);mpz_init(tmp2);
    mpz_mul(tmp2,pre[n_pts-1],pts_j[n_pts-1].Z);mpz_mod(tmp2,tmp2,P_val);
    mpz_powm(inv,tmp2,P_minus_2,P_val);
    for(int i=n_pts-1;i>=0;i--){mpz_mul(zinv[i],inv,pre[i]);mpz_mod(zinv[i],zinv[i],P_val);mpz_mul(inv,inv,pts_j[i].Z);mpz_mod(inv,inv,P_val);}
    mpz_clears(inv,tmp2,NULL);
    mpz_init(G_WINDOW_PRECOMP[0].x);mpz_init(G_WINDOW_PRECOMP[0].y);G_WINDOW_PRECOMP[0].valid=0;
    for(int i=0;i<n_pts;i++){
        mpz_t zi,zi2;mpz_init_set(zi,zinv[i]);mpz_init(zi2);
        mpz_mul(zi2,zi,zi);mpz_mod(zi2,zi2,P_val);
        mpz_init(G_WINDOW_PRECOMP[i+1].x);mpz_init(G_WINDOW_PRECOMP[i+1].y);
        mpz_mul(G_WINDOW_PRECOMP[i+1].x,pts_j[i].X,zi2);mpz_mod(G_WINDOW_PRECOMP[i+1].x,G_WINDOW_PRECOMP[i+1].x,P_val);
        mpz_mul(G_WINDOW_PRECOMP[i+1].y,pts_j[i].Y,zi2);mpz_mul(G_WINDOW_PRECOMP[i+1].y,G_WINDOW_PRECOMP[i+1].y,zi);
        mpz_mod(G_WINDOW_PRECOMP[i+1].y,G_WINDOW_PRECOMP[i+1].y,P_val);G_WINDOW_PRECOMP[i+1].valid=1;
        mpz_clears(zi,zi2,NULL);
    }
    for(int i=0;i<n_pts;i++){mpz_clear(pre[i]);mpz_clear(zinv[i]);}
    free(pre);free(zinv);
    for(int i=0;i<n_pts;i++){mpz_clear(pts_j[i].X);mpz_clear(pts_j[i].Y);mpz_clear(pts_j[i].Z);}
    free(pts_j);jac_workspace_clear(&ws);
}

typedef uint64_t fe_t[4];
typedef struct { fe_t X, Y, Z; int inf; } FePt;
static const uint64_t _FE_P[4]={0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
#define _FE_C 0x1000003D1ULL
static fe_t _FE_Gx, _FE_Gy;

static inline void fe_zero(fe_t r){r[0]=r[1]=r[2]=r[3]=0;}
static inline void fe_copy(fe_t r,const fe_t a){r[0]=a[0];r[1]=a[1];r[2]=a[2];r[3]=a[3];}
static inline int fe_is_zero(const fe_t a){return !((a[0])|(a[1])|(a[2])|(a[3]));}
static inline int fe_is_odd(const fe_t a){return (int)(a[0]&1);}

static void fe_reduce(fe_t r){
    if(r[3]<0xFFFFFFFFFFFFFFFFULL)return; if(r[2]<0xFFFFFFFFFFFFFFFFULL)return;
    if(r[1]<0xFFFFFFFFFFFFFFFFULL)return; if(r[0]<_FE_P[0])return;
    r[0]-=_FE_P[0];r[1]=0;r[2]=0;r[3]=0;
}
static void fe_add(fe_t r,const fe_t a,const fe_t b){
    unsigned __int128 t=(unsigned __int128)a[0]+b[0]; r[0]=(uint64_t)t;t>>=64;
    t+=(unsigned __int128)a[1]+b[1];r[1]=(uint64_t)t;t>>=64;
    t+=(unsigned __int128)a[2]+b[2];r[2]=(uint64_t)t;t>>=64;
    t+=(unsigned __int128)a[3]+b[3];r[3]=(uint64_t)t;t>>=64;
    if(t){unsigned __int128 c=t*_FE_C+r[0];r[0]=(uint64_t)c;c>>=64;
        c+=r[1];r[1]=(uint64_t)c;c>>=64;c+=r[2];r[2]=(uint64_t)c;c>>=64;c+=r[3];r[3]=(uint64_t)c;}
    fe_reduce(r);
}
static void fe_sub(fe_t r,const fe_t a,const fe_t b){
    __int128 borrow=0,t;
    t=(__int128)a[0]-(__int128)b[0]+borrow;r[0]=(uint64_t)t;borrow=t>>64;
    t=(__int128)a[1]-(__int128)b[1]+borrow;r[1]=(uint64_t)t;borrow=t>>64;
    t=(__int128)a[2]-(__int128)b[2]+borrow;r[2]=(uint64_t)t;borrow=t>>64;
    t=(__int128)a[3]-(__int128)b[3]+borrow;r[3]=(uint64_t)t;borrow=t>>64;
    if(borrow<0){unsigned __int128 s=(unsigned __int128)r[0]+_FE_P[0];r[0]=(uint64_t)s;s>>=64;
        s+=(unsigned __int128)r[1]+0xFFFFFFFFFFFFFFFFULL;r[1]=(uint64_t)s;s>>=64;
        s+=(unsigned __int128)r[2]+0xFFFFFFFFFFFFFFFFULL;r[2]=(uint64_t)s;s>>=64;
        s+=(unsigned __int128)r[3]+0xFFFFFFFFFFFFFFFFULL;r[3]=(uint64_t)s;}
}
#define _MULADD(c0,c1,c2,ai,bi) do{ unsigned __int128 _t=(unsigned __int128)(ai)*(bi); \
    uint64_t _lo=(uint64_t)_t,_hi=(uint64_t)(_t>>64); \
    uint64_t _old0=(c0); (c0)+=_lo; unsigned char _cy=((c0)<_old0)?1:0; \
    uint64_t _old1=(c1); (c1)+=_hi; unsigned char _cy2=((c1)<_old1)?1:0; \
    _old1=(c1); (c1)+=(uint64_t)_cy; _cy2|=((c1)<_old1)?1:0; (c2)+=_cy2; }while(0)
#define _NEXTCOL(c0,c1,c2,arr,idx) do{(arr)[idx]=(c0);(c0)=(c1);(c1)=(c2);(c2)=0;}while(0)

static void fe_mul(fe_t r,const fe_t a,const fe_t b){
    uint64_t lo[8]; uint64_t c0=0,c1=0,c2=0;
    _MULADD(c0,c1,c2,a[0],b[0]); _NEXTCOL(c0,c1,c2,lo,0);
    _MULADD(c0,c1,c2,a[0],b[1]);_MULADD(c0,c1,c2,a[1],b[0]); _NEXTCOL(c0,c1,c2,lo,1);
    _MULADD(c0,c1,c2,a[0],b[2]);_MULADD(c0,c1,c2,a[1],b[1]);_MULADD(c0,c1,c2,a[2],b[0]); _NEXTCOL(c0,c1,c2,lo,2);
    _MULADD(c0,c1,c2,a[0],b[3]);_MULADD(c0,c1,c2,a[1],b[2]);_MULADD(c0,c1,c2,a[2],b[1]);_MULADD(c0,c1,c2,a[3],b[0]); _NEXTCOL(c0,c1,c2,lo,3);
    _MULADD(c0,c1,c2,a[1],b[3]);_MULADD(c0,c1,c2,a[2],b[2]);_MULADD(c0,c1,c2,a[3],b[1]); _NEXTCOL(c0,c1,c2,lo,4);
    _MULADD(c0,c1,c2,a[2],b[3]);_MULADD(c0,c1,c2,a[3],b[2]); _NEXTCOL(c0,c1,c2,lo,5);
    _MULADD(c0,c1,c2,a[3],b[3]); _NEXTCOL(c0,c1,c2,lo,6); lo[7]=c0;
    unsigned __int128 acc;
    acc=(unsigned __int128)lo[4]*_FE_C+lo[0];lo[0]=(uint64_t)acc;acc>>=64;
    acc+=(unsigned __int128)lo[5]*_FE_C+lo[1];lo[1]=(uint64_t)acc;acc>>=64;
    acc+=(unsigned __int128)lo[6]*_FE_C+lo[2];lo[2]=(uint64_t)acc;acc>>=64;
    acc+=(unsigned __int128)lo[7]*_FE_C+lo[3];lo[3]=(uint64_t)acc;acc>>=64;
    if(acc){unsigned __int128 carry=acc*_FE_C+(unsigned __int128)lo[0];lo[0]=(uint64_t)carry;carry>>=64;
        carry+=lo[1];lo[1]=(uint64_t)carry;carry>>=64;carry+=lo[2];lo[2]=(uint64_t)carry;carry>>=64;carry+=lo[3];lo[3]=(uint64_t)carry;}
    r[0]=lo[0];r[1]=lo[1];r[2]=lo[2];r[3]=lo[3]; fe_reduce(r);
}
static void fe_sqr(fe_t r,const fe_t a){fe_mul(r,a,a);} /* simplificado p/ host */
static void fe_inv(fe_t r,const fe_t a){
    fe_t x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t45,tx,t1;int j;
    fe_sqr(x2,a);fe_mul(x2,x2,a);fe_sqr(x3,x2);fe_mul(x3,x3,a);
    fe_copy(x6,x3);for(j=0;j<3;j++)fe_sqr(x6,x6);fe_mul(x6,x6,x3);
    fe_copy(x9,x6);for(j=0;j<3;j++)fe_sqr(x9,x9);fe_mul(x9,x9,x3);
    fe_copy(x11,x9);for(j=0;j<2;j++)fe_sqr(x11,x11);fe_mul(x11,x11,x2);
    fe_copy(x22,x11);for(j=0;j<11;j++)fe_sqr(x22,x22);fe_mul(x22,x22,x11);
    fe_copy(x44,x22);for(j=0;j<22;j++)fe_sqr(x44,x44);fe_mul(x44,x44,x22);
    fe_copy(x88,x44);for(j=0;j<44;j++)fe_sqr(x88,x88);fe_mul(x88,x88,x44);
    fe_copy(x176,x88);for(j=0;j<88;j++)fe_sqr(x176,x176);fe_mul(x176,x176,x88);
    fe_copy(x220,x176);for(j=0;j<44;j++)fe_sqr(x220,x220);fe_mul(x220,x220,x44);
    fe_copy(x223,x220);for(j=0;j<3;j++)fe_sqr(x223,x223);fe_mul(x223,x223,x3);
    fe_sqr(t45,a);fe_sqr(t45,t45);fe_mul(t45,t45,a);
    fe_sqr(t45,t45);fe_mul(t45,t45,a);fe_sqr(t45,t45);fe_sqr(t45,t45);fe_mul(t45,t45,a);
    fe_copy(tx,x22);for(j=0;j<10;j++)fe_sqr(tx,tx);fe_mul(tx,tx,t45);
    fe_copy(t1,x223);for(j=0;j<33;j++)fe_sqr(t1,t1);fe_mul(r,t1,tx);
    (void)x6;(void)x9;
}

static void fe_from_mpz(fe_t r,const mpz_t n){
    uint8_t buf[32]={0};mpz_export_32be(n,buf);
    for(int i=0;i<4;i++){int off=(3-i)*8;
        r[i]=((uint64_t)buf[off+7])|((uint64_t)buf[off+6]<<8)|((uint64_t)buf[off+5]<<16)|((uint64_t)buf[off+4]<<24)
            |((uint64_t)buf[off+3]<<32)|((uint64_t)buf[off+2]<<40)|((uint64_t)buf[off+1]<<48)|((uint64_t)buf[off+0]<<56);}
}
static void fe_to_bytes32(const fe_t a,uint8_t *buf){
    for(int i=0;i<4;i++){uint64_t w=a[3-i];
        buf[i*8]=(uint8_t)(w>>56);buf[i*8+1]=(uint8_t)(w>>48);buf[i*8+2]=(uint8_t)(w>>40);buf[i*8+3]=(uint8_t)(w>>32);
        buf[i*8+4]=(uint8_t)(w>>24);buf[i*8+5]=(uint8_t)(w>>16);buf[i*8+6]=(uint8_t)(w>>8);buf[i*8+7]=(uint8_t)(w);}
}
static inline void fept_set_inf(FePt *r){fe_zero(r->X);fe_zero(r->Y);r->Y[0]=1;fe_zero(r->Z);r->inf=1;}
static inline int fept_is_inf(const FePt *p){return p->inf||fe_is_zero(p->Z);}

static void fept_dbl(FePt *r,const FePt *p){
    if(fept_is_inf(p)){fept_set_inf(r);return;}
    fe_t ysq,xysq,S,M,ysq2,X2,Y2,Z2,tmp;
    fe_sqr(ysq,p->Y);fe_mul(xysq,p->X,ysq);fe_add(S,xysq,xysq);fe_add(S,S,S);
    fe_sqr(tmp,p->X);fe_add(M,tmp,tmp);fe_add(M,M,tmp);fe_sqr(X2,M);
    fe_sub(X2,X2,S);fe_sub(X2,X2,S);fe_sqr(ysq2,ysq);fe_sub(tmp,S,X2);
    fe_mul(Y2,M,tmp);fe_add(tmp,ysq2,ysq2);fe_add(tmp,tmp,tmp);fe_add(tmp,tmp,tmp);
    fe_sub(Y2,Y2,tmp);fe_mul(Z2,p->Y,p->Z);fe_add(Z2,Z2,Z2);
    fe_copy(r->X,X2);fe_copy(r->Y,Y2);fe_copy(r->Z,Z2);r->inf=0;
}
static void fept_add_aff(FePt *r,const FePt *p1,const fe_t x2,const fe_t y2){
    if(fept_is_inf(p1)){fe_copy(r->X,x2);fe_copy(r->Y,y2);r->Z[0]=1;r->Z[1]=r->Z[2]=r->Z[3]=0;r->inf=0;return;}
    fe_t Z1sq,U2,S2,H,R,H2,H3,U1H2,X3,Y3,Z3,tmp;
    fe_sqr(Z1sq,p1->Z);fe_mul(U2,x2,Z1sq);fe_mul(S2,y2,Z1sq);fe_mul(S2,S2,p1->Z);
    fe_sub(H,U2,p1->X);fe_sub(R,S2,p1->Y);
    if(fe_is_zero(H)){if(fe_is_zero(R)){fept_dbl(r,p1);return;}fept_set_inf(r);return;}
    fe_sqr(H2,H);fe_mul(H3,H,H2);fe_mul(U1H2,p1->X,H2);fe_sqr(X3,R);
    fe_sub(X3,X3,H3);fe_sub(X3,X3,U1H2);fe_sub(X3,X3,U1H2);fe_sub(tmp,U1H2,X3);
    fe_mul(Y3,R,tmp);fe_mul(tmp,p1->Y,H3);fe_sub(Y3,Y3,tmp);fe_mul(Z3,H,p1->Z);
    fe_copy(r->X,X3);fe_copy(r->Y,Y3);fe_copy(r->Z,Z3);r->inf=0;
}

typedef struct{fe_t x,y;int valid;}FePtAff;
static FePtAff _FE_WIN[WINDOW_N];

static void _fe_init_tables(void){
    _FE_WIN[0].valid=0;
    for(int i=1;i<WINDOW_N;i++){
        if(G_WINDOW_PRECOMP[i].valid){fe_from_mpz(_FE_WIN[i].x,G_WINDOW_PRECOMP[i].x);
            fe_from_mpz(_FE_WIN[i].y,G_WINDOW_PRECOMP[i].y);_FE_WIN[i].valid=1;}
        else _FE_WIN[i].valid=0;
    }
    fe_from_mpz(_FE_Gx,G_x_val);fe_from_mpz(_FE_Gy,G_y_val);
}

static void fe_scalar_mul(FePt *Q,const uint8_t *scalar32be){
    fept_set_inf(Q); int w=WINDOW_W,num_windows=(256+w-1)/w;
    for(int win_idx=num_windows-1;win_idx>=0;win_idx--){
        for(int k=0;k<w;k++)fept_dbl(Q,Q);
        int bit_pos=win_idx*w; int byte_hi=31-(bit_pos>>3); int bit_off=bit_pos&7;
        uint16_t bits_raw=((uint16_t)(byte_hi>=0?scalar32be[byte_hi]:0))
            |((uint16_t)(byte_hi>0?scalar32be[byte_hi-1]:0))<<8;
        int window=(int)((bits_raw>>bit_off)&((1<<w)-1));
        if(window&&_FE_WIN[window].valid) fept_add_aff(Q,Q,_FE_WIN[window].x,_FE_WIN[window].y);
    }
}

static void init_curve(void){
    memset(_B58_MAP,-1,sizeof(_B58_MAP));
    for(int i=0;B58_ALPHA[i];i++) _B58_MAP[(unsigned char)B58_ALPHA[i]]=i;
    mpz_init(P_val);mpz_set_str(P_val,P_HEX,16);
    mpz_init(G_x_val);mpz_set_str(G_x_val,G_x_HEX,16);
    mpz_init(G_y_val);mpz_set_str(G_y_val,G_y_HEX,16);
    mpz_init(P_minus_2);mpz_sub_ui(P_minus_2,P_val,2);
    printf("[init] Pré-computando tabela de base (256 pts)...\n"); _precompute_gbase();
    printf("[init] Pré-computando tabela janela 6-bits (64 pts)...\n"); _precompute_window_gbase();
    printf("[init] Curva pronta.\n\n");
}

static void base58_encode(const uint8_t *payload,size_t payload_len,char *out){
    uint8_t h1[32],h2[32];sha256(payload,payload_len,h1);sha256(h1,32,h2);
    size_t total=payload_len+4;uint8_t *buf=(uint8_t*)malloc(total);
    memcpy(buf,payload,payload_len);memcpy(buf+payload_len,h2,4);
    mpz_t numero;mpz_init(numero);mpz_import(numero,total,1,1,0,0,buf);free(buf);
    char tmp[200];int tmp_len=0;mpz_t rem;mpz_init(rem);
    while(mpz_sgn(numero)>0){mpz_tdiv_qr_ui(numero,rem,numero,58);tmp[tmp_len++]=B58_ALPHA[mpz_get_ui(rem)];}
    mpz_clears(numero,rem,NULL);
    int zeros=0;while((size_t)zeros<payload_len&&payload[zeros]==0)zeros++;
    int j=0;for(int i=0;i<zeros;i++)out[j++]='1';
    for(int i=tmp_len-1;i>=0;i--)out[j++]=tmp[i];out[j]='\0';
}
static void _hash160_do_endereco(const char *addr,uint8_t *out20){
    mpz_t num;mpz_init(num);mpz_set_ui(num,0);
    for(const char *p=addr;*p;p++){int idx=_B58_MAP[(unsigned char)*p];if(idx<0)continue;mpz_mul_ui(num,num,58);mpz_add_ui(num,num,(unsigned int)idx);}
    uint8_t raw[32]={0};size_t count=0;mpz_export(raw,&count,1,1,0,0,num);mpz_clear(num);
    uint8_t buf25[25]={0};if(count<=25)memcpy(buf25+(25-count),raw,count);memcpy(out20,buf25+1,20);
}
static void gerar_endereco_bitcoin(const mpz_t chave_privada_int,char *addr_c,char *addr_nc){

    uint8_t sc[32]; mpz_export_32be(chave_privada_int,sc);
    FePt Q; fe_scalar_mul(&Q,sc);
    if(fept_is_inf(&Q))return;
    fe_t Zi,Zi2,xr,yr;
    fe_inv(Zi,Q.Z);fe_sqr(Zi2,Zi);fe_mul(xr,Q.X,Zi2);fe_mul(yr,Q.Y,Zi2);fe_mul(yr,yr,Zi);
    uint8_t xb[32]; fe_to_bytes32(xr,xb);
    uint8_t pfx=fe_is_odd(yr)?0x03:0x02;
    uint8_t comp[33];comp[0]=pfx;memcpy(comp+1,xb,32);
    uint8_t sha_r[32],rmd_r[20];sha256_33b(comp,sha_r);ripemd160(sha_r,32,rmd_r);
    uint8_t versioned[21];versioned[0]=0x00;memcpy(versioned+1,rmd_r,20);
    base58_encode(versioned,21,addr_c);
    if(!SOMENTE_COMPRIMIDO&&addr_nc){
        uint8_t yb[32];fe_to_bytes32(yr,yb);
        uint8_t uncomp[65];uncomp[0]=0x04;memcpy(uncomp+1,xb,32);memcpy(uncomp+33,yb,32);
        sha256(uncomp,65,sha_r);ripemd160(sha_r,32,rmd_r);
        versioned[0]=0x00;memcpy(versioned+1,rmd_r,20);
        base58_encode(versioned,21,addr_nc);
    }
}


static int detectar_tipo_endereco(const char *addr) {
    if (!addr || !addr[0]) return ADDR_TYPE_P2PKH;
    if (addr[0] == '1') return ADDR_TYPE_P2PKH;
    if (addr[0] == '3') return ADDR_TYPE_P2SH;
    if (strncmp(addr, "bc1q", 4) == 0 || strncmp(addr, "BC1Q", 4) == 0)
        return ADDR_TYPE_P2WPKH;
    if (strncmp(addr, "bc1p", 4) == 0 || strncmp(addr, "BC1P", 4) == 0)
        return ADDR_TYPE_P2TR;
    return ADDR_TYPE_P2PKH;
}

static int extrair_hash_do_endereco(const char *addr, uint8_t *out, int addr_type) {
    if (addr_type == ADDR_TYPE_P2PKH || addr_type == ADDR_TYPE_P2SH) {
        _hash160_do_endereco(addr, out);
        return 20;
    }
    if (addr_type == ADDR_TYPE_P2WPKH || addr_type == ADDR_TYPE_P2TR) {
        int prog_len = 0;
        int wv = bech32_decode_witness(addr, out, &prog_len);
        if (wv < 0) {
            fprintf(stderr, "ERRO: falha ao decodificar endereço bech32: %s\n", addr);
            return -1;
        }
        return prog_len; /* 20 para P2WPKH, 32 para P2TR */
    }
    return -1;
}

static int modo_gpu_do_tipo(int addr_type) {
    switch (addr_type) {
        case ADDR_TYPE_P2PKH:   return GPU_MODE_HASH160;
        case ADDR_TYPE_P2PKH_U: return GPU_MODE_UNCOMP;
        case ADDR_TYPE_P2SH:    return GPU_MODE_P2SH;
        case ADDR_TYPE_P2WPKH:  return GPU_MODE_HASH160; /* mesma hash160! */
        case ADDR_TYPE_P2TR:    return GPU_MODE_P2TR;
        default: return GPU_MODE_HASH160;
    }
}

typedef uint64_t gfe_t[4];

#define GFE_C   0x1000003D1ULL
#define GFE_P0  0xFFFFFFFEFFFFFC2FULL
#define GFE_P1  0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__
void gfe_zero(gfe_t r) { ((ulonglong2*)r)[0] = make_ulonglong2(0,0); ((ulonglong2*)r)[1] = make_ulonglong2(0,0); }

__device__ __forceinline__
void gfe_copy(gfe_t r, const gfe_t a) { ((ulonglong2*)r)[0] = ((const ulonglong2*)a)[0]; ((ulonglong2*)r)[1] = ((const ulonglong2*)a)[1]; }

__device__ __forceinline__
int gfe_is_zero(const gfe_t a) { return !((a[0])|(a[1])|(a[2])|(a[3])); }

__device__ __forceinline__
int gfe_is_odd(const gfe_t a) { return (int)(a[0]&1); }

__device__ __forceinline__
void gfe_reduce(gfe_t r) {
    if (r[3] < GFE_P1) return;
    if (r[2] < GFE_P1) return;
    if (r[1] < GFE_P1) return;
    if (r[0] < GFE_P0) return;
    r[0] -= GFE_P0; r[1]=0; r[2]=0; r[3]=0;
}

__device__ __forceinline__
void gfe_add(gfe_t r, const gfe_t a, const gfe_t b) {
    uint64_t r0,r1,r2,r3,cy;
    asm("add.cc.u64 %0,%5,%9;\n\t""addc.cc.u64 %1,%6,%10;\n\t"
        "addc.cc.u64 %2,%7,%11;\n\t""addc.cc.u64 %3,%8,%12;\n\t""addc.u64 %4,0,0;\n\t"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(cy)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    if(cy){asm("add.cc.u64 %0,%0,%4;\n\t""addc.cc.u64 %1,%1,0;\n\t"
        "addc.cc.u64 %2,%2,0;\n\t""addc.u64 %3,%3,0;\n\t"
        :"+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3):"l"((uint64_t)GFE_C));}
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3; gfe_reduce(r);
}

__device__ __forceinline__
void gfe_sub(gfe_t r, const gfe_t a, const gfe_t b) {
    uint64_t r0,r1,r2,r3,bw;
    asm("sub.cc.u64 %0,%5,%9;\n\t""subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t""subc.cc.u64 %3,%8,%12;\n\t""subc.u64 %4,0,0;\n\t"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(bw)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    if(bw){asm("add.cc.u64 %0,%0,%4;\n\t""addc.cc.u64 %1,%1,%5;\n\t"
        "addc.cc.u64 %2,%2,%5;\n\t""addc.u64 %3,%3,%5;\n\t"
        :"+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3):"l"((uint64_t)GFE_P0),"l"((uint64_t)GFE_P1));}
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

/* ── gfe_mul: Comba 4×4 usando __umul64hi ──────────────────────────── */
__device__ __forceinline__
void _gfe_muladd(uint64_t &c0, uint64_t &c1, uint64_t &c2, uint64_t ai, uint64_t bi) {
    uint64_t lo = ai * bi;
    uint64_t hi = __umul64hi(ai, bi);
    asm("add.cc.u64  %0, %0, %3;\n\t"
        "addc.cc.u64 %1, %1, %4;\n\t"
        "addc.u64    %2, %2, 0;\n\t"
        : "+l"(c0), "+l"(c1), "+l"(c2) : "l"(lo), "l"(hi));
}

__device__ __forceinline__
void gfe_mul(gfe_t r, const gfe_t a, const gfe_t b) {
    uint64_t lo[8];
    uint64_t c0=0,c1=0,c2=0;

    _gfe_muladd(c0,c1,c2,a[0],b[0]); lo[0]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[0],b[1]);_gfe_muladd(c0,c1,c2,a[1],b[0]); lo[1]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[0],b[2]);_gfe_muladd(c0,c1,c2,a[1],b[1]);_gfe_muladd(c0,c1,c2,a[2],b[0]); lo[2]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[0],b[3]);_gfe_muladd(c0,c1,c2,a[1],b[2]);_gfe_muladd(c0,c1,c2,a[2],b[1]);_gfe_muladd(c0,c1,c2,a[3],b[0]); lo[3]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[1],b[3]);_gfe_muladd(c0,c1,c2,a[2],b[2]);_gfe_muladd(c0,c1,c2,a[3],b[1]); lo[4]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[2],b[3]);_gfe_muladd(c0,c1,c2,a[3],b[2]); lo[5]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[3],b[3]); lo[6]=c0; lo[7]=c1;

    uint64_t acc_lo, acc_hi;

    acc_lo = lo[4] * GFE_C; acc_hi = __umul64hi(lo[4], GFE_C);
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(acc_hi) : "l"(lo[0]));
    lo[0] = acc_lo;

    acc_lo = lo[5] * GFE_C; uint64_t hi5 = __umul64hi(lo[5], GFE_C);
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi5) : "l"(lo[1]));
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi5) : "l"(acc_hi));
    lo[1] = acc_lo;

    acc_lo = lo[6] * GFE_C; uint64_t hi6 = __umul64hi(lo[6], GFE_C);
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi6) : "l"(lo[2]));
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi6) : "l"(hi5));
    lo[2] = acc_lo;

    acc_lo = lo[7] * GFE_C; uint64_t hi7 = __umul64hi(lo[7], GFE_C);
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi7) : "l"(lo[3]));
    asm("add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %1, 0;\n\t"
        : "+l"(acc_lo), "+l"(hi7) : "l"(hi6));
    lo[3] = acc_lo;

    if (hi7) {
        uint64_t red = hi7 * GFE_C;
        asm("add.cc.u64  %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64    %3, %3, 0;\n\t"
            : "+l"(lo[0]), "+l"(lo[1]), "+l"(lo[2]), "+l"(lo[3]) : "l"(red));
    }

    r[0]=lo[0]; r[1]=lo[1]; r[2]=lo[2]; r[3]=lo[3];
    gfe_reduce(r);
}

__device__ __forceinline__
void _gfe_dbl_muladd(uint64_t &c0, uint64_t &c1, uint64_t &c2, uint64_t ai, uint64_t bi) {
    uint64_t lo = ai * bi;
    uint64_t hi = __umul64hi(ai, bi);
    uint64_t lo2 = lo << 1;
    uint64_t hi2 = (hi << 1) | (lo >> 63);
    uint64_t cy  = hi >> 63;
    asm("add.cc.u64  %0, %0, %3;\n\t"
        "addc.cc.u64 %1, %1, %4;\n\t"
        "addc.u64    %2, %2, %5;\n\t"
        : "+l"(c0), "+l"(c1), "+l"(c2) : "l"(lo2), "l"(hi2), "l"(cy));
}

__device__ __forceinline__
void gfe_sqr(gfe_t r, const gfe_t a) {
    uint64_t lo[8], c0=0, c1=0, c2=0;
    _gfe_muladd(c0,c1,c2,a[0],a[0]); lo[0]=c0;c0=c1;c1=c2;c2=0;
    _gfe_dbl_muladd(c0,c1,c2,a[0],a[1]); lo[1]=c0;c0=c1;c1=c2;c2=0;
    _gfe_dbl_muladd(c0,c1,c2,a[0],a[2]); _gfe_muladd(c0,c1,c2,a[1],a[1]); lo[2]=c0;c0=c1;c1=c2;c2=0;
    _gfe_dbl_muladd(c0,c1,c2,a[0],a[3]); _gfe_dbl_muladd(c0,c1,c2,a[1],a[2]); lo[3]=c0;c0=c1;c1=c2;c2=0;
    _gfe_dbl_muladd(c0,c1,c2,a[1],a[3]); _gfe_muladd(c0,c1,c2,a[2],a[2]); lo[4]=c0;c0=c1;c1=c2;c2=0;
    _gfe_dbl_muladd(c0,c1,c2,a[2],a[3]); lo[5]=c0;c0=c1;c1=c2;c2=0;
    _gfe_muladd(c0,c1,c2,a[3],a[3]); lo[6]=c0; lo[7]=c1;
    uint64_t al,ah;
    al=lo[4]*GFE_C;ah=__umul64hi(lo[4],GFE_C);
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(ah):"l"(lo[0]));lo[0]=al;
    al=lo[5]*GFE_C;uint64_t h5=__umul64hi(lo[5],GFE_C);
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h5):"l"(lo[1]));
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h5):"l"(ah));lo[1]=al;
    al=lo[6]*GFE_C;uint64_t h6=__umul64hi(lo[6],GFE_C);
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h6):"l"(lo[2]));
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h6):"l"(h5));lo[2]=al;
    al=lo[7]*GFE_C;uint64_t h7=__umul64hi(lo[7],GFE_C);
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h7):"l"(lo[3]));
    asm("add.cc.u64 %0,%0,%2;\n\t""addc.u64 %1,%1,0;\n\t":"+l"(al),"+l"(h7):"l"(h6));lo[3]=al;
    if(h7){uint64_t rd=h7*GFE_C;asm("add.cc.u64 %0,%0,%4;\n\t""addc.cc.u64 %1,%1,0;\n\t""addc.cc.u64 %2,%2,0;\n\t""addc.u64 %3,%3,0;\n\t":"+l"(lo[0]),"+l"(lo[1]),"+l"(lo[2]),"+l"(lo[3]):"l"(rd));}
    r[0]=lo[0];r[1]=lo[1];r[2]=lo[2];r[3]=lo[3];gfe_reduce(r);
}

__device__ __noinline__
void gfe_inv(gfe_t r, const gfe_t a) {
    /* Fermat: a^(p-2) via addition chain para secp256k1 */
    gfe_t x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t45,tx,t1;
    int j;
    gfe_sqr(x2,a);gfe_mul(x2,x2,a);
    gfe_sqr(x3,x2);gfe_mul(x3,x3,a);
    gfe_copy(x6,x3);for(j=0;j<3;j++)gfe_sqr(x6,x6);gfe_mul(x6,x6,x3);
    gfe_copy(x9,x6);for(j=0;j<3;j++)gfe_sqr(x9,x9);gfe_mul(x9,x9,x3);
    gfe_copy(x11,x9);for(j=0;j<2;j++)gfe_sqr(x11,x11);gfe_mul(x11,x11,x2);
    gfe_copy(x22,x11);
    for(j=0;j<11;j++)gfe_sqr(x22,x22);gfe_mul(x22,x22,x11);
    gfe_copy(x44,x22);
    for(j=0;j<22;j++)gfe_sqr(x44,x44);gfe_mul(x44,x44,x22);
    gfe_copy(x88,x44);
    for(j=0;j<44;j++)gfe_sqr(x88,x88);gfe_mul(x88,x88,x44);
    gfe_copy(x176,x88);
    for(j=0;j<88;j++)gfe_sqr(x176,x176);gfe_mul(x176,x176,x88);
    gfe_copy(x220,x176);
    for(j=0;j<44;j++)gfe_sqr(x220,x220);gfe_mul(x220,x220,x44);
    gfe_copy(x223,x220);for(j=0;j<3;j++)gfe_sqr(x223,x223);gfe_mul(x223,x223,x3);
    gfe_sqr(t45,a);gfe_sqr(t45,t45);gfe_mul(t45,t45,a);
    gfe_sqr(t45,t45);gfe_mul(t45,t45,a);gfe_sqr(t45,t45);gfe_sqr(t45,t45);gfe_mul(t45,t45,a);
    gfe_copy(tx,x22);
    for(j=0;j<10;j++)gfe_sqr(tx,tx);gfe_mul(tx,tx,t45);
    gfe_copy(t1,x223);
    for(j=0;j<33;j++)gfe_sqr(t1,t1);gfe_mul(r,t1,tx);
}

__device__ __forceinline__
void gfe_to_bytes32(const gfe_t a, uint8_t *buf) {
    for (int i = 0; i < 4; i++) {
        uint64_t w = a[3-i];
        buf[i*8+0]=(uint8_t)(w>>56); buf[i*8+1]=(uint8_t)(w>>48);
        buf[i*8+2]=(uint8_t)(w>>40); buf[i*8+3]=(uint8_t)(w>>32);
        buf[i*8+4]=(uint8_t)(w>>24); buf[i*8+5]=(uint8_t)(w>>16);
        buf[i*8+6]=(uint8_t)(w>>8);  buf[i*8+7]=(uint8_t)(w);
    }
}

/* ── Ponto Jacobiano GPU ───────────────────────────────────────────── */
typedef struct { gfe_t X, Y, Z; } GFePt;

__device__ __forceinline__
void gfept_set_inf(GFePt *r) { gfe_zero(r->X); gfe_zero(r->Y); r->Y[0]=1; gfe_zero(r->Z); }

__device__ __forceinline__
int gfept_is_inf(const GFePt *p) { return gfe_is_zero(p->Z); }

__device__ __forceinline__
void gfept_dbl(GFePt *r, const GFePt *p) {
    if (__builtin_expect(gfept_is_inf(p),0)) { gfept_set_inf(r); return; }
    gfe_t ysq,xysq,S,M,ysq2,X2,Y2,Z2,tmp;
    gfe_sqr(ysq,p->Y); gfe_mul(xysq,p->X,ysq);
    gfe_add(S,xysq,xysq); gfe_add(S,S,S);
    gfe_sqr(tmp,p->X); gfe_add(M,tmp,tmp); gfe_add(M,M,tmp);
    gfe_sqr(X2,M); gfe_sub(X2,X2,S); gfe_sub(X2,X2,S);
    gfe_sqr(ysq2,ysq); gfe_sub(tmp,S,X2); gfe_mul(Y2,M,tmp);
    gfe_add(tmp,ysq2,ysq2); gfe_add(tmp,tmp,tmp); gfe_add(tmp,tmp,tmp);
    gfe_sub(Y2,Y2,tmp);
    gfe_mul(Z2,p->Y,p->Z); gfe_add(Z2,Z2,Z2);
    gfe_copy(r->X,X2); gfe_copy(r->Y,Y2); gfe_copy(r->Z,Z2);
}

__device__ __forceinline__
void gfept_add_aff(GFePt *r, const GFePt *p1, const gfe_t x2, const gfe_t y2) {
    if (__builtin_expect(gfept_is_inf(p1),0)) {
        gfe_copy(r->X,x2); gfe_copy(r->Y,y2);
        r->Z[0]=1; r->Z[1]=r->Z[2]=r->Z[3]=0; return;
    }
    gfe_t Z1sq,U2,S2,H,R,H2,H3,U1H2,X3,Y3,Z3,tmp;
    gfe_sqr(Z1sq,p1->Z); gfe_mul(U2,x2,Z1sq);
    gfe_mul(S2,y2,Z1sq); gfe_mul(S2,S2,p1->Z);
    gfe_sub(H,U2,p1->X); gfe_sub(R,S2,p1->Y);
    if (__builtin_expect(gfe_is_zero(H),0)) {
        if (gfe_is_zero(R)) { gfept_dbl(r,p1); return; }
        gfept_set_inf(r); return;
    }
    gfe_sqr(H2,H); gfe_mul(H3,H,H2); gfe_mul(U1H2,p1->X,H2);
    gfe_sqr(X3,R); gfe_sub(X3,X3,H3); gfe_sub(X3,X3,U1H2); gfe_sub(X3,X3,U1H2);
    gfe_sub(tmp,U1H2,X3); gfe_mul(Y3,R,tmp); gfe_mul(tmp,p1->Y,H3);
    gfe_sub(Y3,Y3,tmp); gfe_mul(Z3,H,p1->Z);
    gfe_copy(r->X,X3); gfe_copy(r->Y,Y3); gfe_copy(r->Z,Z3);
}

__device__ __forceinline__
void gfe_aff_dbl(gfe_t rx, gfe_t ry, const gfe_t px, const gfe_t py) {
    gfe_t x2, twoY, twoY_inv, lam, lam2, tmp;
    gfe_sqr(x2, px);                  /* x^2 */
    gfe_add(tmp, x2, x2);
    gfe_add(tmp, tmp, x2);            /* 3*x^2 */
    gfe_add(twoY, py, py);            /* 2*y */
    gfe_inv(twoY_inv, twoY);          /* inv(2*y) */
    gfe_mul(lam, tmp, twoY_inv);      /* lambda */
    gfe_sqr(lam2, lam);               /* lambda^2 */
    gfe_sub(rx, lam2, px);
    gfe_sub(rx, rx, px);              /* x2 = lambda^2 - 2*x */
    gfe_sub(tmp, px, rx);             /* x - x2 */
    gfe_mul(ry, lam, tmp);
    gfe_sub(ry, ry, py);              /* y2 = lambda*(x - x2) - y */
}

/* ── GPU SHA-256 (33 bytes) ────────────────────────────────────────── */
#define _GS_ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define _GS_CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define _GS_MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define _GS_EP0(a) (_GS_ROTR(a,2)^_GS_ROTR(a,13)^_GS_ROTR(a,22))
#define _GS_EP1(e) (_GS_ROTR(e,6)^_GS_ROTR(e,11)^_GS_ROTR(e,25))
#define _GS_SG0(x) (_GS_ROTR(x,7)^_GS_ROTR(x,18)^((x)>>3))
#define _GS_SG1(x) (_GS_ROTR(x,17)^_GS_ROTR(x,19)^((x)>>10))

#define _GS_ROUND(a,b,c,d,e,f,g,h,k,w) do{ \
    uint32_t _t1=(h)+_GS_EP1(e)+_GS_CH(e,f,g)+(k)+(w); \
    uint32_t _t2=_GS_EP0(a)+_GS_MAJ(a,b,c); (d)+=_t1;(h)=_t1+_t2;}while(0)

__device__ __forceinline__
void gpu_sha256_33b(const uint8_t *pk33, uint8_t *out32) {
    uint32_t W[64];
    W[0]=((uint32_t)pk33[0]<<24)|((uint32_t)pk33[1]<<16)|((uint32_t)pk33[2]<<8)|pk33[3];
    W[1]=((uint32_t)pk33[4]<<24)|((uint32_t)pk33[5]<<16)|((uint32_t)pk33[6]<<8)|pk33[7];
    W[2]=((uint32_t)pk33[8]<<24)|((uint32_t)pk33[9]<<16)|((uint32_t)pk33[10]<<8)|pk33[11];
    W[3]=((uint32_t)pk33[12]<<24)|((uint32_t)pk33[13]<<16)|((uint32_t)pk33[14]<<8)|pk33[15];
    W[4]=((uint32_t)pk33[16]<<24)|((uint32_t)pk33[17]<<16)|((uint32_t)pk33[18]<<8)|pk33[19];
    W[5]=((uint32_t)pk33[20]<<24)|((uint32_t)pk33[21]<<16)|((uint32_t)pk33[22]<<8)|pk33[23];
    W[6]=((uint32_t)pk33[24]<<24)|((uint32_t)pk33[25]<<16)|((uint32_t)pk33[26]<<8)|pk33[27];
    W[7]=((uint32_t)pk33[28]<<24)|((uint32_t)pk33[29]<<16)|((uint32_t)pk33[30]<<8)|pk33[31];
    W[8]=((uint32_t)pk33[32]<<24)|0x00800000u;
    for(int i=9;i<15;i++) W[i]=0; W[15]=264u;
    #pragma unroll
    for(int i=16;i<64;i++) W[i]=_GS_SG1(W[i-2])+W[i-7]+_GS_SG0(W[i-15])+W[i-16];

    uint32_t a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d_=0xa54ff53au,
             e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;

    #pragma unroll
    for(int i=0;i<64;i+=8){
        _GS_ROUND(a,b,c,d_,e,f,g,h,d_SHA256_K[i+0],W[i+0]);
        _GS_ROUND(h,a,b,c,d_,e,f,g,d_SHA256_K[i+1],W[i+1]);
        _GS_ROUND(g,h,a,b,c,d_,e,f,d_SHA256_K[i+2],W[i+2]);
        _GS_ROUND(f,g,h,a,b,c,d_,e,d_SHA256_K[i+3],W[i+3]);
        _GS_ROUND(e,f,g,h,a,b,c,d_,d_SHA256_K[i+4],W[i+4]);
        _GS_ROUND(d_,e,f,g,h,a,b,c,d_SHA256_K[i+5],W[i+5]);
        _GS_ROUND(c,d_,e,f,g,h,a,b,d_SHA256_K[i+6],W[i+6]);
        _GS_ROUND(b,c,d_,e,f,g,h,a,d_SHA256_K[i+7],W[i+7]);
    }
    uint32_t H[8]={0x6a09e667u+a,0xbb67ae85u+b,0x3c6ef372u+c,0xa54ff53au+d_,
                   0x510e527fu+e,0x9b05688cu+f,0x1f83d9abu+g,0x5be0cd19u+h};
    for(int i=0;i<8;i++){
        out32[i*4]=(uint8_t)(H[i]>>24);out32[i*4+1]=(uint8_t)(H[i]>>16);
        out32[i*4+2]=(uint8_t)(H[i]>>8);out32[i*4+3]=(uint8_t)(H[i]);
    }
}

/* ── GPU RIPEMD-160 (32 bytes) ─────────────────────────────────────── */
#define _GR_ROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define _GR_F1(B,C,D) ((B)^(C)^(D))
#define _GR_F2(B,C,D) (((B)&(C))|(~(B)&(D)))
#define _GR_F3(B,C,D) (((B)|(~(C)))^(D))
#define _GR_F4(B,C,D) (((B)&(D))|((C)&~(D)))
#define _GR_F5(B,C,D) ((B)^((C)|(~(D))))

#define _GR_RL1(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F1(B,C,D)+X[xi],(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RL2(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F2(B,C,D)+X[xi]+0x5A827999u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RL3(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F3(B,C,D)+X[xi]+0x6ED9EBA1u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RL4(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F4(B,C,D)+X[xi]+0x8F1BBCDCu,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RL5(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F5(B,C,D)+X[xi]+0xA953FD4Eu,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RR1(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F5(B,C,D)+X[xi]+0x50A28BE6u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RR2(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F4(B,C,D)+X[xi]+0x5C4DD124u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RR3(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F3(B,C,D)+X[xi]+0x6D703EF3u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RR4(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F2(B,C,D)+X[xi]+0x7A6D76E9u,(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)
#define _GR_RR5(A,B,C,D,E,xi,si) do{uint32_t _T=_GR_ROTL((A)+_GR_F1(B,C,D)+X[xi],(si))+(E);(A)=(E);(E)=(D);(D)=_GR_ROTL((C),10);(C)=(B);(B)=_T;}while(0)

__device__
void gpu_ripemd160_32b(const uint8_t *msg32, uint8_t *out20) {
    uint32_t X[16];
    for(int i=0;i<8;i++)
        X[i]=(uint32_t)msg32[i*4]|((uint32_t)msg32[i*4+1]<<8)|((uint32_t)msg32[i*4+2]<<16)|((uint32_t)msg32[i*4+3]<<24);
    X[8]=0x00000080u; X[9]=0;X[10]=0;X[11]=0;X[12]=0;X[13]=0;X[14]=256u;X[15]=0;

    uint32_t h0=0x67452301u,h1=0xEFCDAB89u,h2=0x98BADCFEu,h3=0x10325476u,h4=0xC3D2E1F0u;
    uint32_t A=h0,B=h1,C=h2,D=h3,E=h4;
    uint32_t Ap=h0,Bp=h1,Cp=h2,Dp=h3,Ep=h4;

    _GR_RL1(A,B,C,D,E, 0,11);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 5, 8);_GR_RL1(A,B,C,D,E, 1,14);_GR_RR1(Ap,Bp,Cp,Dp,Ep,14, 9);
    _GR_RL1(A,B,C,D,E, 2,15);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 7, 9);_GR_RL1(A,B,C,D,E, 3,12);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 0,11);
    _GR_RL1(A,B,C,D,E, 4, 5);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 9,13);_GR_RL1(A,B,C,D,E, 5, 8);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 2,15);
    _GR_RL1(A,B,C,D,E, 6, 7);_GR_RR1(Ap,Bp,Cp,Dp,Ep,11,15);_GR_RL1(A,B,C,D,E, 7, 9);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 4, 5);
    _GR_RL1(A,B,C,D,E, 8,11);_GR_RR1(Ap,Bp,Cp,Dp,Ep,13, 7);_GR_RL1(A,B,C,D,E, 9,13);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 6, 7);
    _GR_RL1(A,B,C,D,E,10,14);_GR_RR1(Ap,Bp,Cp,Dp,Ep,15, 8);_GR_RL1(A,B,C,D,E,11,15);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 8,11);
    _GR_RL1(A,B,C,D,E,12, 6);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 1,14);_GR_RL1(A,B,C,D,E,13, 7);_GR_RR1(Ap,Bp,Cp,Dp,Ep,10,14);
    _GR_RL1(A,B,C,D,E,14, 9);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 3,12);_GR_RL1(A,B,C,D,E,15, 8);_GR_RR1(Ap,Bp,Cp,Dp,Ep,12, 6);
    _GR_RL2(A,B,C,D,E, 7, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 6, 9);_GR_RL2(A,B,C,D,E, 4, 6);_GR_RR2(Ap,Bp,Cp,Dp,Ep,11,13);
    _GR_RL2(A,B,C,D,E,13, 8);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 3,15);_GR_RL2(A,B,C,D,E, 1,13);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 7, 7);
    _GR_RL2(A,B,C,D,E,10,11);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 0,12);_GR_RL2(A,B,C,D,E, 6, 9);_GR_RR2(Ap,Bp,Cp,Dp,Ep,13, 8);
    _GR_RL2(A,B,C,D,E,15, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 5, 9);_GR_RL2(A,B,C,D,E, 3,15);_GR_RR2(Ap,Bp,Cp,Dp,Ep,10,11);
    _GR_RL2(A,B,C,D,E,12, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep,14, 7);_GR_RL2(A,B,C,D,E, 0,12);_GR_RR2(Ap,Bp,Cp,Dp,Ep,15, 7);
    _GR_RL2(A,B,C,D,E, 9,15);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 8,12);_GR_RL2(A,B,C,D,E, 5, 9);_GR_RR2(Ap,Bp,Cp,Dp,Ep,12, 7);
    _GR_RL2(A,B,C,D,E, 2,11);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 4, 6);_GR_RL2(A,B,C,D,E,14, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 9,15);
    _GR_RL2(A,B,C,D,E,11,13);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 1,13);_GR_RL2(A,B,C,D,E, 8,12);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 2,11);
    _GR_RL3(A,B,C,D,E, 3,11);_GR_RR3(Ap,Bp,Cp,Dp,Ep,15, 9);_GR_RL3(A,B,C,D,E,10,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 5, 7);
    _GR_RL3(A,B,C,D,E,14, 6);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 1,15);_GR_RL3(A,B,C,D,E, 4, 7);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 3,11);
    _GR_RL3(A,B,C,D,E, 9,14);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 7, 8);_GR_RL3(A,B,C,D,E,15, 9);_GR_RR3(Ap,Bp,Cp,Dp,Ep,14, 6);
    _GR_RL3(A,B,C,D,E, 8,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 6, 6);_GR_RL3(A,B,C,D,E, 1,15);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 9,14);
    _GR_RL3(A,B,C,D,E, 2,14);_GR_RR3(Ap,Bp,Cp,Dp,Ep,11,12);_GR_RL3(A,B,C,D,E, 7, 8);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 8,13);
    _GR_RL3(A,B,C,D,E, 0,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep,12, 5);_GR_RL3(A,B,C,D,E, 6, 6);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 2,14);
    _GR_RL3(A,B,C,D,E,13, 5);_GR_RR3(Ap,Bp,Cp,Dp,Ep,10,13);_GR_RL3(A,B,C,D,E,11,12);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 0,13);
    _GR_RL3(A,B,C,D,E, 5, 7);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 4, 7);_GR_RL3(A,B,C,D,E,12, 5);_GR_RR3(Ap,Bp,Cp,Dp,Ep,13, 5);
    _GR_RL4(A,B,C,D,E, 1,11);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 8,15);_GR_RL4(A,B,C,D,E, 9,12);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 6, 5);
    _GR_RL4(A,B,C,D,E,11,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 4, 8);_GR_RL4(A,B,C,D,E,10,15);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 1,11);
    _GR_RL4(A,B,C,D,E, 0,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 3,14);_GR_RL4(A,B,C,D,E, 8,15);_GR_RR4(Ap,Bp,Cp,Dp,Ep,11,14);
    _GR_RL4(A,B,C,D,E,12, 9);_GR_RR4(Ap,Bp,Cp,Dp,Ep,15, 6);_GR_RL4(A,B,C,D,E, 4, 8);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 0,14);
    _GR_RL4(A,B,C,D,E,13, 9);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 5, 6);_GR_RL4(A,B,C,D,E, 3,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep,12, 9);
    _GR_RL4(A,B,C,D,E, 7, 5);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 2,12);_GR_RL4(A,B,C,D,E,15, 6);_GR_RR4(Ap,Bp,Cp,Dp,Ep,13, 9);
    _GR_RL4(A,B,C,D,E,14, 8);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 9,12);_GR_RL4(A,B,C,D,E, 5, 6);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 7, 5);
    _GR_RL4(A,B,C,D,E, 6, 5);_GR_RR4(Ap,Bp,Cp,Dp,Ep,10,15);_GR_RL4(A,B,C,D,E, 2,12);_GR_RR4(Ap,Bp,Cp,Dp,Ep,14, 8);
    _GR_RL5(A,B,C,D,E, 4, 9);_GR_RR5(Ap,Bp,Cp,Dp,Ep,12, 8);_GR_RL5(A,B,C,D,E, 0,15);_GR_RR5(Ap,Bp,Cp,Dp,Ep,15, 5);
    _GR_RL5(A,B,C,D,E, 5, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep,10,12);_GR_RL5(A,B,C,D,E, 9,11);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 4, 9);
    _GR_RL5(A,B,C,D,E, 7, 6);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 1,12);_GR_RL5(A,B,C,D,E,12, 8);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 5, 5);
    _GR_RL5(A,B,C,D,E, 2,13);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 8,14);_GR_RL5(A,B,C,D,E,10,12);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 7, 6);
    _GR_RL5(A,B,C,D,E,14, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 6, 8);_GR_RL5(A,B,C,D,E, 1,12);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 2,13);
    _GR_RL5(A,B,C,D,E, 3,13);_GR_RR5(Ap,Bp,Cp,Dp,Ep,13, 6);_GR_RL5(A,B,C,D,E, 8,14);_GR_RR5(Ap,Bp,Cp,Dp,Ep,14, 5);
    _GR_RL5(A,B,C,D,E,11,11);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 0,15);_GR_RL5(A,B,C,D,E, 6, 8);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 3,13);
    _GR_RL5(A,B,C,D,E,15, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 9,11);_GR_RL5(A,B,C,D,E,13, 6);_GR_RR5(Ap,Bp,Cp,Dp,Ep,11,11);

    uint32_t _tt=h1+C+Dp; h1=h2+D+Ep; h2=h3+E+Ap; h3=h4+A+Bp; h4=h0+B+Cp; h0=_tt;
    uint32_t _out[5]={h0,h1,h2,h3,h4};
    memcpy(out20,_out,20);
}

/* ── Target hashes em constant memory ───────────────────────────────
 * MULTI-ALVO v11.0: d_alvos_u32 saiu de __constant__ (limite 64KB)
 * e virou ponteiro p/ global memory dimensionado em runtime.
 * Idem d_alvos_ativos. d_num_alvos continua __constant__ (1 int).
 * ──────────────────────────────────────────────────────────────────── */

/* Single-alvo (legado, ainda usado para target_32 que cobre Taproot 32-byte) */
__constant__ uint8_t  d_hash160_alvo[20];
__constant__ uint8_t  d_target_32[32];

/* Multi-alvo: ponteiros device para arrays alocados via cudaMalloc.
 * Setados via cudaMemcpyToSymbol dos ponteiros (não dos dados). */
__device__ AlvoEntry *gd_alvos_sorted = NULL;   /* [g_num_alvos] sorted by h0 */
__device__ uint32_t  *gd_alvos_prefix = NULL;   /* [ALVOS_PREFIX_SIZE+1] */
__device__ int       *gd_alvos_ativos = NULL;   /* [g_num_alvos] active flags */

/* ════ BUFFER MULTI-MATCH (v10.16) ════
 * Em vez de um slot único que para o kernel no 1º match, mantemos um buffer
 * em anel de até MATCH_BUF_CAP registros. Cada match faz atomicAdd no contador,
 * grava 8 ints (si, wi|tid<<16, iter_lo, iter_hi, micro_off, orig_idx, _, _) e
 * desativa o próprio alvo (atomicExch em gd_alvos_ativos). O kernel NÃO seta
 * s_found — segue varrendo. O host drena o buffer continuamente. Só paramos
 * quando todos os alvos ficam inativos (active_count==0) ou a range se esgota. */
#define MATCH_BUF_CAP   4096   /* registros simultâneos antes de o host drenar */
#define MATCH_REC_INTS  10     /* ints por registro: +2 slots p/ os bits 64..127 do iter (128b) */
__device__ int       *gd_match_buf   = NULL;   /* [MATCH_BUF_CAP * MATCH_REC_INTS] */
__device__ int       *gd_match_count = NULL;   /* contador atômico de registros gravados */
__device__ int       *gd_active_count = NULL;  /* nº de alvos ainda ativos (para parada) */

__constant__ int      d_num_alvos;              /* total de alvos (1..N) */

/* Legado single-alvo: mantém para paths que dependem dele (recovery, etc). */
__constant__ uint32_t d_alvo_u32[5];
__constant__ int      d_gpu_mode;       /* GPU_MODE_HASH160, GPU_MODE_P2SH, etc */

#if MICRO_K > 0
__constant__ gfe_t    d_micro_x[MICRO_K];
__constant__ gfe_t    d_micro_y[MICRO_K];
#endif

__device__ __forceinline__
int gpu_memcmp(const uint8_t *a, const uint8_t *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return (int)a[i] - (int)b[i];
    }
    return 0;
}


#define _KR(a,b,c,d,e,f,g,h,i) {uint32_t _t1=(h)+_GS_EP1(e)+_GS_CH(e,f,g)+d_SHA256_K[i]+W[(i)&15];uint32_t _t2=_GS_EP0(a)+_GS_MAJ(a,b,c);(d)+=_t1;(h)=_t1+_t2;}
#define _KE(a,b,c,d,e,f,g,h,i) {W[(i)&15]+=_GS_SG1(W[((i)-2)&15])+W[((i)-7)&15]+_GS_SG0(W[((i)-15)&15]);uint32_t _t1=(h)+_GS_EP1(e)+_GS_CH(e,f,g)+d_SHA256_K[i]+W[(i)&15];uint32_t _t2=_GS_EP0(a)+_GS_MAJ(a,b,c);(d)+=_t1;(h)=_t1+_t2;}

__device__ __noinline__ int hash160_check_noinline(const gfe_t XI, const gfe_t YI,
                                                    uint32_t *out_h0,
                                                    uint32_t *out_h1,
                                                    uint32_t *out_h2,
                                                    uint32_t *out_h3,
                                                    uint32_t *out_h4) {
    uint32_t _prf = 0x02u | (uint32_t)(YI[0] & 1);
    uint32_t W[16];
    W[0]=(_prf<<24)|(uint32_t)(XI[3]>>40);
    W[1]=(uint32_t)((XI[3]>>8)&0xFFFFFFFFULL);
    W[2]=((uint32_t)(XI[3]&0xFFULL)<<24)|(uint32_t)(XI[2]>>40);
    W[3]=(uint32_t)((XI[2]>>8)&0xFFFFFFFFULL);
    W[4]=((uint32_t)(XI[2]&0xFFULL)<<24)|(uint32_t)(XI[1]>>40);
    W[5]=(uint32_t)((XI[1]>>8)&0xFFFFFFFFULL);
    W[6]=((uint32_t)(XI[1]&0xFFULL)<<24)|(uint32_t)(XI[0]>>40);
    W[7]=(uint32_t)((XI[0]>>8)&0xFFFFFFFFULL);
    W[8]=((uint32_t)(XI[0]&0xFFULL)<<24)|0x00800000u;
    W[9]=0;W[10]=0;W[11]=0;W[12]=0;W[13]=0;W[14]=0;W[15]=264u;
    uint32_t sa=0x6a09e667u,sb=0xbb67ae85u,sc=0x3c6ef372u,sd=0xa54ff53au,
             se=0x510e527fu,sf=0x9b05688cu,sg=0x1f83d9abu,sh=0x5be0cd19u;
    _KR(sa,sb,sc,sd,se,sf,sg,sh,0);_KR(sh,sa,sb,sc,sd,se,sf,sg,1);_KR(sg,sh,sa,sb,sc,sd,se,sf,2);_KR(sf,sg,sh,sa,sb,sc,sd,se,3);
    _KR(se,sf,sg,sh,sa,sb,sc,sd,4);_KR(sd,se,sf,sg,sh,sa,sb,sc,5);_KR(sc,sd,se,sf,sg,sh,sa,sb,6);_KR(sb,sc,sd,se,sf,sg,sh,sa,7);
    _KR(sa,sb,sc,sd,se,sf,sg,sh,8);_KR(sh,sa,sb,sc,sd,se,sf,sg,9);_KR(sg,sh,sa,sb,sc,sd,se,sf,10);_KR(sf,sg,sh,sa,sb,sc,sd,se,11);
    _KR(se,sf,sg,sh,sa,sb,sc,sd,12);_KR(sd,se,sf,sg,sh,sa,sb,sc,13);_KR(sc,sd,se,sf,sg,sh,sa,sb,14);_KR(sb,sc,sd,se,sf,sg,sh,sa,15);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,16);_KE(sh,sa,sb,sc,sd,se,sf,sg,17);_KE(sg,sh,sa,sb,sc,sd,se,sf,18);_KE(sf,sg,sh,sa,sb,sc,sd,se,19);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,20);_KE(sd,se,sf,sg,sh,sa,sb,sc,21);_KE(sc,sd,se,sf,sg,sh,sa,sb,22);_KE(sb,sc,sd,se,sf,sg,sh,sa,23);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,24);_KE(sh,sa,sb,sc,sd,se,sf,sg,25);_KE(sg,sh,sa,sb,sc,sd,se,sf,26);_KE(sf,sg,sh,sa,sb,sc,sd,se,27);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,28);_KE(sd,se,sf,sg,sh,sa,sb,sc,29);_KE(sc,sd,se,sf,sg,sh,sa,sb,30);_KE(sb,sc,sd,se,sf,sg,sh,sa,31);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,32);_KE(sh,sa,sb,sc,sd,se,sf,sg,33);_KE(sg,sh,sa,sb,sc,sd,se,sf,34);_KE(sf,sg,sh,sa,sb,sc,sd,se,35);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,36);_KE(sd,se,sf,sg,sh,sa,sb,sc,37);_KE(sc,sd,se,sf,sg,sh,sa,sb,38);_KE(sb,sc,sd,se,sf,sg,sh,sa,39);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,40);_KE(sh,sa,sb,sc,sd,se,sf,sg,41);_KE(sg,sh,sa,sb,sc,sd,se,sf,42);_KE(sf,sg,sh,sa,sb,sc,sd,se,43);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,44);_KE(sd,se,sf,sg,sh,sa,sb,sc,45);_KE(sc,sd,se,sf,sg,sh,sa,sb,46);_KE(sb,sc,sd,se,sf,sg,sh,sa,47);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,48);_KE(sh,sa,sb,sc,sd,se,sf,sg,49);_KE(sg,sh,sa,sb,sc,sd,se,sf,50);_KE(sf,sg,sh,sa,sb,sc,sd,se,51);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,52);_KE(sd,se,sf,sg,sh,sa,sb,sc,53);_KE(sc,sd,se,sf,sg,sh,sa,sb,54);_KE(sb,sc,sd,se,sf,sg,sh,sa,55);
    _KE(sa,sb,sc,sd,se,sf,sg,sh,56);_KE(sh,sa,sb,sc,sd,se,sf,sg,57);_KE(sg,sh,sa,sb,sc,sd,se,sf,58);_KE(sf,sg,sh,sa,sb,sc,sd,se,59);
    _KE(se,sf,sg,sh,sa,sb,sc,sd,60);_KE(sd,se,sf,sg,sh,sa,sb,sc,61);_KE(sc,sd,se,sf,sg,sh,sa,sb,62);_KE(sb,sc,sd,se,sf,sg,sh,sa,63);
    uint32_t X[16];
    X[0]=__byte_perm(0x6a09e667u+sa,0,0x0123);X[1]=__byte_perm(0xbb67ae85u+sb,0,0x0123);
    X[2]=__byte_perm(0x3c6ef372u+sc,0,0x0123);X[3]=__byte_perm(0xa54ff53au+sd,0,0x0123);
    X[4]=__byte_perm(0x510e527fu+se,0,0x0123);X[5]=__byte_perm(0x9b05688cu+sf,0,0x0123);
    X[6]=__byte_perm(0x1f83d9abu+sg,0,0x0123);X[7]=__byte_perm(0x5be0cd19u+sh,0,0x0123);
    X[8]=0x00000080u;X[9]=0;X[10]=0;X[11]=0;X[12]=0;X[13]=0;X[14]=256u;X[15]=0;
    uint32_t rh0=0x67452301u,rh1=0xEFCDAB89u,rh2=0x98BADCFEu,rh3=0x10325476u,rh4=0xC3D2E1F0u;
    uint32_t A=rh0,B=rh1,C=rh2,D=rh3,E=rh4,Ap=rh0,Bp=rh1,Cp=rh2,Dp=rh3,Ep=rh4;
    _GR_RL1(A,B,C,D,E, 0,11);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 5, 8);_GR_RL1(A,B,C,D,E, 1,14);_GR_RR1(Ap,Bp,Cp,Dp,Ep,14, 9);
    _GR_RL1(A,B,C,D,E, 2,15);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 7, 9);_GR_RL1(A,B,C,D,E, 3,12);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 0,11);
    _GR_RL1(A,B,C,D,E, 4, 5);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 9,13);_GR_RL1(A,B,C,D,E, 5, 8);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 2,15);
    _GR_RL1(A,B,C,D,E, 6, 7);_GR_RR1(Ap,Bp,Cp,Dp,Ep,11,15);_GR_RL1(A,B,C,D,E, 7, 9);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 4, 5);
    _GR_RL1(A,B,C,D,E, 8,11);_GR_RR1(Ap,Bp,Cp,Dp,Ep,13, 7);_GR_RL1(A,B,C,D,E, 9,13);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 6, 7);
    _GR_RL1(A,B,C,D,E,10,14);_GR_RR1(Ap,Bp,Cp,Dp,Ep,15, 8);_GR_RL1(A,B,C,D,E,11,15);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 8,11);
    _GR_RL1(A,B,C,D,E,12, 6);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 1,14);_GR_RL1(A,B,C,D,E,13, 7);_GR_RR1(Ap,Bp,Cp,Dp,Ep,10,14);
    _GR_RL1(A,B,C,D,E,14, 9);_GR_RR1(Ap,Bp,Cp,Dp,Ep, 3,12);_GR_RL1(A,B,C,D,E,15, 8);_GR_RR1(Ap,Bp,Cp,Dp,Ep,12, 6);
    _GR_RL2(A,B,C,D,E, 7, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 6, 9);_GR_RL2(A,B,C,D,E, 4, 6);_GR_RR2(Ap,Bp,Cp,Dp,Ep,11,13);
    _GR_RL2(A,B,C,D,E,13, 8);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 3,15);_GR_RL2(A,B,C,D,E, 1,13);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 7, 7);
    _GR_RL2(A,B,C,D,E,10,11);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 0,12);_GR_RL2(A,B,C,D,E, 6, 9);_GR_RR2(Ap,Bp,Cp,Dp,Ep,13, 8);
    _GR_RL2(A,B,C,D,E,15, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 5, 9);_GR_RL2(A,B,C,D,E, 3,15);_GR_RR2(Ap,Bp,Cp,Dp,Ep,10,11);
    _GR_RL2(A,B,C,D,E,12, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep,14, 7);_GR_RL2(A,B,C,D,E, 0,12);_GR_RR2(Ap,Bp,Cp,Dp,Ep,15, 7);
    _GR_RL2(A,B,C,D,E, 9,15);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 8,12);_GR_RL2(A,B,C,D,E, 5, 9);_GR_RR2(Ap,Bp,Cp,Dp,Ep,12, 7);
    _GR_RL2(A,B,C,D,E, 2,11);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 4, 6);_GR_RL2(A,B,C,D,E,14, 7);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 9,15);
    _GR_RL2(A,B,C,D,E,11,13);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 1,13);_GR_RL2(A,B,C,D,E, 8,12);_GR_RR2(Ap,Bp,Cp,Dp,Ep, 2,11);
    _GR_RL3(A,B,C,D,E, 3,11);_GR_RR3(Ap,Bp,Cp,Dp,Ep,15, 9);_GR_RL3(A,B,C,D,E,10,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 5, 7);
    _GR_RL3(A,B,C,D,E,14, 6);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 1,15);_GR_RL3(A,B,C,D,E, 4, 7);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 3,11);
    _GR_RL3(A,B,C,D,E, 9,14);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 7, 8);_GR_RL3(A,B,C,D,E,15, 9);_GR_RR3(Ap,Bp,Cp,Dp,Ep,14, 6);
    _GR_RL3(A,B,C,D,E, 8,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 6, 6);_GR_RL3(A,B,C,D,E, 1,15);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 9,14);
    _GR_RL3(A,B,C,D,E, 2,14);_GR_RR3(Ap,Bp,Cp,Dp,Ep,11,12);_GR_RL3(A,B,C,D,E, 7, 8);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 8,13);
    _GR_RL3(A,B,C,D,E, 0,13);_GR_RR3(Ap,Bp,Cp,Dp,Ep,12, 5);_GR_RL3(A,B,C,D,E, 6, 6);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 2,14);
    _GR_RL3(A,B,C,D,E,13, 5);_GR_RR3(Ap,Bp,Cp,Dp,Ep,10,13);_GR_RL3(A,B,C,D,E,11,12);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 0,13);
    _GR_RL3(A,B,C,D,E, 5, 7);_GR_RR3(Ap,Bp,Cp,Dp,Ep, 4, 7);_GR_RL3(A,B,C,D,E,12, 5);_GR_RR3(Ap,Bp,Cp,Dp,Ep,13, 5);
    _GR_RL4(A,B,C,D,E, 1,11);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 8,15);_GR_RL4(A,B,C,D,E, 9,12);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 6, 5);
    _GR_RL4(A,B,C,D,E,11,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 4, 8);_GR_RL4(A,B,C,D,E,10,15);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 1,11);
    _GR_RL4(A,B,C,D,E, 0,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 3,14);_GR_RL4(A,B,C,D,E, 8,15);_GR_RR4(Ap,Bp,Cp,Dp,Ep,11,14);
    _GR_RL4(A,B,C,D,E,12, 9);_GR_RR4(Ap,Bp,Cp,Dp,Ep,15, 6);_GR_RL4(A,B,C,D,E, 4, 8);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 0,14);
    _GR_RL4(A,B,C,D,E,13, 9);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 5, 6);_GR_RL4(A,B,C,D,E, 3,14);_GR_RR4(Ap,Bp,Cp,Dp,Ep,12, 9);
    _GR_RL4(A,B,C,D,E, 7, 5);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 2,12);_GR_RL4(A,B,C,D,E,15, 6);_GR_RR4(Ap,Bp,Cp,Dp,Ep,13, 9);
    _GR_RL4(A,B,C,D,E,14, 8);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 9,12);_GR_RL4(A,B,C,D,E, 5, 6);_GR_RR4(Ap,Bp,Cp,Dp,Ep, 7, 5);
    _GR_RL4(A,B,C,D,E, 6, 5);_GR_RR4(Ap,Bp,Cp,Dp,Ep,10,15);_GR_RL4(A,B,C,D,E, 2,12);_GR_RR4(Ap,Bp,Cp,Dp,Ep,14, 8);
    _GR_RL5(A,B,C,D,E, 4, 9);_GR_RR5(Ap,Bp,Cp,Dp,Ep,12, 8);_GR_RL5(A,B,C,D,E, 0,15);_GR_RR5(Ap,Bp,Cp,Dp,Ep,15, 5);
    _GR_RL5(A,B,C,D,E, 5, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep,10,12);_GR_RL5(A,B,C,D,E, 9,11);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 4, 9);
    _GR_RL5(A,B,C,D,E, 7, 6);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 1,12);_GR_RL5(A,B,C,D,E,12, 8);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 5, 5);
    _GR_RL5(A,B,C,D,E, 2,13);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 8,14);_GR_RL5(A,B,C,D,E,10,12);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 7, 6);
    _GR_RL5(A,B,C,D,E,14, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 6, 8);_GR_RL5(A,B,C,D,E, 1,12);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 2,13);
    _GR_RL5(A,B,C,D,E, 3,13);_GR_RR5(Ap,Bp,Cp,Dp,Ep,13, 6);_GR_RL5(A,B,C,D,E, 8,14);_GR_RR5(Ap,Bp,Cp,Dp,Ep,14, 5);
    _GR_RL5(A,B,C,D,E,11,11);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 0,15);_GR_RL5(A,B,C,D,E, 6, 8);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 3,13);
    _GR_RL5(A,B,C,D,E,15, 5);_GR_RR5(Ap,Bp,Cp,Dp,Ep, 9,11);_GR_RL5(A,B,C,D,E,13, 6);_GR_RR5(Ap,Bp,Cp,Dp,Ep,11,11);
    uint32_t _tt=rh1+C+Dp;
    uint32_t _h1=rh2+D+Ep, _h2=rh3+E+Ap, _h3=rh4+A+Bp, _h4=rh0+B+Cp;

    /* MULTI-ALVO v11.0: prefix table lookup O(1) em vez de scan linear.
     *
     * pfx = upper 24 bits de _tt → bucket de [start, end) no array sorted.
     * Bucket médio ≤1.25 alvo para 20M alvos uniformemente distribuídos.
     * 99% dos hashes: bucket vazio → 2 loads + 0 compares.
     *  1% dos hashes: bucket com 1-5 alvos → 2 loads + ~5 compares.
     *
     * vs. v10.13: era N compares por hash (catastrófico para N>10k). */
    uint32_t pfx = _tt >> (32 - ALVOS_PREFIX_BITS);
    uint32_t b_start = gd_alvos_prefix[pfx];
    uint32_t b_end   = gd_alvos_prefix[pfx + 1];
    for (uint32_t i = b_start; i < b_end; i++) {
        AlvoEntry e = gd_alvos_sorted[i];
        if (__builtin_expect(e.h0 == _tt, 0)) {
            if (e.h1 == _h1 && e.h2 == _h2 && e.h3 == _h3 && e.h4 == _h4) {
                if (gd_alvos_ativos[e.orig_idx]) {
                    *out_h0 = _tt; *out_h1 = _h1; *out_h2 = _h2; *out_h3 = _h3; *out_h4 = _h4;
                    return (int)e.orig_idx + 1; /* +1 para distinguir de "no match" (0) */
                }
            }
        }
    }
    return 0;
}

#define _FUSED_HASH160_CHECK(XI, YI, MICRO_OFF) do { \
    uint32_t _h0,_h1,_h2,_h3,_h4; \
    int _alvo_match = hash160_check_noinline((XI), (YI), &_h0, &_h1, &_h2, &_h3, &_h4); \
    if (_alvo_match) { \
        int _oi = _alvo_match - 1; /* índice 0-based do alvo */ \
        /* Desativa o alvo de forma atômica. Só o PRIMEIRO thread a desativá-lo \
         * grava o registro no buffer — evita duplicatas do mesmo alvo. Outros \
         * threads que acharem o mesmo alvo veem o flag já 0 e não regravam. */ \
        if (atomicExch(&gd_alvos_ativos[_oi], 0) != 0) { \
            int _widx = atomicAdd(gd_match_count, 1); /* índice de escrita MONOTÔNICO */ \
            int _slot = _widx % MATCH_BUF_CAP;        /* posição no anel */ \
            { \
                int *_rec = gd_match_buf + _slot * MATCH_REC_INTS; \
                unsigned __int128 _global_iter = (unsigned __int128)tid + ((unsigned __int128)m * BATCH_SIZE + (unsigned __int128)kk) * (unsigned __int128)GPU_THREADS; \
                _rec[1] = si; \
                _rec[2] = wi | (tid << 16); \
                _rec[3] = (int)((unsigned long long)_global_iter & 0xFFFFFFFFULL); \
                _rec[5] = (int)(((unsigned long long)_global_iter >> 32) & 0xFFFFFFFFULL); \
                _rec[8] = (int)((unsigned long long)(_global_iter >> 64) & 0xFFFFFFFFULL); \
                _rec[9] = (int)(((unsigned long long)(_global_iter >> 64) >> 32) & 0xFFFFFFFFULL); \
                _rec[6] = (MICRO_OFF); \
                _rec[7] = _oi; \
                __threadfence_system(); \
                _rec[0] = 1; /* marca registro pronto por último */ \
                __threadfence_system(); \
            } \
            /* Se era o último alvo ativo, sinaliza parada GLOBAL (worker não para por match). */ \
            if (atomicSub(gd_active_count, 1) <= 1) { \
                atomicExch(d_found_flag, 1); \
            } \
        } \
    } \
} while (0)


#define PERSISTENT_K_MAX 7   /* max init bits = log2(128) */

__global__ __launch_bounds__(GPU_THREADS, LB_BLOCKS_PER_SM)
void kernel_persistent(
    long long task_count_total,        /* total de tasks (legado, não usado mais) */
    int num_workers,
    unsigned long long *d_step_counter,/* counter 64-bit (legado, não usado mais no v10.14) */
    const gfe_t * __restrict__ d_pow_x,
    const gfe_t * __restrict__ d_pow_y,
    const gfe_t * __restrict__ d_starts_x,
    const gfe_t * __restrict__ d_starts_y,
    const unsigned __int128 * __restrict__ d_max_iter,
    int *d_found_flag,
    int *d_mapped_out,
    GPUBlockStatus *d_block_status,
    int *d_worker_next_si,             /* [num_workers] próximo si de cada worker (v10.14) */
    unsigned long long *d_tasks_done,  /* counter 64-bit de tasks completas */
    int num_steps_total_const          /* num_steps (para validação de display) */
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ long long s_task_id;
    __shared__ int s_found;   /* flag: alguém no bloco encontrou a chave */

    long long my_keys_total = 0;
    if (tid == 0) {
        d_block_status[bid].task_id = -1;
        d_block_status[bid].loops_done = 0;
        d_block_status[bid].max_iter = 0;
        d_block_status[bid].keys_total = 0;
        d_block_status[bid].worker_idx = -1;
        d_block_status[bid].actual_N = GPU_THREADS;
    }
    if (tid == 0) s_found = 0;
    __syncthreads();

    unsigned __int128 my_keys_total_local = 0;

    /* ══════════════════════════════════════════════════════════════════
     * DISPATCH FIXO BLOCO↔WORKER + FAN-OUT COOPERATIVO (v10.12)
     *
     * • Cada bloco bid processa EXCLUSIVAMENTE worker wi = bid
     * • As GPU_THREADS threads do bloco COOPERAM no mesmo (step, worker):
     *     - Thread tid começa em (worker_start + tid * step)
     *     - Avança em stride GPU_THREADS * step a cada iteração
     *     - Cobre 1/GPU_THREADS das max_iter chaves
     * • Cada bloco itera todos os steps em ordem (linear, sem oscilar)
     *
     * Garantias:
     *   - Bloco N ↔ Worker N (fixo, estável)
     *   - Cada (si, wi) processado completamente por 1 bloco
     *   - Nenhuma thread idle: 32 threads × 612 blocos = 19584 ativas
     *   - Display estável: bloco mostra "Bloco N / Worker N / Step si"
     *     durante toda execução; passa para si+1 só quando termina
     * ══════════════════════════════════════════════════════════════════ */
    int my_wi = bid;
    if (my_wi >= num_workers) return; /* Blocos extras: idle */

    /* Marca worker fixo no display */
    if (tid == 0) {
        d_block_status[bid].worker_idx = my_wi;
    }

    /* Carrega worker_start em afim uma vez (usado em cada step) */
    gfe_t my_start_x, my_start_y;
    gfe_copy(my_start_x, d_starts_x[my_wi]);
    gfe_copy(my_start_y, d_starts_y[my_wi]);

    /* v10.14: cada bloco começa do step indicado por d_worker_next_si[bid].
     * Isso permite retomar EXATAMENTE de onde parou (não há mais um global
     * ckpt_skip que comprometia o progresso heterogêneo entre workers). */
    int start_si = d_worker_next_si[bid];
    if (start_si < 0) start_si = 0;
    if (start_si > num_steps_total_const) start_si = num_steps_total_const;

#if STEP_ONLY_EXHAUSTIVE
    /* Modo exaustivo: pula direto para o PRIMEIRO step sequencial, varrendo todos
     * os SEQ_STEPS_COUNT steps finais (S = SEQ_STEPS_COUNT, ..., 2, 1). Os steps
     * sequenciais ocupam os últimos SEQ_STEPS_COUNT índices [num_steps-SEQ, num_steps).
     * Respeita o checkpoint: se o worker já passou desse ponto, mantém start_si. */
    {
        int seq_start = num_steps_total_const - SEQ_STEPS_COUNT;
        if (seq_start < 0) seq_start = 0;
        if (start_si < seq_start) start_si = seq_start;
    }
#endif

    for (int my_si = start_si; my_si < num_steps_total_const; my_si++) {
        if (tid == 0) {
            s_found = s_found | *d_found_flag;
        }
        __syncthreads();
        if (s_found || *d_found_flag) {
            /* Parada global: TODOS os alvos foram achados. Salva progresso e sai.
             * Uniforme: todos os threads veem s_found (sincronizado pela barreira
             * acima) e *d_found_flag (global) com o mesmo valor → saída sem deadlock. */
            if (tid == 0) d_worker_next_si[bid] = my_si;
            return;
        }

        unsigned __int128 step_max_iter = d_max_iter[my_si];
        gfe_t step_x, step_y;
        gfe_copy(step_x, d_pow_x[my_si]);
        gfe_copy(step_y, d_pow_y[my_si]);

        /* Display */
        if (tid == 0) {
            d_block_status[bid].task_id = (long long)my_si * (long long)num_workers + (long long)my_wi;
            d_block_status[bid].step_idx = my_si;
            d_block_status[bid].worker_idx = my_wi;
            d_block_status[bid].loops_done = 0;
            d_block_status[bid].max_iter = clamp_u128_to_i64(step_max_iter);
            /* v10.17: resume_m só vale para o step retomado (== start_si na entrada).
             * Para qualquer step POSTERIOR ao start_si, começa do zero. */
            if (my_si != start_si) d_block_status[bid].resume_m = 0;
        }
        __syncthreads();

        if (step_max_iter == 0) {
            __syncthreads();
            if (tid == 0) {
                d_worker_next_si[bid] = my_si + 1;
                atomicAdd(d_tasks_done, 1ULL);
            }
            continue;
        }

        /* ════ FAN-OUT INICIAL ════
         * Thread tid precisa começar em: pt_initial = worker_start + tid * step
         * Calculamos via: começando do worker_start, fazemos `tid` adições
         * sequenciais de step. Cada thread faz suas próprias adições — paralelo
         * mas redundante. Custo: tid adições afim por thread, ~32×7 = 224 adds
         * para o bloco inteiro, mas amortizado pelas ~max_iter/32 iterações
         * subsequentes que são puro stride. Para max_iter alto, custo é negligível.
         *
         * Alternativa cooperativa (binária): seria 5 syncs + log2 etapas,
         * mas adds afim são baratos e o método sequencial mantém código simples. */
        GFePt pt;
        gfe_copy(pt.X, my_start_x);
        gfe_copy(pt.Y, my_start_y);
        pt.Z[0] = 1; pt.Z[1] = pt.Z[2] = pt.Z[3] = 0;
        for (int adv = 0; adv < tid; adv++) {
            gfept_add_aff(&pt, &pt, step_x, step_y);
        }

        /* stride_x/y = GPU_THREADS * step. Calculamos via doublings + adds. */
        /* GPU_THREADS = 32. 32 = 2^5. Então stride = step << 5 (no grupo).
         * Em coords afim isso é: dbl(dbl(dbl(dbl(dbl(step))))) = 32×step. */
        gfe_t stride_x, stride_y;
        {
            /* Começar de (step_x, step_y) em coords afim (Z=1 implícito).
             * 5 dobramentos afim → 32×step. */
            gfe_t cur_x, cur_y;
            gfe_copy(cur_x, step_x);
            gfe_copy(cur_y, step_y);
            for (int d = 0; d < GPU_THREADS_LOG2; d++) {
                gfe_t nx, ny;
                gfe_aff_dbl(nx, ny, cur_x, cur_y);
                gfe_copy(cur_x, nx);
                gfe_copy(cur_y, ny);
            }
            gfe_copy(stride_x, cur_x);
            gfe_copy(stride_y, cur_y);
        }

        /* Cada thread processa ceil(max_iter / GPU_THREADS) chaves */
        unsigned __int128 my_iters = (step_max_iter + GPU_THREADS - 1) / GPU_THREADS;
        (void)my_iters;
        /* Mas o último thread pode ter menos: precisa cobrir só [tid, max_iter) com stride 32 */
        /* my_iters real = floor((max_iter - tid - 1) / 32) + 1 se tid < max_iter, senão 0 */
        unsigned __int128 my_iters_real;
        if ((unsigned __int128)tid >= step_max_iter) my_iters_real = 0;
        else my_iters_real = ((step_max_iter - (unsigned __int128)tid - 1) / GPU_THREADS) + 1;

        unsigned __int128 total_batches = (my_iters_real + BATCH_SIZE - 1) / BATCH_SIZE;
        gfe_t batch_X[BATCH_SIZE], batch_Y[BATCH_SIZE], batch_Z[BATCH_SIZE], prefix[BATCH_SIZE];

        /* v10.17: retomada dentro do step. Se este worker já avançou m_start
         * rodadas neste step num lançamento anterior, pula direto para lá
         * adiantando o ponto pt por m_start*BATCH_SIZE strides (sem hashear). */
        unsigned __int128 m_start = (unsigned __int128)(unsigned long long)d_block_status[bid].resume_m;
        if (m_start > total_batches) m_start = total_batches;
        for (unsigned __int128 adv = 0; adv < m_start * BATCH_SIZE; adv++) {
            gfept_add_aff(&pt, &pt, stride_x, stride_y);
        }

        for (unsigned __int128 m = m_start; m < total_batches; m++) {
            if (s_found) break;
            if (((unsigned long long)m & 4095ULL) == 0 && *d_found_flag) { s_found = 1; break; }

            gfe_copy(batch_X[0], pt.X); gfe_copy(batch_Y[0], pt.Y); gfe_copy(batch_Z[0], pt.Z);
            gfe_copy(prefix[0], pt.Z);

            for (int kk = 1; kk < BATCH_SIZE; kk++) {
                gfept_add_aff(&pt, &pt, stride_x, stride_y);
                gfe_copy(batch_X[kk], pt.X); gfe_copy(batch_Y[kk], pt.Y); gfe_copy(batch_Z[kk], pt.Z);
                gfe_mul(prefix[kk], prefix[kk-1], pt.Z);
            }
            gfept_add_aff(&pt, &pt, stride_x, stride_y);

            gfe_t inv;
            gfe_inv(inv, prefix[BATCH_SIZE-1]);

            for (int kk = BATCH_SIZE-1; kk >= 0; kk--) {
                gfe_t zi;
                if (kk > 0) {
                    gfe_mul(zi, inv, prefix[kk-1]);
                    gfe_mul(inv, inv, batch_Z[kk]);
                } else {
                    gfe_copy(zi, inv);
                }

                /* índice da chave deste ponto: tid + (m*BATCH_SIZE + kk)*GPU_THREADS.
                 * Só processa se dentro de [0, my_iters_real) — guarda contra
                 * o último batch parcial. */
                unsigned __int128 k_idx = m * BATCH_SIZE + (unsigned __int128)kk;
                if (k_idx >= my_iters_real) continue;

                gfe_t zi2, xi, yi;
                gfe_sqr(zi2, zi);
                gfe_mul(xi, batch_X[kk], zi2);
                gfe_mul(yi, batch_Y[kk], zi2); gfe_mul(yi, yi, zi);

                int si = my_si;
                int wi = my_wi;
                _FUSED_HASH160_CHECK(xi, yi, 0);
                if (s_found) break;
            }

            /* Update display + salva progresso de retomada (tid==0). Campos do
             * display são long long: saturamos em LLONG_MAX (sem truncar com lixo). */
            if (tid == 0 && ((unsigned long long)m & 31ULL) == 0) {
                d_block_status[bid].loops_done = clamp_u128_to_i64((m + 1) * BATCH_SIZE * GPU_THREADS);
                d_block_status[bid].resume_m   = clamp_u128_to_i64(m + 1);   /* v10.17: próxima rodada a processar */
            }
        }

        my_keys_total_local += step_max_iter;

        /* NOTA: sem __syncthreads() aqui. Os threads podem sair do loop de batches
         * em iterações diferentes (break por s_found/d_found_flag), então uma barreira
         * neste ponto seria divergente e causaria DEADLOCK (threads que saíram esperam
         * os que ainda não saíram, que nunca chegam). Só tid==0 escreve abaixo, sem
         * leitura cruzada entre threads, então a barreira não é necessária. */
        if (tid == 0) {
            d_block_status[bid].keys_total = clamp_u128_to_i64(my_keys_total_local);
            d_block_status[bid].loops_done = clamp_u128_to_i64(step_max_iter);
            d_worker_next_si[bid] = my_si + 1;
            atomicAdd(d_tasks_done, 1ULL);
        }
        /* Saída por parada global é tratada de forma UNIFORME no topo do loop
         * (após o __syncthreads), evitando break divergente + barreira = deadlock. */
    }

    /* Worker terminou todos os steps */
    if (tid == 0) {
        d_worker_next_si[bid] = num_steps_total_const;
    }
}


typedef struct {
    fe_t pow_x[1], pow_y[1]; /* mantém indexação [0] para compatibilidade */
} StepData;

static StepData *g_step_data = NULL;  /* [num_steps] */
static int g_num_steps_pre = 0;

typedef struct {
    char hex[130]; /* até 256 bits em hex */
} StepEntry;

static StepEntry *g_steps_hex = NULL; /* Array de steps em hex (apenas se carregado de arquivo) */
static int g_num_steps_hex = 0;

static int g_step_min_mag = 0;
static int g_step_max_mag = 0;
static int g_step_per_mag = 0;
static int g_step_log_count = 0;   /* total de steps logarítmicos */
static int g_step_seq_count = 100; /* steps sequenciais no final */
static int g_step_dynamic = 0;    /* 1 = on-demand, 0 = g_steps_hex */


static void compute_step_value(int si, mpz_t result, mpz_t tmp1, mpz_t tmp2) {
    if (si >= g_step_log_count) {
        int seq_idx = si - g_step_log_count;
        mpz_set_ui(result, (unsigned long)(g_step_seq_count - seq_idx));
        return;
    }
    int mag_index = si / g_step_per_mag;
    int j = (g_step_per_mag - 1) - (si % g_step_per_mag);
    int m = g_step_max_mag - mag_index;

    mpz_ui_pow_ui(tmp1, 2, m);               /* tmp1 = 2^m = mag_base */
    mpz_tdiv_q_ui(tmp2, tmp1, (unsigned long)g_step_per_mag); /* tmp2 = step_inc */
    if (mpz_sgn(tmp2) == 0) mpz_set_ui(tmp2, 1);

    mpz_mul_ui(result, tmp2, (unsigned long)j);
    mpz_add(result, result, tmp1);
    if (mpz_sgn(result) == 0) mpz_set_ui(result, 1);
}

static void get_step_value(int si, mpz_t result, mpz_t tmp1, mpz_t tmp2) {
    if (g_step_dynamic) {
        compute_step_value(si, result, tmp1, tmp2);
    } else {
        mpz_set_str(result, g_steps_hex[si].hex, 16);
    }
}

typedef struct { int tid; int nt; int ns; int offset; } PrecompStepArg;

static void *precomp_step_thread(void *arg) {
    PrecompStepArg *a = (PrecompStepArg*)arg;
    int base_offset = a->offset;

    for (int si = a->tid; si < a->ns; si += a->nt) {
        StepData *sd = &g_step_data[si - base_offset];

        mpz_t salto_mpz, _t1, _t2; mpz_inits(salto_mpz, _t1, _t2, NULL);
        get_step_value(si, salto_mpz, _t1, _t2);

        uint8_t sb[32]; mpz_export_32be(salto_mpz, sb);
        FePt S_host; fe_scalar_mul(&S_host, sb);

        if (!fept_is_inf(&S_host)) {
            fe_t Zi, Zi2;
            fe_inv(Zi, S_host.Z); fe_sqr(Zi2, Zi);
            fe_mul(sd->pow_x[0], S_host.X, Zi2);
            fe_mul(sd->pow_y[0], S_host.Y, Zi2);
            fe_mul(sd->pow_y[0], sd->pow_y[0], Zi);
        } else {
            memset(sd->pow_x[0], 0, sizeof(fe_t));
            memset(sd->pow_y[0], 0, sizeof(fe_t));
        }

        mpz_clears(salto_mpz, _t1, _t2, NULL);
    }
    return NULL;
}

static void precomp_all_steps(int ns, int ncpu) {
    g_num_steps_pre = ns;
    g_step_data = (StepData*)calloc(ns, sizeof(StepData));
    if (!g_step_data) { fprintf(stderr, "FATAL: calloc StepData\n"); return; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (ncpu > ns) ncpu = ns;
    pthread_t *thr = (pthread_t*)malloc(ncpu * sizeof(pthread_t));
    PrecompStepArg *args = (PrecompStepArg*)malloc(ncpu * sizeof(PrecompStepArg));
    for (int i = 0; i < ncpu; i++) {
        args[i] = (PrecompStepArg){i, ncpu, ns, 0};
        pthread_create(&thr[i], NULL, precomp_step_thread, &args[i]);
    }
    for (int i = 0; i < ncpu; i++) pthread_join(thr[i], NULL);
    free(thr); free(args);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("[Precomp Steps] %d steps precomputados em %.0f ms (%d threads CPU).\n", ns, ms, ncpu);
}

// Função que calcula como dividir um espaço de busca gigante em blocos menores (steps)
// baseados em "magnitudes" (potências de 2), ideal para varredura em chaves criptográficas.
static void gerar_steps_logaritmicos(
    const char *inicio_hex, const char *fim_hex, // Strings com o início e o fim do range em Hexadecimal
    int num_workers, int target_total_steps)     // Número de threads e a quantidade "alvo" de divisões totais
{
    // Declara variáveis do tipo mpz_t (da biblioteca GMP). Elas servem para armazenar 
    // "números gigantes" que não cabem nas variáveis normais do C (como int ou long).
    mpz_t inicio, fim, range; 
    mpz_inits(inicio, fim, range, NULL); // Aloca espaço na memória para essas variáveis

    // Converte os textos hexadecimais passados na função em números gigantes (base 16)
    mpz_set_str(inicio, inicio_hex, 16); // Define o ponto de partida
    mpz_set_str(fim, fim_hex, 16);       // Define o ponto de chegada
    
    // Subtrai o inicio do fim (range = fim - inicio) para descobrir o tamanho total do espaço a ser buscado
    mpz_sub(range, fim, inicio);

    // Conta quantos bits são necessários para representar o tamanho total desse espaço
    // Ex: Um range de 0 a 15 usa 4 bits. Isso define o escopo do nosso trabalho.
    int range_bits = (int)mpz_sizeinbase(range, 2); 
    printf("[StepGen] Range: %d bits\n", range_bits);

    int min_mag = 0;              // Magnitude mínima é 0 (representa variações de 2^0, ou seja, 1)
    int max_mag = range_bits - 1; // Magnitude máxima baseada no tamanho dos bits do range

    // Proteção: impede que a magnitude máxima seja um número negativo (caso o range fosse muito pequeno ou 0)
    if (max_mag < 0) max_mag = 0; 
    
    // Proteção: Limita a magnitude a 255 bits. 
    // MODIFIQUE AQUI: Se for trabalhar com RSA ou chaves maiores que 256 bits, aumente este limite.
    if (max_mag > 255) max_mag = 255; 

    // Calcula quantas potências de 2 diferentes (magnitudes) existem dentro do nosso espaço
    int num_mags = max_mag - min_mag + 1; 
    if (num_mags < 1) num_mags = 1; // Garante que teremos pelo menos 1 grupo de magnitude para trabalhar
    
    // Descobre quantos passos devem ser gerados POR magnitude.
    // Pega o alvo total de passos que o usuário pediu e divide pelo número de magnitudes.
    // int steps_per_mag = target_total_steps / num_mags; // ORIGINAL
    int steps_per_mag = STEPS_PER_MAGNITUDE; 
    if (steps_per_mag < 1) steps_per_mag = 1; // Garante no mínimo 1 passo por magnitude

    // Recalcula o total exato de passos logarítmicos reais.
    // (A divisão anterior pode ter ignorado o resto, então recalcularmos para ter o número preciso)
    // int total_log = 2218400; // vai gerar 2240584 steps
    // int total_log = num_mags * steps_per_mag;  // ORIGINAL
    int total_log = num_mags * steps_per_mag; 
    
    // Adiciona passos lineares/sequenciais. 
    // MODIFIQUE AQUI: 100 é o padrão. Aumente se quiser fazer uma varredura mais minuciosa e sequencial nas bordas do range.
    int seq_count = SEQ_STEPS_COUNT; 

    // =========================================================================
    // SALVANDO CONFIGURAÇÕES GLOBAIS
    // As variáveis abaixo começam com 'g_' (globais). O código as configura aqui
    // para que o resto do programa saiba como gerar o trabalho na hora (on-demand),
    // sem precisar pré-calcular e lotar a memória RAM.
    // =========================================================================
    g_step_min_mag   = min_mag;       // Define de onde as magnitudes começam
    g_step_max_mag   = max_mag;       // Define onde as magnitudes terminam
    g_step_per_mag   = steps_per_mag; // Define quantos blocos cada magnitude terá
    g_step_log_count = total_log;     // Define a quantidade total de blocos logarítmicos
    g_step_seq_count = seq_count;     // Define a quantidade de blocos sequenciais
    g_step_dynamic   = 1;             // Ativa a geração de passos sob demanda (dinâmica)
    
    // Soma total de todos os passos (os espalhados + os em sequência)
    g_num_steps_hex  = total_log + seq_count; 

    // Imprime um resumo da configuração gerada para o usuário no terminal
    printf("[StepGen] Magnitudes: %d a %d (%d magnitudes, cobertura 100%%)\n",
           min_mag, max_mag, num_mags);
    printf("[StepGen] Steps/magnitude: %d | Total: %d steps (ON-DEMAND — zero RAM)\n",
           steps_per_mag, g_num_steps_hex);

    // Etapa crucial: Limpa os números gigantes da memória. 
    // Se você remover isso, o programa vai sofrer de "Memory Leak" e travar o PC com o tempo.
    mpz_clears(inicio, fim, range, NULL);
}

static int carregar_steps_arquivo(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;

    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\r' || *p == '#' || *p == 0) continue;
        count++;
    }
    if (count == 0) { fclose(f); return 0; }

    g_steps_hex = (StepEntry*)calloc(count, sizeof(StepEntry));
    g_num_steps_hex = 0;

    rewind(f);
    while (fgets(line, sizeof(line), f)) {
        char *p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\r' || *p == '#' || *p == 0) continue;
        int len = strlen(p);
        while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r' ||
               p[len-1] == ',' || p[len-1] == ' ')) len--;
        p[len] = 0;

        mpz_t val; mpz_init(val);
        if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
            mpz_set_str(val, p+2, 16);
        else
            mpz_set_str(val, p, 10);

        if (mpz_sgn(val) > 0) {
            char tmp[260];
            mpz_get_str(tmp, 16, val);
            snprintf(g_steps_hex[g_num_steps_hex].hex,
                     sizeof(g_steps_hex[g_num_steps_hex].hex), "%s", tmp);
            g_num_steps_hex++;
        }
        mpz_clear(val);
    }
    fclose(f);
    printf("[StepLoad] Carregados %d steps de %s\n", g_num_steps_hex, filename);
    return g_num_steps_hex;
}


static cudaStream_t g_persistent_stream = 0;
static int                *g_d_found_flag = NULL;
static int                *g_h_mapped = NULL;
static int                *g_d_mapped = NULL;
/* v10.16 buffer multi-match */
static int                *g_h_match_buf = NULL;
static int                *g_d_match_buf = NULL;
static int                *g_h_match_count = NULL;
static int                *g_d_match_count = NULL;
static int                *g_d_active_count = NULL;
static unsigned long long *g_d_task_counter = NULL;  /* atomicAdd 64-bit (escala >2G tasks) */
static unsigned long long *g_h_tasks_done   = NULL;  /* pinned: tasks completas (substitui g_h_mapped[4]) */
static unsigned long long *g_d_tasks_done   = NULL;  /* device pointer mapped do mesmo */
static GPUBlockStatus *g_h_block_status = NULL;
static GPUBlockStatus *g_d_block_status = NULL;
int g_persistent_blocks = 0;

/* ── PROGRESS PER-WORKER (v10.14) ─────────────────────────────────────
 * Cada bloco bid tem worker fixo wi = bid. Cada worker tem seu próprio
 * "próximo step a processar" — armazenado aqui. No relauncamento ou
 * retomada de checkpoint, cada bloco lê seu valor inicial daqui.
 *
 * Sem esse array, o checkpoint global (1 número só) é insuficiente para
 * representar 612 workers que podem estar em si's diferentes quando
 * a busca foi interrompida (por achado, falha, ou shutdown).
 * ──────────────────────────────────────────────────────────────────── */
int *g_h_worker_next_si = NULL; /* pinned: [num_workers] */
static int *g_d_worker_next_si = NULL; /* device pointer do mesmo */

static int gpu_persistent_alloc(long long total_tasks, int persistent_blocks) {
    (void)total_tasks;
    g_persistent_blocks = persistent_blocks;
    CUDA_CHECK(cudaStreamCreate(&g_persistent_stream));
    CUDA_CHECK(cudaMalloc(&g_d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_task_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_d_task_counter, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc(&g_h_mapped, MATCH_REC_INTS * sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_mapped, g_h_mapped, 0));
    memset(g_h_mapped, 0, MATCH_REC_INTS * sizeof(int));

    /* ── BUFFER MULTI-MATCH (v10.16) ──
     * Ring de MATCH_BUF_CAP registros em memória mapeada (host lê direto).
     * gd_match_buf/gd_match_count são símbolos __device__: setados via
     * cudaMemcpyToSymbol com os ponteiros de device. gd_active_count guarda
     * quantos alvos ainda estão ativos (inicia em g_num_alvos). */
    CUDA_CHECK(cudaHostAlloc(&g_h_match_buf, (size_t)MATCH_BUF_CAP * MATCH_REC_INTS * sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_match_buf, g_h_match_buf, 0));
    memset(g_h_match_buf, 0, (size_t)MATCH_BUF_CAP * MATCH_REC_INTS * sizeof(int));
    CUDA_CHECK(cudaHostAlloc(&g_h_match_count, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_match_count, g_h_match_count, 0));
    *g_h_match_count = 0;
    CUDA_CHECK(cudaMalloc(&g_d_active_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(gd_match_buf,   &g_d_match_buf,   sizeof(void*)));
    CUDA_CHECK(cudaMemcpyToSymbol(gd_match_count, &g_d_match_count, sizeof(void*)));
    CUDA_CHECK(cudaMemcpyToSymbol(gd_active_count,&g_d_active_count,sizeof(void*)));
    CUDA_CHECK(cudaHostAlloc(&g_h_tasks_done, sizeof(unsigned long long), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_tasks_done, g_h_tasks_done, 0));
    *g_h_tasks_done = 0ULL;
    /* Aloca progress per-worker em pinned memory (acesso direto host↔device).
     * Tamanho = persistent_blocks * sizeof(int) ~ 612*4 = 2.4 KB. Inicializado a 0. */
    CUDA_CHECK(cudaHostAlloc(&g_h_worker_next_si, persistent_blocks * sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_worker_next_si, g_h_worker_next_si, 0));
    memset(g_h_worker_next_si, 0, persistent_blocks * sizeof(int));
    CUDA_CHECK(cudaHostAlloc(&g_h_block_status,
        (size_t)persistent_blocks * sizeof(GPUBlockStatus), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_d_block_status, g_h_block_status, 0));
    memset(g_h_block_status, 0, (size_t)persistent_blocks * sizeof(GPUBlockStatus));
    for (int i = 0; i < persistent_blocks; i++) {
        g_h_block_status[i].task_id = -1;
        g_h_block_status[i].step_idx = -1;
        g_h_block_status[i].worker_idx = -1;
    }
    printf("[Persistent Alloc] Block status: %d blocos × %zu bytes = %.1f KB (pinned)\n",
           persistent_blocks, sizeof(GPUBlockStatus),
           persistent_blocks * sizeof(GPUBlockStatus) / 1024.0);
    return 1;
}

static void gpu_persistent_free(void) {
    if (g_d_found_flag) cudaFree(g_d_found_flag);
    if (g_d_task_counter) cudaFree(g_d_task_counter);
    if (g_h_mapped) cudaFreeHost(g_h_mapped);
    if (g_h_tasks_done) cudaFreeHost(g_h_tasks_done);
    if (g_h_worker_next_si) cudaFreeHost(g_h_worker_next_si);
    if (g_h_block_status) cudaFreeHost(g_h_block_status);
    if (g_persistent_stream) cudaStreamDestroy(g_persistent_stream);
    g_d_found_flag = NULL;
    g_d_task_counter = NULL;
    g_h_tasks_done = g_d_tasks_done = NULL;
    g_h_worker_next_si = g_d_worker_next_si = NULL;
    g_h_mapped = NULL; g_d_mapped = NULL;
    g_h_block_status = NULL; g_d_block_status = NULL;
    g_persistent_stream = 0;
}

static gfe_t *d_gpu_pow_x = NULL, *d_gpu_pow_y = NULL;
static gfe_t *d_gpu_starts_x = NULL, *d_gpu_starts_y = NULL;
static gfe_t *d_gpu_one_z = NULL;
static unsigned __int128 *d_gpu_max_iter = NULL;  /* [num_steps] (128 bits) */

static void gpu_free_data(void) {
    if (d_gpu_pow_x) cudaFree(d_gpu_pow_x); if (d_gpu_pow_y) cudaFree(d_gpu_pow_y);
    if (d_gpu_starts_x) cudaFree(d_gpu_starts_x); if (d_gpu_starts_y) cudaFree(d_gpu_starts_y);
    if (d_gpu_max_iter) cudaFree(d_gpu_max_iter);
    if (d_gpu_one_z) cudaFree(d_gpu_one_z);
    d_gpu_pow_x=d_gpu_pow_y=d_gpu_starts_x=d_gpu_starts_y=d_gpu_one_z=NULL;
    d_gpu_max_iter=NULL;
}


static void rainha_dos_processos(int num_workers) {
    int persistent_blocks = g_persistent_blocks;
    char final_key_hex[130] = {0}; // Captura a chave para uso fora do loop

    printf("\033[94m[Rainha v10.6] task=(si,wi) uniforme | %d threads/bloco × %d blocos = %d tasks concorrentes\033[0m\n",
           GPU_THREADS, persistent_blocks, GPU_THREADS * persistent_blocks);
    printf("[Persistent] %d workers | %d steps | %lld tasks (steps decrescentes: maior→menor)\n",
           num_workers, num_steps_total, task_count);

    int max_display = 54;  /* linhas de display — round-robin sobre TODOS os workers */
    if (persistent_blocks < max_display) max_display = persistent_blocks;

    int total_pages = (persistent_blocks + max_display - 1) / max_display;
    printf("[Display] %d linhas × round-robin → %d workers em %d páginas de %.0f ms\n",
           max_display, num_workers, total_pages,
           (double)total_pages * DISPLAY_MS);

    for (int i = max_display - 1; i >= 0; i--) printf("[Slot %02d] Aguardando...\n", i);
    printf("\n");

    struct timespec ts_display, ts_ckpt, ts_start;
    pthread_t ckpt_thread;
    int ckpt_thread_started = 0;
    long long prev_keys = 0;
    mpz_t _d_pos, _d_step, _d_off, _d_tmp1, _d_tmp2;
    mpz_inits(_d_pos, _d_step, _d_off, _d_tmp1, _d_tmp2, NULL);
    /* v10.14: relaunch_ckpt removido — progresso fica em g_h_worker_next_si */
    cudaError_t qs;

relaunch_kernel:
    CUDA_CHECK(cudaMemset(g_d_task_counter, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_d_found_flag, 0, sizeof(int)));
    memset(g_h_mapped, 0, MATCH_REC_INTS * sizeof(int));
    /* v10.16: zera o buffer multi-match e recarrega o contador de alvos ativos
     * a partir do espelho host (alvos já achados ficam inativos). */
    *g_h_match_count = 0;
    memset(g_h_match_buf, 0, (size_t)MATCH_BUF_CAP * MATCH_REC_INTS * sizeof(int));
    {
        int _ativos = 0;
        for (int ai = 0; ai < g_num_alvos; ai++) if (g_alvos_ativos_host[ai]) _ativos++;
        CUDA_CHECK(cudaMemcpy(g_d_active_count, &_ativos, sizeof(int), cudaMemcpyHostToDevice));
    }
    __sync_synchronize();
    /* v10.14: progresso preservado em g_h_worker_next_si[bid] por bloco.
     * Não precisa zerar — kernel lê o valor inicial de cada bloco daqui. */
    prev_keys = 0;

    kernel_persistent<<<persistent_blocks, GPU_THREADS, 0, g_persistent_stream>>>(
        task_count,
        num_workers,
        g_d_task_counter,
        d_gpu_pow_x, d_gpu_pow_y,
        d_gpu_starts_x, d_gpu_starts_y,
        d_gpu_max_iter,
        g_d_found_flag, g_d_mapped,
        g_d_block_status,
        g_d_worker_next_si,
        g_d_tasks_done,
        num_steps_total
    );

    clock_gettime(CLOCK_MONOTONIC, &ts_display);
    clock_gettime(CLOCK_MONOTONIC, &ts_ckpt);
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    if (!ckpt_thread_started) {
        pthread_create(&ckpt_thread, NULL, checkpoint_thread_func, NULL);
        ckpt_thread_started = 1;
    }

    long long g_drained = 0;   /* índice de leitura MONOTÔNICO do ring multi-match */
    while ((qs = cudaStreamQuery(g_persistent_stream)) == cudaErrorNotReady) {
        __sync_synchronize();
        /* v10.16: DRENA o buffer multi-match com índices MONOTÔNICOS. O kernel
         * anexa cada achado em write_idx (atomicAdd) e segue varrendo. O host
         * consome [g_drained, write_idx) sem nunca decrementar o escritor — assim
         * cada registro tem slot único e nenhum match é perdido (o bug anterior
         * subtraía do contador e corrompia o índice de slots). */
        long long _widx = (long long)(*((volatile int *)g_h_match_count));
        while (g_drained < _widx) {
            int _slot = (int)(g_drained % MATCH_BUF_CAP);
            volatile int *_rec = (volatile int *)(g_h_match_buf + _slot * MATCH_REC_INTS);
            if (!_rec[0]) break;                        /* registro ainda não pronto — espera próximo poll */
            /* Copia o registro para g_h_mapped — o código de verificação abaixo
             * consome g_h_mapped[1..9]. */
            for (int _q = 0; _q < MATCH_REC_INTS; _q++) g_h_mapped[_q] = _rec[_q];
            __sync_synchronize();
        {
            int _mc = 1; (void)_mc;
            int found_si   = g_h_mapped[1];
            int found_wi   = g_h_mapped[2] & 0xFFFF;
            int found_tid  = (g_h_mapped[2] >> 16) & 0xFFFF;
            /* iter global de 128 bits: slots [3]=b0..31 [5]=b32..63 [8]=b64..95 [9]=b96..127 */
            unsigned __int128 found_iter = (unsigned __int128)(unsigned int)g_h_mapped[3]
                                         | ((unsigned __int128)(unsigned int)g_h_mapped[5] << 32)
                                         | ((unsigned __int128)(unsigned int)g_h_mapped[8] << 64)
                                         | ((unsigned __int128)(unsigned int)g_h_mapped[9] << 96);
            int found_micro_off = g_h_mapped[6];
            /* v10.12: kernel grava o ESCALAR GLOBAL (offset dentro do range do worker)
             * direto no registro. Host reconstrói candidate = worker_start + found_iter*salto. */
            unsigned long long found_iter_lo = (unsigned long long)found_iter; /* p/ logs %llu */

            printf("\n\033[93m[DEBUG] GPU MATCH: step=%d worker=%d tid=%d found_iter(lo64)=%llu micro_off=%d\033[0m\n",
                   found_si, found_wi, found_tid, found_iter_lo, found_micro_off);

            if (found_si >= 0 && found_si < num_steps_total &&
                found_wi >= 0 && found_wi < num_workers) {
                
                mpz_t inicio_mpz, salto_mpz, candidate, _ft1, _ft2;
                mpz_inits(inicio_mpz, salto_mpz, candidate, _ft1, _ft2, NULL);
                mpz_set_str(inicio_mpz, g_worker_inicio_hex[found_wi], 16);
                get_step_value(found_si, salto_mpz, _ft1, _ft2);

                char inicio_hex[130], salto_hex[130];
                mpz_get_str(inicio_hex, 16, inicio_mpz);
                mpz_get_str(salto_hex, 16, salto_mpz);
                printf("[DEBUG] worker_start=0x%s step_value=0x%s\n", inicio_hex, salto_hex);

                /* MULTI-ALVO: o índice reportado pela GPU (g_h_mapped[7]) é só uma DICA.
                 * Para não depender do mapeamento índice→alvo (origem de match reportado
                 * no alvo errado), o host recalcula o hash160 do candidato e procura QUAL
                 * alvo realmente bate. Assim a chave achada é sempre verificada/reportada
                 * contra o alvo correto, independente do índice que veio da GPU. */
                int hint_idx = g_h_mapped[7];
                printf("[Multi-Alvo] Match reportado (dica GPU: #%d) — verificando contra todos os alvos...\n", hint_idx);

                int verified = 0;
                /* offset de 128 bits -> mpz (sem truncar). candidate = inicio + salto*(found_iter+delta) */
                mpz_t iter_mpz;
                mpz_init(iter_mpz);
                {
                    uint64_t _lo = (uint64_t)found_iter;
                    uint64_t _hi = (uint64_t)(found_iter >> 64);
                    uint64_t _limbs[2] = { _lo, _hi };
                    mpz_import(iter_mpz, 2, -1 /*LSW first*/, sizeof(uint64_t), 0, 0, _limbs);
                }
                for (int delta = -50; delta <= 50 && !verified; delta++) {
                    mpz_t iter_try; mpz_init(iter_try);
                    mpz_set(iter_try, iter_mpz);
                    if (delta >= 0) mpz_add_ui(iter_try, iter_try, (unsigned long)delta);
                    else            mpz_sub_ui(iter_try, iter_try, (unsigned long)(-delta));
                    if (mpz_sgn(iter_try) < 0) { mpz_clear(iter_try); continue; }

                    mpz_set(candidate, inicio_mpz);
                    mpz_addmul(candidate, salto_mpz, iter_try);   /* candidate += salto * iter_try (precisão total) */
                    if (found_micro_off > 0)
                        mpz_add_ui(candidate, candidate, (unsigned long)found_micro_off);
                    mpz_clear(iter_try);

                    uint8_t sc[32]; mpz_export_32be(candidate, sc);
                    FePt Q; fe_scalar_mul(&Q, sc);
                    if (!fept_is_inf(&Q)) {
                        fe_t Zi, Zi2, xr, yr;
                        fe_inv(Zi, Q.Z); fe_sqr(Zi2, Zi);
                        fe_mul(xr, Q.X, Zi2); fe_mul(yr, Q.Y, Zi2); fe_mul(yr, yr, Zi);
                        uint8_t xb[32]; fe_to_bytes32(xr, xb);
                        uint8_t pfx = fe_is_odd(yr) ? 0x03 : 0x02;
                        uint8_t comp[33]; comp[0]=pfx; memcpy(comp+1,xb,32);
                        uint8_t sha_r[32], rmd_r[20];
                        sha256_33b(comp, sha_r); ripemd160(sha_r, 32, rmd_r);

                        /* Procura QUAL alvo bate com o hash recalculado (dica primeiro,
                         * depois varredura completa) — robusto a índice trocado. */
                        int hit = -1;
                        if (hint_idx >= 0 && hint_idx < g_num_alvos &&
                            memcmp(rmd_r, g_alvos_hash[hint_idx], 20) == 0) {
                            hit = hint_idx;
                        } else {
                            for (int t = 0; t < g_num_alvos; t++) {
                                if (memcmp(rmd_r, g_alvos_hash[t], 20) == 0) { hit = t; break; }
                            }
                        }
                        if (hit >= 0) {
                            int found_alvo_idx = hit;
                            char key_hex[130];
                            mpz_to_hex_zfill(candidate, key_hex, 64);

                            strncpy(final_key_hex, key_hex, 129);

                            pthread_mutex_lock(&found_mutex);
                            FILE *fp = fopen("FOUND_YIPPIE.txt", "a");
                            if(fp){
                                fprintf(fp,"Alvo: %s\n", g_alvos_lista[found_alvo_idx]);
                                fprintf(fp,"Chave: %s (delta=%d)\n\n", key_hex, delta);
                                fclose(fp);
                            }
                            pthread_mutex_unlock(&found_mutex);

                            printf("\033[J\n\n\033[92m[ENCONTRADA] Alvo #%d (%s): %s (delta=%d)\033[0m\n",
                                   found_alvo_idx, g_alvos_lista[found_alvo_idx], key_hex, delta);
                            verified = 1;

                            /* Marca alvo como inativo SÓ no espelho host (para contar
                             * alvos_restantes). NÃO fazemos cudaMemcpy para o device:
                             * a GPU já desativou gd_alvos_ativos[oi] via atomicExch no
                             * momento do match. Um cudaMemcpy aqui BLOQUEARIA o host
                             * esperando o kernel persistente (que não termina) -> congela. */
                            g_alvos_ativos_host[found_alvo_idx] = 0;
                        }
                    }
                }
                if (!verified) {
                    mpz_set(candidate, inicio_mpz);
                    mpz_addmul(candidate, salto_mpz, iter_mpz);
                    if (found_micro_off > 0)
                        mpz_add_ui(candidate, candidate, (unsigned long)found_micro_off);
                    char cand_hex[130];
                    mpz_get_str(cand_hex, 16, candidate);
                    printf("\033[91m[ERRO] GPU encontrou hash match mas verificação FALHOU!\033[0m\n");
                    printf("[ERRO] candidate=0x%s found_iter(lo64)=%llu micro_off=%d delta testado=[-50,+50]\n",
                           cand_hex, found_iter_lo, found_micro_off);
                    printf("[ERRO] GPU_THREADS=%d BATCH_SIZE=%d BLOCKS_PER_SM=%d MICRO_K=%d\n",
                           GPU_THREADS, BATCH_SIZE, BLOCKS_PER_SM, MICRO_K);

                    {
                        uint8_t sc[32]; mpz_export_32be(candidate, sc);
                        FePt Q; fe_scalar_mul(&Q, sc);
                        if (!fept_is_inf(&Q)) {
                            fe_t Zi, Zi2, xr, yr;
                            fe_inv(Zi, Q.Z); fe_sqr(Zi2, Zi);
                            fe_mul(xr, Q.X, Zi2);
                            fe_mul(yr, Q.Y, Zi2); fe_mul(yr, yr, Zi);
                            uint8_t xb[32], yb[32];
                            fe_to_bytes32(xr, xb);
                            fe_to_bytes32(yr, yb);
                            uint8_t pfx = fe_is_odd(yr) ? 0x03 : 0x02;
                            uint8_t comp[33]; comp[0] = pfx; memcpy(comp+1, xb, 32);
                            uint8_t sha_r[32], rmd_r[20];
                            sha256_33b(comp, sha_r);
                            ripemd160(sha_r, 32, rmd_r);

                            /* alvo de referência: usa o hash do host (dica), sem
                             * cudaMemcpyFromSymbol — que bloquearia no kernel persistente. */
                            uint8_t tgt[20] = {0};
                            if (hint_idx >= 0 && hint_idx < g_num_alvos)
                                memcpy(tgt, g_alvos_hash[hint_idx], 20);

                            printf("[DIAG] candidate scalar (32B BE):");
                            for (int i = 0; i < 32; i++) printf(" %02x", sc[i]);
                            printf("\n");
                            printf("[DIAG] Q.x (afim, 32B BE)        :");
                            for (int i = 0; i < 32; i++) printf(" %02x", xb[i]);
                            printf("\n");
                            printf("[DIAG] Q.y (afim, 32B BE)        : (paridade=%s, prefix=0x%02x)",
                                   fe_is_odd(yr) ? "ímpar" : "par", pfx);
                            for (int i = 0; i < 32; i++) printf(" %02x", yb[i]);
                            printf("\n");
                            printf("[DIAG] hash160 calculado (HOST)  :");
                            for (int i = 0; i < 20; i++) printf(" %02x", rmd_r[i]);
                            printf("\n");
                            printf("[DIAG] hash160 target (d_hash160):");
                            for (int i = 0; i < 20; i++) printf(" %02x", tgt[i]);
                            printf("\n");

                            int matching = 0;
                            for (int i = 0; i < 20 && rmd_r[i] == tgt[i]; i++) matching++;
                            printf("[DIAG] bytes batendo (prefixo)   : %d/20\n", matching);
                        } else {
                            printf("[DIAG] Q é ponto no infinito (escalar inválido)\n");
                        }
                    }
                    /* Para evitar loop infinito (kernel relança e acha o mesmo "match"
                     * espúrio), marca o alvo da DICA como inativo no espelho host.
                     * A GPU já desativou gd_alvos_ativos[hint] via atomicExch; NÃO
                     * fazemos cudaMemcpy aqui (bloquearia no kernel persistente). */
                    if (hint_idx >= 0 && hint_idx < g_num_alvos) {
                        printf("\033[91m[Multi-Alvo] Alvo #%d (dica) marcado como FALHO (verificacao falhou) — busca continua.\033[0m\n",
                               hint_idx);
                        g_alvos_ativos_host[hint_idx] = 0;
                    } else {
                        printf("\033[91m[Multi-Alvo] Match espurio sem alvo valido (dica #%d) — ignorado.\033[0m\n",
                               hint_idx);
                    }
                }
                mpz_clears(inicio_mpz, salto_mpz, candidate, _ft1, _ft2, NULL);
                mpz_clear(iter_mpz);
            } else {
                printf("\033[91m[ERRO] Indices invalidos: found_si=%d found_wi=%d\033[0m\n", found_si, found_wi);
            }
            set_evento();
        }   /* fim do bloco de verificação de um registro */
            /* Marca o registro como consumido e avança o índice de leitura. */
            _rec[0] = 0;
            __sync_synchronize();
            g_drained++;
        }   /* fim do while (g_drained < _widx) — drenagem do buffer */

        {
            /* Após drenar o que estava pronto, confere se todos os alvos já foram
             * achados. O kernel SEGUE rodando enquanto houver alvos ativos. */
            int alvos_restantes = 0;
            for (int ai = 0; ai < g_num_alvos; ai++)
                if (g_alvos_ativos_host[ai]) alvos_restantes++;

            if (alvos_restantes == 0) {
                printf("\033[92m[Multi-Alvo] Todos os alvos encontrados — encerrando.\033[0m\n");
                break;
            }
        }

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long long ms = (now.tv_sec - ts_display.tv_sec) * 1000LL
                     + (now.tv_nsec - ts_display.tv_nsec) / 1000000LL;
        
        if (ms >= DISPLAY_MS) {
            ts_display = now;
            __sync_synchronize();

            int cur_offset = g_display_offset;
            if (persistent_blocks > max_display)
                g_display_offset = (g_display_offset + max_display) % persistent_blocks;

            int cur_page    = cur_offset / max_display + 1;
            long long steps_done_gpu = (long long)(*g_h_tasks_done);

            long long total_keys = 0;
            int active_blocks = 0;
            for (int b = 0; b < persistent_blocks; b++) {
                total_keys += g_h_block_status[b].keys_total;
                if (g_h_block_status[b].task_id >= 0) {
                    active_blocks++;
                    total_keys += g_h_block_status[b].loops_done;
                }
            }

            double inst_keys = (double)(total_keys - prev_keys) / (ms / 1000.0);
            prev_keys = total_keys;

            pthread_mutex_lock(&stdout_mutex);

            for (int slot = 0; slot < max_display; slot++) {
                int block_idx  = (cur_offset + slot) % persistent_blocks;
                int offset_linha = slot + 2;
                GPUBlockStatus *bs = &g_h_block_status[block_idx];
                int disp_si = bs->step_idx;

                if (bs->task_id < 0 || disp_si < 0 || disp_si >= num_steps_total) {
                    printf("\033[s\033[%dA\r\033[K\033[90m[Block %03d] (idle)\033[0m\033[u",
                           offset_linha, block_idx);
                    continue;
                }

                int disp_wi = bs->worker_idx;
                if (disp_wi < 0 || disp_wi >= num_workers) disp_wi = 0;

                long long faltam = bs->max_iter - bs->loops_done;
                if (faltam < 0) faltam = 0;

                /* No novo dispatch (v10.11): bloco processa 1 STEP inteiro,
                 * cobrindo workers [0..num_workers-1] do step distribuídos
                 * entre as threads. Mostro o range de workers que o bloco
                 * cobre: começa em block_idx % num_workers e vai em stride
                 * GPU_THREADS. Display mostra o PRIMEIRO worker do bloco
                 * apenas como referência visual. */
                mpz_set_str(_d_pos, g_worker_inicio_hex[disp_wi], 16);
                get_step_value(disp_si, _d_step, _d_tmp1, _d_tmp2);
                mpz_mul_ui(_d_off, _d_step, (unsigned long)bs->loops_done);
                mpz_add(_d_pos, _d_pos, _d_off);

                char tmp_hex[70], hex_display[40];
                mpz_to_hex_zfill(_d_pos, tmp_hex, 64);
                snprintf(hex_display, 39, "%s", tmp_hex + 27);

                char fw[50], fs[100], fmi[40];
                format_commas(faltam, fw);
                mpz_get_str(fs, 10, _d_step);
                /* Passos = nº REAL de iterações deste step = ceil(range_worker/salto),
                 * recomputado em gmp. bs->max_iter vem TRUNCADO em 64 bits (LLONG_MAX
                 * quando o valor real não cabe), então mostrá-lo daria o mesmo número
                 * gigante para steps diferentes. Aqui mostramos o valor exato (que
                 * pode ter centenas de dígitos para steps pequenos sobre ranges grandes). */
                {
                    mpz_t _passos, _rng, _wstart, _wnext;
                    mpz_inits(_passos, _rng, _wstart, _wnext, NULL);
                    mpz_set_str(_wstart, g_worker_inicio_hex[disp_wi], 16);
                    /* range deste worker = inicio do próximo worker - inicio deste.
                     * Para o último worker, usa o mesmo tamanho do penúltimo intervalo. */
                    if (disp_wi + 1 < num_workers) {
                        mpz_set_str(_wnext, g_worker_inicio_hex[disp_wi + 1], 16);
                        mpz_sub(_rng, _wnext, _wstart);
                    } else if (disp_wi > 0) {
                        mpz_t _wprev; mpz_init(_wprev);
                        mpz_set_str(_wprev, g_worker_inicio_hex[disp_wi - 1], 16);
                        mpz_sub(_rng, _wstart, _wprev);
                        mpz_clear(_wprev);
                    } else {
                        mpz_set_ui(_rng, 0);
                    }
                    if (mpz_sgn(_rng) <= 0) mpz_set_ui(_rng, 0);
                    if (mpz_sgn(_d_step) > 0 && mpz_sgn(_rng) > 0)
                        mpz_cdiv_q(_passos, _rng, _d_step);   /* ceil(range_worker / salto) */
                    else
                        mpz_set_ui(_passos, 0);
                    gmp_snprintf(fmi, sizeof(fmi), "%Zd", _passos);
                    mpz_clears(_passos, _rng, _wstart, _wnext, NULL);
                }

                /* Fila = steps que AINDA FALTAM para ESTE bloco (não o contador
                 * global de tasks). disp_si é o step atual; o último step é
                 * num_steps_total-1. Logo faltam (num_steps_total-1 - disp_si)
                 * steps depois do atual. No último step (S=1) isso é 0. */
                long long fila_bloco = (long long)num_steps_total - 1 - (long long)disp_si;
                if (fila_bloco < 0) fila_bloco = 0;

                printf("\033[s\033[%dA\r\033[K\033[92m[Block %03d / Wkr %03d] %37s | Faltam: %30s | Fila: %lld | Step: %s | Passos: %s\033[0m\033[u",
                       offset_linha, block_idx, disp_wi, hex_display, fw, fila_bloco, fs, fmi);
            }

            char bk[40], bt[40], btot[40];
            format_commas(total_keys, bk);
            format_commas(steps_done_gpu, bt);
            format_commas(task_count, btot);
            printf("\r\033[K\033[96m[GPU] %s keys | %.1f Mkey/s | Tasks: %s/%s | "
                   "Ativos: %d/%d workers | Pg %d/%d\033[0m",
                   bk, (inst_keys / 1e6), bt, btot,
                   active_blocks, num_workers, cur_page, total_pages);
            fflush(stdout);

            g_ckpt_tasks_done = steps_done_gpu;
            pthread_mutex_unlock(&stdout_mutex);
        }

        long long ckpt_ms = (now.tv_sec - ts_ckpt.tv_sec) * 1000LL + (now.tv_nsec - ts_ckpt.tv_nsec) / 1000000LL;
        if (ckpt_ms >= CHECKPOINT_INTERVAL_S * 1000LL) {
            ts_ckpt = now;
            __sync_synchronize();
            g_ckpt_tasks_done = (long long)(*g_h_tasks_done);
            save_checkpoint();
        }

        #if defined(__x86_64__)
            for (int p = 0; p < 16; p++) __asm__ volatile("pause":::"memory");
        #else
            sched_yield();
        #endif
    }

    if (qs != cudaSuccess) {
        fprintf(stderr, "\n\033[93m[AVISO] Kernel interrompido (%s) — relançando automaticamente...\033[0m\n",
                cudaGetErrorString(qs));
        cudaGetLastError();                     /* limpa erro do cudaStreamQuery */
        cudaStreamDestroy(g_persistent_stream); /* destrói stream em estado de erro */
        g_persistent_stream = 0;
        cudaStreamCreate(&g_persistent_stream); /* nova stream limpa */
        cudaGetLastError();                     /* limpa qualquer erro residual */
        __sync_synchronize();
        /* v10.14: progresso já está em g_h_worker_next_si — atualiza tasks_done para checkpoint info */
        g_ckpt_tasks_done = (long long)(*g_h_tasks_done);
        save_checkpoint();
        goto relaunch_kernel;
    }

    cudaStreamSynchronize(g_persistent_stream);
    set_evento();

    /* v10.16: kernel terminou normalmente. Se ainda há alvos ativos E algum worker
     * não esgotou seus steps, RELANÇA para continuar varrendo de onde parou
     * (g_h_worker_next_si[bid] guarda o próximo si de cada worker). Só encerra
     * quando todos os alvos foram achados OU todos os workers esgotaram a range. */
    {
        int alvos_restantes = 0;
        for (int ai = 0; ai < g_num_alvos; ai++)
            if (g_alvos_ativos_host[ai]) alvos_restantes++;

        int workers_pendentes = 0;
        for (int b = 0; b < persistent_blocks; b++)
            if (g_h_worker_next_si[b] < num_steps_total) workers_pendentes++;

        if (alvos_restantes > 0 && workers_pendentes > 0) {
            g_ckpt_tasks_done = (long long)(*g_h_tasks_done);
            save_checkpoint();
            goto relaunch_kernel;
        }
    }

    mpz_clears(_d_pos, _d_step, _d_off, _d_tmp1, _d_tmp2, NULL);

    __sync_synchronize();
    long long final_done = (long long)(*g_h_tasks_done);
    long long total_keys_final = 0;
    for (int b = 0; b < persistent_blocks; b++)
        total_keys_final += g_h_block_status[b].keys_total;
    
    char b1[40], b2[40], b3[40];
    format_commas((long long)final_done, b1);
    format_commas((long long)task_count, b2);
    format_commas(total_keys_final, b3);
    printf("\r\033[K[Rainha v10.0] Final: %s/%s steps | %s keys verificadas", b1, b2, b3);
    if (!((volatile int *)g_h_mapped)[0]) {
        printf(" | \033[93mNAO ENCONTRADA\033[0m");
    }
    printf("\n");

    pthread_join(ckpt_thread, NULL);

    if (((volatile int *)g_h_mapped)[0]) {
        printf("\033[92m[SUCESSO] Chave encontrada pela GPU autônoma.\033[0m\n");
        printf("\033[J\n\n\033[92m[PERSISTENT] !!! ENCONTRADA !!!: %s\033[0m\n", final_key_hex);
        
        pthread_mutex_lock(&found_mutex);
        FILE *fp = fopen("FOUND_YIPPIE.txt", "a");
        if(fp) {
            fprintf(fp, "Persistent Final v8: %s\n", final_key_hex);
            fflush(fp);
            fclose(fp);
        }
        pthread_mutex_unlock(&found_mutex);
    } 
    else {
        printf("\033[91m[FIM] Todas as tarefas concluídas. Chave não encontrada.\033[0m\n");
    }
}


/* ── MULTI-ALVO v11.0: helpers para o sorted+prefix table ─────────── */

/* qsort comparator: ordena AlvoEntry por h0 ascendente (ordem dos buckets). */
static int alvo_entry_cmp_h0(const void *a, const void *b) {
    uint32_t ha = ((const AlvoEntry*)a)->h0;
    uint32_t hb = ((const AlvoEntry*)b)->h0;
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}

/* Constrói g_alvos_sorted (ordenado por h0) e g_alvos_prefix
 * (bucket bounds indexado por upper-24 bits de h0).
 *
 * Layout dos uint32 em AlvoEntry: 4 bytes consecutivos do hash160 em
 * little-endian de memória — exatamente o mesmo que o código v10.13 fazia
 * via memcpy(alvos_u32_host[ai], g_alvos_hash[ai], 20). Em x86_64 (LE),
 * AlvoEntry.h0 == h[0] | h[1]<<8 | h[2]<<16 | h[3]<<24, igual ao _tt do
 * kernel (que é rh1+C+Dp como uint32_t nativo LE no NVIDIA).
 *
 * Pré-condição: g_alvos_hash[0..g_num_alvos-1] populado, g_num_alvos > 0.
 * Pós-condição: g_alvos_sorted, g_alvos_prefix alocados e prontos. */
static void build_alvos_prefix_table(void) {
    if (g_alvos_sorted) { free(g_alvos_sorted); g_alvos_sorted = NULL; }
    if (g_alvos_prefix) { free(g_alvos_prefix); g_alvos_prefix = NULL; }

    g_alvos_sorted = (AlvoEntry*)malloc((size_t)g_num_alvos * sizeof(AlvoEntry));
    if (!g_alvos_sorted) {
        fprintf(stderr, "FATAL: malloc g_alvos_sorted (%zu bytes)\n",
                (size_t)g_num_alvos * sizeof(AlvoEntry));
        exit(1);
    }
    for (int i = 0; i < g_num_alvos; i++) {
        const uint8_t *h = g_alvos_hash[i];
        memcpy(&g_alvos_sorted[i].h0, h + 0,  4);
        memcpy(&g_alvos_sorted[i].h1, h + 4,  4);
        memcpy(&g_alvos_sorted[i].h2, h + 8,  4);
        memcpy(&g_alvos_sorted[i].h3, h + 12, 4);
        memcpy(&g_alvos_sorted[i].h4, h + 16, 4);
        g_alvos_sorted[i].orig_idx = (uint32_t)i;
    }
    qsort(g_alvos_sorted, (size_t)g_num_alvos, sizeof(AlvoEntry), alvo_entry_cmp_h0);

    /* Build prefix table: bucket[i] = primeiro índice em g_alvos_sorted onde
     * upper_24(h0) == i. bucket[ALVOS_PREFIX_SIZE] = g_num_alvos (sentinela). */
    size_t pfx_bytes = (size_t)(ALVOS_PREFIX_SIZE + 1) * sizeof(uint32_t);
    g_alvos_prefix = (uint32_t*)malloc(pfx_bytes);
    if (!g_alvos_prefix) {
        fprintf(stderr, "FATAL: malloc g_alvos_prefix (%zu bytes)\n", pfx_bytes);
        exit(1);
    }

    uint32_t cur_pfx = 0;
    g_alvos_prefix[0] = 0;
    for (int i = 0; i < g_num_alvos; i++) {
        uint32_t pfx = g_alvos_sorted[i].h0 >> (32 - ALVOS_PREFIX_BITS);
        while (cur_pfx < pfx) {
            cur_pfx++;
            g_alvos_prefix[cur_pfx] = (uint32_t)i;
        }
    }
    while (cur_pfx < ALVOS_PREFIX_SIZE) {
        cur_pfx++;
        g_alvos_prefix[cur_pfx] = (uint32_t)g_num_alvos;
    }

    /* Stats para o usuário. */
    uint32_t max_bucket = 0;
    uint64_t nonzero_buckets = 0;
    for (uint32_t i = 0; i < ALVOS_PREFIX_SIZE; i++) {
        uint32_t bsz = g_alvos_prefix[i+1] - g_alvos_prefix[i];
        if (bsz > max_bucket) max_bucket = bsz;
        if (bsz > 0) nonzero_buckets++;
    }
    printf("[Multi-Alvo] Prefix table: %u buckets (%u bits), max bucket=%u, "
           "preenchimento=%.4f%%\n",
           ALVOS_PREFIX_SIZE, ALVOS_PREFIX_BITS, max_bucket,
           100.0 * (double)nonzero_buckets / (double)ALVOS_PREFIX_SIZE);
}

/* Aloca arrays device e sobe os dados. Substitui o
 * cudaMemcpyToSymbol(d_alvos_u32, ...) original. */
static void upload_alvos_to_device(void) {
    /* Libera anterior se houver (caso seja chamado mais de uma vez). */
    if (g_d_alvos_sorted_ptr) { cudaFree(g_d_alvos_sorted_ptr); g_d_alvos_sorted_ptr = NULL; }
    if (g_d_alvos_prefix_ptr) { cudaFree(g_d_alvos_prefix_ptr); g_d_alvos_prefix_ptr = NULL; }
    if (g_d_alvos_ativos_ptr) { cudaFree(g_d_alvos_ativos_ptr); g_d_alvos_ativos_ptr = NULL; }

    size_t sorted_bytes = (size_t)g_num_alvos * sizeof(AlvoEntry);
    size_t prefix_bytes = (size_t)(ALVOS_PREFIX_SIZE + 1) * sizeof(uint32_t);
    size_t ativos_bytes = (size_t)g_num_alvos * sizeof(int);

    CUDA_CHECK(cudaMalloc(&g_d_alvos_sorted_ptr, sorted_bytes));
    CUDA_CHECK(cudaMalloc(&g_d_alvos_prefix_ptr, prefix_bytes));
    CUDA_CHECK(cudaMalloc(&g_d_alvos_ativos_ptr, ativos_bytes));

    CUDA_CHECK(cudaMemcpy(g_d_alvos_sorted_ptr, g_alvos_sorted, sorted_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_alvos_prefix_ptr, g_alvos_prefix, prefix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_alvos_ativos_ptr, g_alvos_ativos_host, ativos_bytes, cudaMemcpyHostToDevice));

    /* Sobe os PONTEIROS para os símbolos device. */
    CUDA_CHECK(cudaMemcpyToSymbol(gd_alvos_sorted, &g_d_alvos_sorted_ptr, sizeof(void*)));
    CUDA_CHECK(cudaMemcpyToSymbol(gd_alvos_prefix, &g_d_alvos_prefix_ptr, sizeof(void*)));
    CUDA_CHECK(cudaMemcpyToSymbol(gd_alvos_ativos, &g_d_alvos_ativos_ptr, sizeof(void*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_alvos, &g_num_alvos, sizeof(int)));

    double total_mb = (sorted_bytes + prefix_bytes + ativos_bytes) / (1024.0*1024.0);
    printf("[Multi-Alvo] VRAM uso: sorted=%.1fMB prefix=%.1fMB ativos=%.1fMB total=%.1fMB\n",
           sorted_bytes/(1024.0*1024.0),
           prefix_bytes/(1024.0*1024.0),
           ativos_bytes/(1024.0*1024.0),
           total_mb);
}

int main(void) {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    CUDA_CHECK(cudaSetDevice(0));
    { void *dummy = NULL; cudaMalloc(&dummy, 256); if (dummy) cudaFree(dummy); }

    printf("\033[96m[GPU] %s — %d SMs, %.1f GB VRAM\033[0m\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0*1024.0*1024.0));

    int sm_count = prop.multiProcessorCount;
    int persistent_blocks = sm_count * BLOCKS_PER_SM;
    int NUM_WORKERS = persistent_blocks;  /* 1 bloco = 1 worker */

    printf("[GPU] PERSISTENT KERNEL v10.0 — COOPERATIVE MODEL\n");
    printf("[GPU] GPU_THREADS=%d | BATCH=%d | SMs=%d | BLOCKS_PER_SM=%d\n",
           GPU_THREADS, BATCH_SIZE, sm_count, BLOCKS_PER_SM);
    printf("[GPU] NUM_WORKERS=%d (%d blocos × %d threads/bloco cooperam)\n",
           NUM_WORKERS, persistent_blocks, GPU_THREADS);
    printf("[GPU] Pontos em registradores — ZERO d_X/d_Y/d_Z VRAM\n");

    {
        size_t worker_vram = (size_t)2 * NUM_WORKERS * sizeof(gfe_t);
        size_t fixed_vram = worker_vram + sizeof(gfe_t) + 2 * sizeof(int); /* one_z + found + counter */
        size_t step_budget = VRAM_BUDGET_BYTES - fixed_vram;
        size_t bytes_per_step = (size_t)2 * sizeof(gfe_t) + sizeof(unsigned __int128); /* pow[0] only + max_iter(128b) = 80 bytes FIXO */
        g_target_steps = (int)(step_budget / bytes_per_step);
        printf("[VRAM] Budget FIXO: %.1f GB | Workers VRAM: %.1f MB | Bytes/step: %zu | INIT_NBITS: %d | Steps: %d\n",
               VRAM_BUDGET_BYTES / (1024.0*1024.0*1024.0), worker_vram / (1024.0*1024.0),
               bytes_per_step, INIT_NBITS, g_target_steps);
    }

    /* Allocate dynamic worker arrays */
    g_worker_inicio_hex = (worker_hex_t*)calloc(NUM_WORKERS, sizeof(worker_hex_t));
    g_worker_fim_hex    = (worker_hex_t*)calloc(NUM_WORKERS, sizeof(worker_hex_t));
    g_worker_start_x    = (fe_t_fwd*)calloc(NUM_WORKERS, sizeof(fe_t_fwd));
    g_worker_start_y    = (fe_t_fwd*)calloc(NUM_WORKERS, sizeof(fe_t_fwd));
    printf("[RAM] Worker arrays: %.1f MB para %d workers\n",
           (double)NUM_WORKERS * (2*260 + 2*32) / (1024.0*1024.0), NUM_WORKERS);
    printf("\n");

    CUDA_CHECK(cudaMemcpyToSymbol(d_SHA256_K, _SHA256_K, sizeof(_SHA256_K)));
    cudaFuncSetCacheConfig(kernel_persistent, cudaFuncCachePreferL1);

    init_curve();
    _fe_init_tables();
    bech32_init_map();

#if MICRO_K > 0
    {
        gfe_t h_micro_x[MICRO_K], h_micro_y[MICRO_K];
        for (int j = 0; j < MICRO_K; j++) {
            mpz_t scalar; mpz_init_set_ui(scalar, (unsigned long)(j + 1));
            uint8_t sc[32]; mpz_export_32be(scalar, sc);
            FePt P; fe_scalar_mul(&P, sc);
            fe_t Zi, Zi2, xr, yr;
            fe_inv(Zi, P.Z); fe_sqr(Zi2, Zi);
            fe_mul(xr, P.X, Zi2);
            fe_mul(yr, P.Y, Zi2); fe_mul(yr, yr, Zi);
            memcpy(h_micro_x[j], xr, sizeof(fe_t));
            memcpy(h_micro_y[j], yr, sizeof(fe_t));
            mpz_clear(scalar);
        }
        CUDA_CHECK(cudaMemcpyToSymbol(d_micro_x, h_micro_x, sizeof(h_micro_x)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_micro_y, h_micro_y, sizeof(h_micro_y)));
        printf("[MicroStride] %d affine points (j*G, j=1..%d) carregados.\n",
               MICRO_K, MICRO_K);
    }
#else
    printf("[MicroStride] DESABILITADO (MICRO_K=0) — apenas walk primária.\n");
#endif

    const char *ALVO   = "16zRPnT8znwq42q7XeMkZUhb1bKqgRogyy";
    const char *INICIO = "400000000000000000000000000000000";
    const char *FIM    = "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140";

    /* ══════════════════════════════════════════════════════════════════
     * MULTI-ALVO v11.0: carrega lista de endereços de "alvos.txt" se
     * existir; senão usa ALVO único (compatibilidade retroativa).
     *
     * Formato de alvos.txt: 1 endereço por linha, # para comentários,
     * linhas em branco ignoradas. Todos os endereços devem ser do MESMO
     * tipo (P2PKH/P2SH/P2WPKH/P2TR) — o tipo é detectado do primeiro.
     *
     * SEM hard cap em runtime — limitado por RAM/VRAM disponíveis.
     * Cap de sanidade em MAX_ALVOS_HARD_CAP (100M, ajustável no topo).
     * ══════════════════════════════════════════════════════════════════ */
    g_num_alvos = 0;
    g_alvos_capacity = 0;

    /* Helper: garante g_alvos_capacity >= needed, realocando se preciso. */
    #define GROW_ALVOS(needed) do { \
        if ((needed) > g_alvos_capacity) { \
            int new_cap = g_alvos_capacity ? g_alvos_capacity * 2 : 1024; \
            while (new_cap < (needed)) new_cap *= 2; \
            if ((unsigned)new_cap > MAX_ALVOS_HARD_CAP) { \
                fprintf(stderr, "FATAL: g_num_alvos excede MAX_ALVOS_HARD_CAP=%u\n", \
                        MAX_ALVOS_HARD_CAP); exit(1); \
            } \
            g_alvos_lista        = (char (*)[80])  realloc(g_alvos_lista,        (size_t)new_cap * 80); \
            g_alvos_hash         = (uint8_t (*)[32])realloc(g_alvos_hash,        (size_t)new_cap * 32); \
            g_alvos_ativos_host  = (int *)         realloc(g_alvos_ativos_host,  (size_t)new_cap * sizeof(int)); \
            if (!g_alvos_lista || !g_alvos_hash || !g_alvos_ativos_host) { \
                fprintf(stderr, "FATAL: realloc alvos arrays (cap=%d)\n", new_cap); exit(1); \
            } \
            g_alvos_capacity = new_cap; \
        } \
    } while(0)

    FILE *fa = fopen("legacy_compressed.txt", "r");
    if (fa) {
        char linha[128];
        while (fgets(linha, sizeof(linha), fa)) {
            /* strip whitespace e comentários */
            char *p = linha;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
            char *end = p + strlen(p) - 1;
            while (end > p && (*end == '\n' || *end == '\r' || *end == ' ' || *end == '\t')) {
                *end-- = '\0';
            }
            if (*p == '\0') continue;
            GROW_ALVOS(g_num_alvos + 1);
            strncpy(g_alvos_lista[g_num_alvos], p, 79);
            g_alvos_lista[g_num_alvos][79] = '\0';
            /* Print só os primeiros e os últimos para evitar inundar o terminal */
            if (g_num_alvos < 5)
                printf("[Alvo Carregado #%d]: %s\n", g_num_alvos, g_alvos_lista[g_num_alvos]);
            else if (g_num_alvos == 5)
                printf("[Alvo Carregado] ... (suprimindo log para grandes listas) ...\n");
            g_num_alvos++;
        }
        fclose(fa);
        if (g_num_alvos == 0) {
            printf("[Multi-Alvo] alvos.txt vazio — usando ALVO embutido.\n");
        } else {
            printf("[Multi-Alvo] Carregados %d endereços de alvos.txt\n", g_num_alvos);
        }
    }
    if (g_num_alvos == 0) {
        /* Fallback: usa ALVO embutido como único alvo */
        GROW_ALVOS(1);
        strncpy(g_alvos_lista[0], ALVO, 79);
        g_alvos_lista[0][79] = '\0';
        g_num_alvos = 1;
        printf("[Multi-Alvo] Modo single-alvo: %s\n", ALVO);
    }
    #undef GROW_ALVOS

    /* Detecta tipo de endereço do PRIMEIRO. Todos devem ser o mesmo tipo. */
    g_addr_type = detectar_tipo_endereco(g_alvos_lista[0]);
    g_gpu_mode = modo_gpu_do_tipo(g_addr_type);
    const char *tipo_str[] = {"P2PKH (Legacy)", "P2PKH (Uncomp)", "P2SH (Nested SegWit)",
                              "P2WPKH (Native SegWit)", "P2TR (Taproot)"};
    printf("[Endereço] Tipo detectado: %s\n", tipo_str[g_addr_type]);
    printf("[Endereço] Modo GPU: %d\n", g_gpu_mode);

    /* Extrai hash160 de cada alvo + marca todos ativos */
    {
        for (int ai = 0; ai < g_num_alvos; ai++) {
            int tipo_ai = detectar_tipo_endereco(g_alvos_lista[ai]);
            if (tipo_ai != g_addr_type) {
                fprintf(stderr, "FATAL: alvo %d (%s) tem tipo diferente do primeiro alvo.\n"
                                "Todos os alvos em alvos.txt devem ser do mesmo tipo.\n",
                        ai, g_alvos_lista[ai]);
                return 1;
            }
            int len = extrair_hash_do_endereco(g_alvos_lista[ai], g_alvos_hash[ai], g_addr_type);
            if (len < 0) {
                fprintf(stderr, "FATAL: não foi possível extrair hash do endereço alvo %d (%s)\n",
                        ai, g_alvos_lista[ai]);
                return 1;
            }
            g_alvos_ativos_host[ai] = 1;
        }

        /* Constrói prefix table host-side, sobe arrays para device. */
        struct timespec _t0, _t1;
        clock_gettime(CLOCK_MONOTONIC, &_t0);
        build_alvos_prefix_table();
        upload_alvos_to_device();
        clock_gettime(CLOCK_MONOTONIC, &_t1);
        double _ms = (_t1.tv_sec-_t0.tv_sec)*1000.0 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
        printf("[Multi-Alvo] Sort + prefix build + upload em %.0f ms para %d alvos.\n",
               _ms, g_num_alvos);

        /* Setup legado single-alvo: usa primeiro alvo */
        uint32_t alvo0_u32[5]; memset(alvo0_u32, 0, sizeof(alvo0_u32));
        memcpy(alvo0_u32, g_alvos_hash[0], 20);
        CUDA_CHECK(cudaMemcpyToSymbol(d_hash160_alvo, g_alvos_hash[0], 20));
        CUDA_CHECK(cudaMemcpyToSymbol(d_alvo_u32, alvo0_u32, sizeof(alvo0_u32)));
        if (g_addr_type == 4) /* P2TR */
            CUDA_CHECK(cudaMemcpyToSymbol(d_target_32, g_alvos_hash[0], 32));
        CUDA_CHECK(cudaMemcpyToSymbol(d_gpu_mode, &g_gpu_mode, sizeof(int)));

        printf("[Multi-Alvo] %d alvos enviados para GPU. Primeiro hash: ", g_num_alvos);
        for (int i = 0; i < 20; i++) printf("%02x", g_alvos_hash[0][i]);
        printf("\n\n");
    }

    {
        mpz_t range_check, ini_c, fim_c;
        mpz_inits(range_check, ini_c, fim_c, NULL);
        mpz_set_str(ini_c, INICIO, 16);
        mpz_set_str(fim_c, FIM, 16);
        mpz_sub(range_check, fim_c, ini_c);
        int range_bits = (int)mpz_sizeinbase(range_check, 2);
        mpz_clears(ini_c, fim_c, range_check, NULL);
        int loaded = 0;
        if (range_bits <= 64) loaded = carregar_steps_arquivo("dados.txt");
        if (!loaded) {
            printf("[StepGen] Gerando steps logarítmicos dinamicamente...\n");
            gerar_steps_logaritmicos(INICIO, FIM, NUM_WORKERS, g_target_steps);
        }
    }
    int num_steps = g_num_steps_hex;
    printf("[Steps] Total: %d steps\n", num_steps);

    mpz_t inicio_mpz_g, fim_mpz_g, range_mpz_g;
    mpz_init(inicio_mpz_g); mpz_set_str(inicio_mpz_g, INICIO, 16);
    mpz_init(fim_mpz_g);    mpz_set_str(fim_mpz_g, FIM, 16);
    mpz_init(range_mpz_g);  mpz_sub(range_mpz_g, fim_mpz_g, inicio_mpz_g);

    {
        struct timespec tp0, tp1;
        clock_gettime(CLOCK_MONOTONIC, &tp0);

        for (int wi = 0; wi < NUM_WORKERS; wi++) {
            mpz_t blk_s, blk_e, tmp_w;
            mpz_inits(blk_s, blk_e, tmp_w, NULL);
            mpz_mul_ui(tmp_w, range_mpz_g, (unsigned long)wi);
            mpz_tdiv_q_ui(tmp_w, tmp_w, (unsigned long)NUM_WORKERS);
            mpz_add(blk_s, inicio_mpz_g, tmp_w);
            mpz_mul_ui(tmp_w, range_mpz_g, (unsigned long)(wi+1));
            mpz_tdiv_q_ui(tmp_w, tmp_w, (unsigned long)NUM_WORKERS);
            mpz_add(blk_e, inicio_mpz_g, tmp_w);
            if (wi == NUM_WORKERS-1) mpz_set(blk_e, fim_mpz_g);

            mpz_get_str(g_worker_inicio_hex[wi], 16, blk_s);
            mpz_get_str(g_worker_fim_hex[wi], 16, blk_e);

            uint8_t sc[32]; mpz_export_32be(blk_s, sc);
            FePt A; fe_scalar_mul(&A, sc);
            if (!fept_is_inf(&A)) {
                fe_t Zi, Zi2;
                fe_inv(Zi, A.Z); fe_sqr(Zi2, Zi);
                fe_mul(g_worker_start_x[wi], A.X, Zi2);
                fe_mul(g_worker_start_y[wi], A.Y, Zi2);
                fe_mul(g_worker_start_y[wi], g_worker_start_y[wi], Zi);
            } else { fe_zero(g_worker_start_x[wi]); fe_zero(g_worker_start_y[wi]); }

            mpz_clears(blk_s, blk_e, tmp_w, NULL);
        }
        printf("[Precomp] %d starting points computados.\n", NUM_WORKERS);

#if STEP_ONLY_EXHAUSTIVE
        /* Modo exaustivo: só os últimos SEQ_STEPS_COUNT steps (sequenciais) são
         * processados, então o total de tasks real é SEQ_STEPS_COUNT × workers. */
        {
            int eff_steps = SEQ_STEPS_COUNT;
            if (eff_steps > num_steps) eff_steps = num_steps;
            task_count = (long long)eff_steps * (long long)NUM_WORKERS;
        }
        printf("[Tasks] %lld tasks (%d steps sequenciais × %d workers) — MODO EXAUSTIVO\n",
               task_count, (SEQ_STEPS_COUNT > num_steps ? num_steps : SEQ_STEPS_COUNT), NUM_WORKERS);
#else
        task_count = (long long)num_steps * (long long)NUM_WORKERS;
        printf("[Tasks] %lld tasks (%d steps × %d workers) — paralelismo uniforme task=(si,wi)\n",
               task_count, num_steps, NUM_WORKERS);
#endif
        printf("[Tasks] Steps: maior→menor monotonicamente | Fallback S=1 com cobertura 100%%\n");

        clock_gettime(CLOCK_MONOTONIC, &tp1);
        printf("[Precomp] Starting points em %.0f ms.\n",
               (tp1.tv_sec-tp0.tv_sec)*1000.0+(tp1.tv_nsec-tp0.tv_nsec)/1e6);
    }

    {
        #define PRECOMP_CHUNK 150000
        struct timespec tp0, tp1;
        clock_gettime(CLOCK_MONOTONIC, &tp0);

        size_t pow_size = (size_t)num_steps * 1 * sizeof(gfe_t);
        CUDA_CHECK(cudaMalloc(&d_gpu_pow_x, pow_size));
        CUDA_CHECK(cudaMalloc(&d_gpu_pow_y, pow_size));
        CUDA_CHECK(cudaMalloc(&d_gpu_max_iter, (size_t)num_steps * sizeof(unsigned __int128)));

        printf("[Precomp] Processando %d steps em chunks de %d (RAM mínima — on-demand)...\n", num_steps, PRECOMP_CHUNK);
        int ncpu = 54;
        if (ncpu > num_steps) ncpu = 1;

        mpz_t worker_max_range;
        mpz_init(worker_max_range);
        mpz_cdiv_q_ui(worker_max_range, range_mpz_g, (unsigned long)NUM_WORKERS);


        for (int chunk_start = 0; chunk_start < num_steps; chunk_start += PRECOMP_CHUNK) {
            int chunk_end = chunk_start + PRECOMP_CHUNK;
            if (chunk_end > num_steps) chunk_end = num_steps;
            int chunk_size = chunk_end - chunk_start;

            g_step_data = (StepData*)calloc(chunk_size, sizeof(StepData));

            int nt = ncpu;
            if (nt > chunk_size) nt = chunk_size;
            pthread_t *thr = (pthread_t*)malloc(nt * sizeof(pthread_t));
            PrecompStepArg *args = (PrecompStepArg*)malloc(nt * sizeof(PrecompStepArg));

            g_num_steps_pre = chunk_end;

            for (int i = 0; i < nt; i++) {
                args[i] = (PrecompStepArg){chunk_start + i, nt, chunk_end, chunk_start};
                pthread_create(&thr[i], NULL, precomp_step_thread, &args[i]);
            }
            for (int i = 0; i < nt; i++) pthread_join(thr[i], NULL);
            free(thr); free(args);

            size_t chunk_pow_size = (size_t)chunk_size * sizeof(gfe_t);
            gfe_t *h_px = (gfe_t*)malloc(chunk_pow_size);
            gfe_t *h_py = (gfe_t*)malloc(chunk_pow_size);
            for (int si = 0; si < chunk_size; si++) {
                memcpy(&h_px[si], &g_step_data[si].pow_x[0], sizeof(gfe_t));
                memcpy(&h_py[si], &g_step_data[si].pow_y[0], sizeof(gfe_t));
            }
            size_t offset = (size_t)chunk_start * sizeof(gfe_t);
            CUDA_CHECK(cudaMemcpy((char*)d_gpu_pow_x + offset, h_px, chunk_pow_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy((char*)d_gpu_pow_y + offset, h_py, chunk_pow_size, cudaMemcpyHostToDevice));
            free(h_px); free(h_py);
            free(g_step_data);
            g_step_data = NULL;

            {
                unsigned __int128 *h_mi = (unsigned __int128*)malloc((size_t)chunk_size * sizeof(unsigned __int128));
                mpz_t salto_tmp, mi_tmp, _mt1, _mt2, max128;
                mpz_inits(salto_tmp, mi_tmp, _mt1, _mt2, max128, NULL);
                /* max128 = 2^128 - 1 (teto do contador de 128 bits do kernel) */
                mpz_ui_pow_ui(max128, 2, 128);
                mpz_sub_ui(max128, max128, 1);
                for (int si = 0; si < chunk_size; si++) {
                    get_step_value(chunk_start + si, salto_tmp, _mt1, _mt2);
                    /* nº de iterações = ceil(range_do_worker / salto): cobre o range
                     * inteiro do worker, SEM limite artificial de quantidade. O contador
                     * do kernel agora é 128 bits, então valores até 2^128-1 são EXATOS.
                     * Só se o valor real exceder 2^128-1 (salto minúsculo sobre range
                     * astronomicamente grande) ele é fixado no teto de 128 bits — limite
                     * físico do contador, não uma escolha. */
                    mpz_cdiv_q(mi_tmp, worker_max_range, salto_tmp);  /* CEILING: cobre toda a range do worker */
                    if (mpz_cmp(mi_tmp, max128) > 0) mpz_set(mi_tmp, max128);
                    if (mpz_sgn(mi_tmp) <= 0) mpz_set_ui(mi_tmp, 1);
                    /* mpz -> unsigned __int128 via export (8 bytes baixos + 8 altos, little-endian) */
                    unsigned __int128 v = 0;
                    {
                        size_t cnt = 0;
                        uint64_t limbs[2] = {0, 0};
                        mpz_export(limbs, &cnt, -1 /*LSW first*/, sizeof(uint64_t), 0 /*host endian*/, 0, mi_tmp);
                        v = (unsigned __int128)limbs[0] | ((unsigned __int128)limbs[1] << 64);
                    }
                    h_mi[si] = v;
                }
                mpz_clears(salto_tmp, mi_tmp, _mt1, _mt2, max128, NULL);
                size_t mi_offset = (size_t)chunk_start * sizeof(unsigned __int128);
                CUDA_CHECK(cudaMemcpy((char*)d_gpu_max_iter + mi_offset, h_mi,
                    (size_t)chunk_size * sizeof(unsigned __int128), cudaMemcpyHostToDevice));
                free(h_mi);
            }

            if ((chunk_start / PRECOMP_CHUNK) % 10 == 0 || chunk_end == num_steps) {
                printf("[Precomp] %d/%d steps (%.0f%%)\n", chunk_end, num_steps,
                       100.0 * chunk_end / num_steps);
            }
        }
        mpz_clear(worker_max_range);

        CUDA_CHECK(cudaMalloc(&d_gpu_starts_x, NUM_WORKERS * sizeof(gfe_t)));
        CUDA_CHECK(cudaMalloc(&d_gpu_starts_y, NUM_WORKERS * sizeof(gfe_t)));
        CUDA_CHECK(cudaMemcpy(d_gpu_starts_x, g_worker_start_x, NUM_WORKERS*sizeof(gfe_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gpu_starts_y, g_worker_start_y, NUM_WORKERS*sizeof(gfe_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_gpu_one_z, sizeof(gfe_t)));
        gfe_t oz; oz[0]=1; oz[1]=oz[2]=oz[3]=0;
        CUDA_CHECK(cudaMemcpy(d_gpu_one_z, oz, sizeof(gfe_t), cudaMemcpyHostToDevice));

        free(g_worker_fim_hex);    g_worker_fim_hex = NULL;
        free(g_worker_start_x);    g_worker_start_x = NULL;
        free(g_worker_start_y);    g_worker_start_y = NULL;
        if (g_steps_hex) { free(g_steps_hex); g_steps_hex = NULL; } /* caso arquivo — dinâmico já é NULL */

        size_t total_vram = pow_size * 2 + (size_t)num_steps * sizeof(unsigned __int128)  /* pow_x + pow_y + max_iter(128b) */
                          + (size_t)NUM_WORKERS * 2 * sizeof(gfe_t) + sizeof(gfe_t);
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        printf("[Precomp+Upload] %d steps (%.1f GB VRAM) em %.0f s.\n", num_steps,
               total_vram / (1024.0*1024*1024),
               (tp1.tv_sec-tp0.tv_sec) + (tp1.tv_nsec-tp0.tv_nsec)/1e9);
    }

    num_workers_global = NUM_WORKERS;
    num_steps_total = num_steps;

    struct timespec ta, tb;
    clock_gettime(CLOCK_MONOTONIC, &ta);
    gpu_persistent_alloc(task_count, persistent_blocks);
    clock_gettime(CLOCK_MONOTONIC, &tb);
    double alloc_ms = ((tb.tv_sec-ta.tv_sec)*1000.0 + (tb.tv_nsec-ta.tv_nsec)/1e6);
    printf("[Persistent Alloc] Tempo: %.0f ms\n\n", alloc_ms);

    /* v10.14: load_checkpoint após alloc para popular g_h_worker_next_si */
    if (load_checkpoint()) {
        printf("[Checkpoint] %lld tarefas puladas.\n", g_ckpt_tasks_done);
    } else {
        printf("[Checkpoint] Nenhum checkpoint — iniciando do zero.\n");
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rainha_dos_processos(NUM_WORKERS);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("Tempo total de execução: %.2f segundos.\n", elapsed);

    gpu_persistent_free();
    gpu_free_data();
    mpz_clears(P_val, G_x_val, G_y_val, P_minus_2, NULL);
    mpz_clears(inicio_mpz_g, fim_mpz_g, range_mpz_g, NULL);
    if (g_step_data) free(g_step_data);
    if (g_steps_hex) free(g_steps_hex);
    if (g_worker_inicio_hex) free(g_worker_inicio_hex);
    if (g_worker_fim_hex) free(g_worker_fim_hex);
    if (g_worker_start_x) free(g_worker_start_x);
    if (g_worker_start_y) free(g_worker_start_y);
    return 0;
}
