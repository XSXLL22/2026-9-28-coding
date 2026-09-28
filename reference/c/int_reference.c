/* Independent C integer reference model for P4PKB01 packs (P4.4).
 *
 * Contract: p4-contract-1.2 (docs/定点格式说明.md). This file must reproduce
 * reference/python/int_reference.py bit-exactly on every golden vector in tests/golden.
 * Implemented for gcc on little-endian hosts; the right-shift of signed values is the
 * arithmetic (floor) shift, which the negative rounding vectors are designed to catch.
 *
 * Build: gcc -std=c11 -O2 -Wall -Wextra -o int_reference int_reference.c
 * Usage: ./int_reference --pack <model_pack.bin> --input <input.bin> --output-dir <dir>
 *
 * Writes every arithmetic node output (conv/requantize/add/maxpool) as raw int8 bytes
 * named <node>.bin, matching the frozen expected files in tests/golden/<vector>/expected.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS 8
#define MAX_NAME 64

static void die(const char *msg) {
    fprintf(stderr, "int_reference: %s\n", msg);
    exit(1);
}

static uint8_t *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "int_reference: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)size);
    if (!data) die("out of memory");
    if (fread(data, 1, (size_t)size, f) != (size_t)size) die("short read");
    fclose(f);
    *len = (size_t)size;
    return data;
}

static uint32_t ru32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t ri32(const uint8_t *p) { return (int32_t)ru32(p); }

typedef struct {
    char name[MAX_NAME];
    int op, act, cls;
    int32_t cin, cout, kh, kw, stride;
    int32_t pt, pb, pl, pr;
    int n_in;
    int32_t inputs[MAX_INPUTS];
    int32_t out, out2;
    uint8_t *param;
    uint32_t param_len;
    /* decoded conv parameters */
    int8_t *qw;
    int32_t *qb;
    int64_t *M;
    int64_t *shift;
    int8_t *lut;
    int64_t M_scalar, shift_scalar;
} Node;

typedef struct {
    int n_nodes;
    Node *nodes;
    int n_tensors;
    int32_t *shapes;   /* 3 per tensor */
    char *meta;
} Pack;

static int8_t sat_i8(int64_t v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static Pack load_pack(const char *path) {
    size_t len;
    uint8_t *data = read_all(path, &len);
    if (len < 12 || memcmp(data, "P4PKB01\n", 8) != 0) die("bad pack magic");
    Pack pack;
    memset(&pack, 0, sizeof pack);
    size_t pos = 8;
    pack.n_nodes = (int)ru32(data + pos); pos += 4;
    pack.nodes = calloc((size_t)pack.n_nodes, sizeof(Node));
    if (!pack.nodes) die("out of memory");
    for (int i = 0; i < pack.n_nodes; i++) {
        Node *n = &pack.nodes[i];
        n->op = data[pos]; n->act = data[pos + 1]; n->cls = data[pos + 2]; pos += 4;
        n->cin = (int32_t)ru32(data + pos); pos += 4;
        n->cout = (int32_t)ru32(data + pos); pos += 4;
        n->kh = (int32_t)ru32(data + pos); pos += 4;
        n->kw = (int32_t)ru32(data + pos); pos += 4;
        n->stride = (int32_t)ru32(data + pos); pos += 4;
        n->pt = ri32(data + pos); pos += 4;
        n->pb = ri32(data + pos); pos += 4;
        n->pl = ri32(data + pos); pos += 4;
        n->pr = ri32(data + pos); pos += 4;
        n->n_in = (int)ru32(data + pos); pos += 4;
        if (n->n_in < 0 || n->n_in > MAX_INPUTS) die("too many inputs");
        for (int j = 0; j < n->n_in; j++) { n->inputs[j] = ri32(data + pos); pos += 4; }
        n->out = ri32(data + pos); pos += 4;
        n->out2 = ri32(data + pos); pos += 4;
        uint32_t off = ru32(data + pos); pos += 4;
        n->param_len = ru32(data + pos); pos += 4;
        uint32_t name_len = ru32(data + pos); pos += 4;
        if (name_len >= MAX_NAME) die("node name too long");
        memcpy(n->name, data + pos, name_len);
        n->name[name_len] = 0;
        pos += name_len;
        if (n->param_len) {
            if (off + n->param_len > len) die("param out of range");
            n->param = data + 0; /* body offset resolved after body position known */
            n->param = NULL;
            n->param_len = n->param_len;
            /* stash offset/len in unused fields temporarily via param pointer arithmetic below */
            n->qw = NULL;
            n->param = malloc(n->param_len);
            if (!n->param) die("out of memory");
            /* body starts later in the file; we re-read after the sweep below */
            /* store offset in M_scalar temporarily */
            n->M_scalar = (int64_t)off;
            n->shift_scalar = (int64_t)n->param_len;
        }
    }
    pack.n_tensors = (int)ru32(data + pos); pos += 4;
    pack.shapes = malloc(sizeof(int32_t) * 3 * (size_t)pack.n_tensors);
    if (!pack.shapes) die("out of memory");
    for (int t = 0; t < pack.n_tensors; t++) {
        pack.shapes[3 * t] = (int32_t)ru32(data + pos);
        pack.shapes[3 * t + 1] = (int32_t)ru32(data + pos + 4);
        pack.shapes[3 * t + 2] = (int32_t)ru32(data + pos + 8);
        pos += 12;
    }
    uint32_t meta_len = ru32(data + pos); pos += 4;
    pack.meta = malloc(meta_len + 1);
    if (!pack.meta) die("out of memory");
    memcpy(pack.meta, data + pos, meta_len);
    pack.meta[meta_len] = 0;
    pos += meta_len;
    uint32_t body_len = ru32(data + pos); pos += 4;
    uint8_t *body = data + pos;
    if (pos + body_len != len) die("pack size mismatch");
    /* decode parameters now that body is located */
    for (int i = 0; i < pack.n_nodes; i++) {
        Node *n = &pack.nodes[i];
        if (!n->param) continue;
        size_t off = (size_t)n->M_scalar;
        memcpy(n->param, body + off, n->param_len);
        size_t cur = 0;
        if (n->op == 0 || n->op == 1 || n->op == 2 || n->op == 3) { /* conv */
            size_t n_w = (size_t)n->cout * n->cin * n->kh * n->kw;
            n->qw = malloc(n_w);
            n->qb = malloc(sizeof(int32_t) * (size_t)n->cout);
            n->M = malloc(sizeof(int64_t) * (size_t)n->cout);
            n->shift = malloc(sizeof(int64_t) * (size_t)n->cout);
            if (!n->qw || !n->qb || !n->M || !n->shift) die("out of memory");
            memcpy(n->qw, n->param + cur, n_w); cur += n_w;
            memcpy(n->qb, n->param + cur, sizeof(int32_t) * (size_t)n->cout); cur += 4u * (size_t)n->cout;
            /* int32 bit patterns must be CONVERTED to int64 element-wise; a memcpy would
               pack four int32 patterns into the first int64 element and read past the end. */
            for (int o = 0; o < n->cout; o++) {
                n->M[o] = (int64_t)ri32(n->param + cur + 4u * (size_t)o);
                n->shift[o] = (int64_t)ri32(n->param + cur + 4u * ((size_t)n->cout + (size_t)o));
            }
            cur += 8u * (size_t)n->cout;
            /* int32 bit patterns are already the two's-complement values we need */
            if (n->act == 2) { /* silu lut: 256 bytes */
                n->lut = malloc(256);
                if (!n->lut) die("out of memory");
                memcpy(n->lut, n->param + cur, 256);
            }
        } else if (n->op == 10) { /* requantize: two i32 (M, shift) */
            n->M_scalar = (int64_t)ri32(n->param);
            n->shift_scalar = (int64_t)ri32(n->param + 4);
        }
    }
    return pack;
}

static int64_t *run_conv(const Node *n, const int8_t *x, int h_in, int w_in) {
    int h_out = (h_in + n->pt + n->pb - n->kh) / n->stride + 1;
    int w_out = (w_in + n->pl + n->pr - n->kw) / n->stride + 1;
    int64_t *acc = malloc(sizeof(int64_t) * (size_t)n->cout * (size_t)h_out * (size_t)w_out);
    if (!acc) die("out of memory");
    memset(acc, 0, sizeof(int64_t) * (size_t)n->cout * (size_t)h_out * (size_t)w_out);
    for (int o = 0; o < n->cout; o++) {
        for (int i = 0; i < n->kh; i++) {
            for (int j = 0; j < n->kw; j++) {
                const int8_t *w = n->qw + (((size_t)o * n->cin + 0) * n->kh + i) * n->kw + j;
                for (int c = 0; c < n->cin; c++) {
                    const int8_t wc = w[(size_t)c * n->kh * n->kw];
                    for (int oh = 0; oh < h_out; oh++) {
                        int ih = oh * n->stride - n->pt + i;
                        if (ih < 0 || ih >= h_in) continue;
                        for (int ow = 0; ow < w_out; ow++) {
                            int iw = ow * n->stride - n->pl + j;
                            if (iw < 0 || iw >= w_in) continue;
                            acc[(((size_t)o * h_out) + oh) * w_out + ow] +=
                                (int64_t)x[((size_t)c * h_in * w_in) + ih * w_in + iw] * (int64_t)wc;
                        }
                    }
                }
            }
        }
        for (int k = 0; k < h_out * w_out; k++) {
            size_t idx = (size_t)o * h_out * w_out + k;
            acc[idx] += n->qb[o];
        }
    }
    return acc;
}

static void requant_per_channel(const int64_t *acc, const Node *n, int plane, int8_t *out) {
    for (int o = 0; o < n->cout; o++) {
        int sh = (int)n->shift[o];
        int64_t half = sh > 0 ? ((int64_t)1 << (sh - 1)) : 0;
        for (int k = 0; k < plane; k++) {
            int64_t t = acc[(size_t)o * plane + k] * n->M[o] + half;
            out[(size_t)o * plane + k] = sat_i8(t >> sh);
        }
    }
}

int main(int argc, char **argv) {
    const char *pack_path = NULL, *input_path = NULL, *out_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack_path = argv[++i];
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) input_path = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i + 1 < argc) out_dir = argv[++i];
        else die("usage: int_reference --pack <bin> --input <input.bin> --output-dir <dir>");
    }
    if (!pack_path || !input_path || !out_dir) die("missing arguments");
    Pack pack = load_pack(pack_path);

    int64_t in_elems = (int64_t)pack.shapes[0] * pack.shapes[1] * pack.shapes[2];
    size_t in_len;
    uint8_t *input = read_all(input_path, &in_len);
    if ((int64_t)in_len != in_elems) die("input size mismatch");

    int n_buf = pack.n_tensors;
    int8_t **buf = calloc((size_t)n_buf, sizeof(int8_t *));
    if (!buf) die("out of memory");
    buf[0] = malloc((size_t)in_elems);
    if (!buf[0]) die("out of memory");
    memcpy(buf[0], input, (size_t)in_elems);

    int written = 0;
    for (int i = 0; i < pack.n_nodes; i++) {
        Node *n = &pack.nodes[i];
        const int8_t *a = buf[n->inputs[0]];
        int32_t c = pack.shapes[3 * n->inputs[0]];
        int32_t h = pack.shapes[3 * n->inputs[0] + 1];
        int32_t w = pack.shapes[3 * n->inputs[0] + 2];
        int8_t *y = NULL;
        if (n->op == 0 || n->op == 1 || n->op == 2 || n->op == 3) { /* conv */
            int h_out = (h + n->pt + n->pb - n->kh) / n->stride + 1;
            int w_out = (w + n->pl + n->pr - n->kw) / n->stride + 1;
            int64_t *acc = run_conv(n, a, h, w);
            y = malloc((size_t)n->cout * h_out * w_out);
            if (!y) die("out of memory");
            requant_per_channel(acc, n, h_out * w_out, y);
            if (n->act == 2) { /* silu lut */
                for (size_t k = 0; k < (size_t)n->cout * h_out * w_out; k++)
                    y[k] = n->lut[(int)y[k] + 128];
            } else if (n->act == 1) { /* relu */
                for (size_t k = 0; k < (size_t)n->cout * h_out * w_out; k++)
                    if (y[k] < 0) y[k] = 0;
            }
            free(acc);
        } else if (n->op == 10) { /* requantize */
            int64_t elems = (int64_t)c * h * w;
            y = malloc((size_t)elems);
            if (!y) die("out of memory");
            int64_t M = n->M_scalar, shift = n->shift_scalar;
            int64_t half = (int64_t)1 << (shift - 1);
            for (int64_t k = 0; k < elems; k++)
                y[k] = sat_i8((((int64_t)a[k]) * M + half) >> shift);
        } else if (n->op == 4) { /* maxpool 5x5: -inf border semantics */
            int h_out = (h + n->pt + n->pb - n->kh) / n->stride + 1;
            int w_out = (w + n->pl + n->pr - n->kw) / n->stride + 1;
            y = malloc((size_t)c * h_out * w_out);
            if (!y) die("out of memory");
            for (int ch = 0; ch < c; ch++)
                for (int oh = 0; oh < h_out; oh++)
                    for (int ow = 0; ow < w_out; ow++) {
                        int m = -128;
                        for (int i2 = 0; i2 < n->kh; i2++)
                            for (int j2 = 0; j2 < n->kw; j2++) {
                                int ih = oh * n->stride - n->pt + i2;
                                int iw = ow * n->stride - n->pl + j2;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue; /* -inf */
                                int v = a[((size_t)ch * h * w) + ih * w + iw];
                                if (v > m) m = v;
                            }
                        y[((size_t)ch * h_out * w_out) + oh * w_out + ow] = (int8_t)m;
                    }
            if (getenv("INT_REF_DEBUG")) {
                fprintf(stderr, "maxpool %s: in_tid=%d c=%d h=%d w=%d kh=%d stride=%d pt=%d | ch0=", n->name,
                        n->inputs[0], c, h, w, n->kh, n->stride, n->pt);
                for (int k = 0; k < h * w; k++) fprintf(stderr, "%d ", a[k]);
                fprintf(stderr, "| y=");
                for (int k = 0; k < h_out * w_out; k++) fprintf(stderr, "%d ", y[k]);
                fprintf(stderr, "\n");
            }
        } else if (n->op == 5) { /* upsample nearest 2x */
            int h_out = h * 2, w_out = w * 2;
            y = malloc((size_t)c * h_out * w_out);
            if (!y) die("out of memory");
            for (int ch = 0; ch < c; ch++)
                for (int oh = 0; oh < h_out; oh++)
                    for (int ow = 0; ow < w_out; ow++)
                        y[((size_t)ch * h_out * w_out) + oh * w_out + ow] =
                            a[((size_t)ch * h * w) + (oh / 2) * w + (ow / 2)];
        } else if (n->op == 6) { /* concat */
            int64_t total = 0;
            for (int j = 0; j < n->n_in; j++)
                total += (int64_t)pack.shapes[3 * n->inputs[j]] * pack.shapes[3 * n->inputs[j] + 1] * pack.shapes[3 * n->inputs[j] + 2];
            y = malloc((size_t)total);
            if (!y) die("out of memory");
            int64_t cur = 0;
            for (int j = 0; j < n->n_in; j++) {
                int32_t tc = pack.shapes[3 * n->inputs[j]];
                int64_t elems = (int64_t)tc * pack.shapes[3 * n->inputs[j] + 1] * pack.shapes[3 * n->inputs[j] + 2];
                memcpy(y + cur, buf[n->inputs[j]], (size_t)elems);
                cur += elems;
            }
        } else if (n->op == 7) { /* split_chunk */
            int64_t elems = (int64_t)c * h * w;
            int64_t half = elems / 2;
            y = malloc((size_t)half);
            if (!y) die("out of memory");
            memcpy(y, a, (size_t)half);
            int8_t *second = malloc((size_t)(elems - half));
            if (!second) die("out of memory");
            memcpy(second, a + half, (size_t)(elems - half));
            buf[n->out2] = second;
        } else if (n->op == 8) { /* add with int16 widening and int8 saturation */
            int64_t elems = (int64_t)c * h * w;
            y = malloc((size_t)elems);
            if (!y) die("out of memory");
            const int8_t *b2 = buf[n->inputs[1]];
            for (int64_t k = 0; k < elems; k++) {
                int32_t s = (int32_t)a[k] + (int32_t)b2[k];
                if (s > 127) s = 127;
                if (s < -128) s = -128;
                y[k] = (int8_t)s;
            }
        } else {
            die("unsupported op");
        }
        buf[n->out] = y;
        if (n->op == 0 || n->op == 1 || n->op == 2 || n->op == 3 || n->op == 10 ||
            n->op == 8 || n->op == 4) {
            char path[512];
            snprintf(path, sizeof path, "%s/%s.bin", out_dir, n->name);
            FILE *f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
            int32_t oc = pack.shapes[3 * n->out];
            int32_t oh2 = pack.shapes[3 * n->out + 1];
            int32_t ow2 = pack.shapes[3 * n->out + 2];
            fwrite(y, 1, (size_t)oc * oh2 * ow2, f);
            fclose(f);
            written++;
        }
    }
    printf("{\"nodes\": %d, \"written\": %d}\n", pack.n_nodes, written);
    return 0;
}
