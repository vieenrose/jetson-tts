// ggml offload runtime for vits-melo-tts-zh_en-8k — flow + dec(Vocos8k).
//
// Stage 1 (this file, parity-first): the Vocos8k decoder graph + a parity
// harness against ORT-dumped boundary tensors (see make_parity_vectors.py /
// export_parity_bins.py). Stage 2 adds the VITS2 transformer-coupling flow;
// stage 3 builds with GGML_CUDA on the Jetson toolchain.
//
// Conventions:
//  * ggml ne[] is fastest-first; a numpy array [a,b,c] loads as ne={c,b,a}.
//  * Activations flow in two layouts: [T, C] (ne0=T) for convs,
//    [C, T] (ne0=C) for LayerNorm / Linear matmuls. Transposes are explicit.
//  * Linear weights from the converter are ONNX MatMul-B layout [in, out]
//    (numpy) -> ne={out, in}; we materialize the [in-fastest] transpose once
//    at load so ggml_mul_mat(W_t, x) yields [out, T].
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------- constants
static const int   Z_CH        = 192;
static const int   DEC_DIM     = 256;
static const int   N_BLOCKS    = 8;
static const int   N_BINS      = 129;   // 256/2+1
static const int   N_FFT       = 256;
static const int   HOP         = 64;
static const float RESAMPLE    = 125.0f / (44100.0f / 512.0f); // 1.451247...
static const float LN_EPS      = 1e-5f;  // torch LayerNorm default
static const float MAG_CLAMP   = 9.0f;

// ----------------------------------------------------------------- model
struct Model {
    struct ggml_context *wctx = nullptr;   // weights (from gguf)
    struct ggml_context *xctx = nullptr;   // transposed linears etc.
    std::map<std::string, struct ggml_tensor *> t;

    struct ggml_tensor *get(const std::string &name) {
        auto it = t.find(name);
        if (it == t.end()) {
            fprintf(stderr, "missing tensor: %s\n", name.c_str());
            exit(1);
        }
        return it->second;
    }
    bool has(const std::string &name) const { return t.count(name) > 0; }
};

static bool load_model(const char *path, Model &m) {
    struct gguf_init_params p = { /*no_alloc=*/false, /*ctx=*/&m.wctx };
    struct gguf_context *g = gguf_init_from_file(path, p);
    if (!g) return false;
    const int n = gguf_get_n_tensors(g);
    for (int i = 0; i < n; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        m.t[name] = ggml_get_tensor(m.wctx, name);
    }
    gguf_free(g);

    // materialize transposed Linear weights ([in,out] numpy -> we want ne0=in)
    static const char *linears[] = {
        "dec.student.blocks.%d.pw1.weight", "dec.student.blocks.%d.pw2.weight",
    };
    size_t need = 64u * 1024 * 1024;
    struct ggml_init_params ip = { need, nullptr, false };
    m.xctx = ggml_init(ip);
    auto transpose_inplace = [&](const std::string &name) {
        struct ggml_tensor *w = m.get(name);             // ne = {out, in}
        struct ggml_tensor *wt = ggml_new_tensor_2d(m.xctx, GGML_TYPE_F32,
                                                    w->ne[1], w->ne[0]);
        float *src = (float *) w->data, *dst = (float *) wt->data;
        for (int64_t o = 0; o < w->ne[0]; o++)
            for (int64_t i = 0; i < w->ne[1]; i++)
                dst[o * w->ne[1] + i] = src[i * w->ne[0] + o];
        m.t[name + ".T"] = wt;
    };
    for (auto pat : linears)
        for (int b = 0; b < N_BLOCKS; b++) {
            char buf[128]; snprintf(buf, sizeof buf, pat, b);
            transpose_inplace(buf);
        }
    transpose_inplace("dec.student.head.weight");
    return true;
}

// ----------------------------------------------------------------- bins io
struct Bin { std::vector<int64_t> shape; std::vector<float> data; };

static std::map<std::string, Bin> load_bins(const std::string &dir) {
    std::map<std::string, Bin> out;
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) { fprintf(stderr, "no manifest in %s\n", dir.c_str()); exit(1); }
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream is(line);
        std::string name; is >> name;
        if (name.empty()) continue;
        Bin b; int64_t d, n = 1;
        while (is >> d) { b.shape.push_back(d); n *= d; }
        b.data.resize(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read((char *) b.data.data(), n * sizeof(float));
        if (!f) { fprintf(stderr, "short read: %s\n", name.c_str()); exit(1); }
        out[name] = std::move(b);
    }
    return out;
}

// ----------------------------------------------------------------- helpers
// LayerNorm over channels; x is [C, T] (ne0 = C). gamma/beta ne = {C}.
static struct ggml_tensor *layer_norm(struct ggml_context *ctx,
                                      struct ggml_tensor *x,
                                      struct ggml_tensor *gamma,
                                      struct ggml_tensor *beta) {
    x = ggml_norm(ctx, x, LN_EPS);
    x = ggml_mul(ctx, x, gamma);   // broadcast over ne1
    x = ggml_add(ctx, x, beta);
    return x;
}

// Linear via pre-transposed weight (ne = {in, out}); x is [in, T] -> [out, T]
static struct ggml_tensor *linear(struct ggml_context *ctx,
                                  struct ggml_tensor *w_t,
                                  struct ggml_tensor *x,
                                  struct ggml_tensor *bias /*ne={out}*/) {
    struct ggml_tensor *y = ggml_mul_mat(ctx, w_t, x);   // [out, T]
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// ----------------------------------------------------------------- dec graph
// in : z_masked [T, Z_CH] (ne0 = T), resample matrix R [T, T2] (built on host)
// out: audio [S] with S = (T2-1)*HOP
static struct ggml_tensor *build_dec(struct ggml_context *ctx, Model &m,
                                     struct ggml_tensor *z,    // [T, 192]
                                     struct ggml_tensor *R,    // [T, T2]
                                     struct ggml_tensor *gvec  // [256]
                                     ) {
    // 1. resample along T: out[t2, c] = sum_t R[t, t2] * z[t, c]
    struct ggml_tensor *x_tc = ggml_mul_mat(ctx, R, z);       // A=[T,T2] B=[T,C] -> [T2, C]

    // 2. conv_pre k=7 p=3 (+ cond(g) broadcast over T)
    x_tc = ggml_conv_1d(ctx, m.get("dec.student.conv_pre.weight"), x_tc, 1, 3, 1); // [T2, 256]
    {
        struct ggml_tensor *b = m.get("dec.student.conv_pre.bias");      // {256}
        x_tc = ggml_add(ctx, x_tc, ggml_reshape_2d(ctx, b, 1, DEC_DIM)); // bcast ne0
        // cond: 1x1 conv == matvec: cond_w {1,256,256} -> view [256,256]
        struct ggml_tensor *cw = m.get("dec.student.cond.weight");
        struct ggml_tensor *cw2 = ggml_reshape_2d(ctx, cw, DEC_DIM, DEC_DIM); // ne={in,out}
        struct ggml_tensor *cg = ggml_mul_mat(ctx, cw2,
            ggml_reshape_2d(ctx, gvec, DEC_DIM, 1));                      // [out,1]
        cg = ggml_add(ctx, cg, ggml_reshape_2d(ctx, m.get("dec.student.cond.bias"), DEC_DIM, 1));
        // x_tc is [T2, C]; cg is [C,1] -> need [1, C]
        x_tc = ggml_add(ctx, x_tc, ggml_cont(ctx, ggml_transpose(ctx, cg)));
    }

    // 3. norm_in ([C, T2] layout)
    struct ggml_tensor *x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));  // [C, T2]
    x_ct = layer_norm(ctx, x_ct, m.get("dec.student.norm_in.weight"),
                      m.get("dec.student.norm_in.bias"));

    // 4. ConvNeXt blocks
    for (int b = 0; b < N_BLOCKS; b++) {
        char nm[128];
        auto T = [&](const char *suffix) {
            snprintf(nm, sizeof nm, "dec.student.blocks.%d.%s", b, suffix);
            return m.get(nm);
        };
        char base[96]; snprintf(base, sizeof base, "dec.student.blocks.%d", b);
        const std::string pw1t = std::string(base) + ".pw1.weight.T";
        const std::string pw2t = std::string(base) + ".pw2.weight.T";
        struct ggml_tensor *res = x_ct;
        // dw conv k=7 p=3 groups=C ; data layout [T, C]
        struct ggml_tensor *h_tc = ggml_cont(ctx, ggml_transpose(ctx, x_ct));
        h_tc = ggml_conv_1d_dw(ctx, T("dw.weight"), h_tc, 1, 3, 1);        // [T, C]
        struct ggml_tensor *h_ct = ggml_cont(ctx, ggml_transpose(ctx, h_tc));
        h_ct = ggml_add(ctx, h_ct, ggml_reshape_2d(ctx, T("dw.bias"), DEC_DIM, 1));
        h_ct = layer_norm(ctx, h_ct, T("norm.weight"), T("norm.bias"));
        h_ct = linear(ctx, m.get(pw1t), h_ct, T("pw1.bias"));             // [768, T]
        h_ct = ggml_gelu_erf(ctx, h_ct);
        h_ct = linear(ctx, m.get(pw2t), h_ct, T("pw2.bias"));             // [256, T]
        h_ct = ggml_mul(ctx, h_ct, ggml_reshape_2d(ctx, T("gamma"), DEC_DIM, 1));
        x_ct = ggml_add(ctx, res, h_ct);
    }

    // 5. norm_out + head -> [258, T2]
    x_ct = layer_norm(ctx, x_ct, m.get("dec.student.norm_out.weight"),
                      m.get("dec.student.norm_out.bias"));
    struct ggml_tensor *h = linear(ctx, m.get("dec.student.head.weight.T"),
                                   x_ct, m.get("dec.student.head.bias"));  // [258, T2]

    // 6. mag/phase  (rows 0..128 mag, 129..257 phase)
    const int64_t T2 = h->ne[1];
    struct ggml_tensor *mag = ggml_view_2d(ctx, h, N_BINS, T2, h->nb[1], 0);
    mag = ggml_exp(ctx, ggml_clamp(ctx, ggml_cont(ctx, mag), -1e30f, MAG_CLAMP));
    struct ggml_tensor *ph  = ggml_cont(ctx, ggml_view_2d(ctx, h, N_BINS, T2,
                                        h->nb[1], N_BINS * sizeof(float)));
    struct ggml_tensor *real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));      // [129, T2]
    struct ggml_tensor *imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));

    // 7. iSTFT via conv_transpose_1d, stride HOP. data layout [T, Cin]
    struct ggml_tensor *real_tc = ggml_cont(ctx, ggml_transpose(ctx, real)); // [T2, 129]
    struct ggml_tensor *imag_tc = ggml_cont(ctx, ggml_transpose(ctx, imag));
    struct ggml_tensor *y =
        ggml_add(ctx,
            ggml_conv_transpose_1d(ctx, m.get("dec.student.istft.cos_w"), real_tc, HOP, 0, 1),
            ggml_conv_transpose_1d(ctx, m.get("dec.student.istft.sin_w"), imag_tc, HOP, 0, 1));
    // window normalisation: conv_transpose of ones with win_sq {1,1,256}->ne {256,1,1}
    struct ggml_tensor *ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T2, 1);
    ggml_set_name(ones, "ones");   // filled by caller
    ggml_set_input(ones);
    struct ggml_tensor *norm =
        ggml_conv_transpose_1d(ctx, m.get("dec.student.istft.win_sq"), ones, HOP, 0, 1);
    y = ggml_div(ctx, y, ggml_add(ctx, norm,
            ggml_new_f32(ctx, 1e-8f)));
    // 8. trim N_FFT/2 each side -> [S]
    const int64_t S = (T2 - 1) * HOP;
    y = ggml_cont(ctx, ggml_view_1d(ctx, ggml_reshape_1d(ctx, y, y->ne[0]),
                                    S, (N_FFT / 2) * sizeof(float)));
    ggml_set_output(y);
    return y;
}

// ----------------------------------------------------------------- main
int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "flowdec-f32.gguf";
    const char *bins_dir  = argc > 2 ? argv[2] : "testdata/bins";
    int n_threads         = argc > 3 ? atoi(argv[3]) : 4;

    Model m;
    if (!load_model(gguf_path, m)) { fprintf(stderr, "gguf load failed\n"); return 1; }
    fprintf(stderr, "loaded %zu tensors\n", m.t.size());

    auto bins = load_bins(bins_dir);
    const Bin &z_b = bins.at("z");          // [1,192,T]
    const Bin &mask_b = bins.at("y_mask");  // [1,1,T]
    const Bin &g_b = bins.at("g");          // [1,1,256]
    const Bin &y_ref = bins.at("y");        // [1,1,S]
    const int64_t T  = z_b.shape[2];
    const int64_t T2 = (int64_t)(T * RESAMPLE);
    const int64_t S  = (T2 - 1) * HOP;
    fprintf(stderr, "T=%lld T2=%lld S=%lld (ref S=%lld)\n",
            (long long)T, (long long)T2, (long long)S, (long long)y_ref.shape[2]);

    struct ggml_init_params gp = { 512u * 1024 * 1024, nullptr, false };
    struct ggml_context *ctx = ggml_init(gp);

    // inputs
    struct ggml_tensor *z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Z_CH);
    struct ggml_tensor *R = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T2);
    struct ggml_tensor *gv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DEC_DIM);
    // z_masked[t, c] = z[c, t] (npz layout [192, T]) * mask[t]
    for (int64_t c = 0; c < Z_CH; c++)
        for (int64_t t = 0; t < T; t++)
            ((float *) z->data)[c * T + t] =
                z_b.data[c * T + t] * mask_b.data[t];
    // ONNX Resize linear, half_pixel: src = (t2 + 0.5)/scale - 0.5
    memset(R->data, 0, ggml_nbytes(R));
    for (int64_t t2 = 0; t2 < T2; t2++) {
        float src = (t2 + 0.5f) / RESAMPLE - 0.5f;
        int64_t i0 = (int64_t) floorf(src);
        float w1 = src - i0;
        int64_t ia = i0 < 0 ? 0 : (i0 >= T ? T - 1 : i0);
        int64_t ib = i0 + 1 < 0 ? 0 : (i0 + 1 >= T ? T - 1 : i0 + 1);
        ((float *) R->data)[t2 * T + ia] += 1.0f - w1;
        ((float *) R->data)[t2 * T + ib] += w1;
    }
    memcpy(gv->data, g_b.data.data(), DEC_DIM * sizeof(float));

    struct ggml_tensor *y = build_dec(ctx, m, z, R, gv);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, y);
    // fill 'ones'
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor *n = ggml_graph_node(gf, i);
        if (n->name && strcmp(n->name, "ones") == 0)
            for (int64_t j = 0; j < n->ne[0]; j++) ((float *) n->data)[j] = 1.0f;
    }
    // also check leafs (inputs are leafs, not nodes)
    {
        struct ggml_tensor *t = ggml_get_tensor(ctx, "ones");
        if (t) for (int64_t j = 0; j < t->ne[0]; j++) ((float *) t->data)[j] = 1.0f;
    }

    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // parity vs y_ref
    const float *got = (const float *) y->data;
    const float *ref = y_ref.data.data();
    int64_t n = S < y_ref.shape[2] ? S : y_ref.shape[2];
    double max_abs = 0, ref_max = 0, sum_sq = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double) got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        if (fabs(ref[i]) > ref_max) ref_max = fabs(ref[i]);
        sum_sq += d * d;
    }
    printf("dec parity: n=%lld  max_abs=%.3e  rel=%.3e  rmse=%.3e  %s\n",
           (long long) n, max_abs, max_abs / (ref_max + 1e-12),
           sqrt(sum_sq / n),
           (max_abs / (ref_max + 1e-12)) < 5e-3 ? "OK" : "MISMATCH");
    return (max_abs / (ref_max + 1e-12)) < 5e-3 ? 0 : 2;
}
