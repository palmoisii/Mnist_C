#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"

typedef enum
{
    INITIALIZER_RANDOM,
    INITIALIZER_HE,
    INITIALIZER_XAVIER
} InitializerType;

typedef enum
{
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAGRAD
} OptimizerType;

typedef struct
{
    float *A1, *b1, *A2, *b2, *A3, *b3;
} NNParameters;

typedef struct
{
    float *dA1, *db1, *dA2, *db2, *dA3, *db3;
    float *mA1, *vA1, *mb1, *vb1;
    float *mA2, *vA2, *mb2, *vb2;
    float *mA3, *vA3, *mb3, *vb3;
    float *hA1, *hb1, *hA2, *hb2, *hA3, *hb3;
} NNMemory;

// パラメータを保存する関数
/*
 * 引数:
 *   filename: 保存するファイルの名前 (const char *)
 *   m: 行数 (int)
 *   n: 列数 (int)
 *   A: 保存する行列データへのポインタ (const float *)
 *   b: 保存するバイアスベクトルへのポインタ (const float *)
 * 戻り値: なし (void)
 */
void save(const char *filename, int m, int n, const float *A, const float *b)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }

    fwrite(A, sizeof(float), m * n, file);
    fwrite(b, sizeof(float), m, file);

    fclose(file);
}

// パラメータを読み込む関数
/*
 * 引数:
 *   filename: 読み込むファイルの名前 (const char *)
 *   m: 行数 (int)
 *   n: 列数 (int)
 *   A: 読み込んだ行列データを格納するポインタ (float *)
 *   b: 読み込んだバイアスベクトルを格納するポインタ (float *)
 * 戻り値: なし (void)
 */
void load(const char *filename, int m, int n, float *A, float *b)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s for reading\n", filename);
        return;
    }

    fread(A, sizeof(float), m * n, file);
    fread(b, sizeof(float), m, file);

    fclose(file);
}

// 可算関数：配列の各要素を足し合わせる
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: 加算する配列へのポインタ (const float *)
 *   o: 結果を格納する配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void add(int n, const float *x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] += x[i];
    }
}

// スケーリング関数：配列の各要素に定数を掛ける
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: 掛ける定数 (float)
 *   o: スケーリングされる配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void scale(int n, float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] *= x;
    }
}

// 初期化関数：配列の全要素を指定された値で初期化する
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: 初期化する値 (float)
 *   o: 初期化される配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void init(int n, float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}

// 初期化関数：配列の要素を-1から1の範囲で初期化する
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   o: ランダムに初期化される配列へのポインタ (float *)
 *   initializer: 初期化方法のタイプ (InitializerType)
 *   fan_in: 前の層のノード数 (int)
 *   fan_out: 次の層のノード数 (int)
 * 戻り値: なし (void)
 */
void optimize_init(int n, float *o, InitializerType initializer, int fan_in, int fan_out)
{
    float scale;
    switch (initializer)
    {
    case INITIALIZER_HE:
        scale = sqrt(2.0f / fan_in);
        break;
    case INITIALIZER_XAVIER:
        scale = sqrt(2.0f / (fan_in + fan_out));
        break;
    case INITIALIZER_RANDOM:
    default:
        scale = 1.0f;
        break;
    }

    for (int i = 0; i < n; i++)
    {
        float r = (float)rand() / RAND_MAX;
        o[i] = scale * (2.0f * r - 1.0f);
    }
}

// 全結合層の順伝播計算
/*
 * 引数:
 *   m: 出力の次元数 (int)
 *   n: 入力の次元数 (int)
 *   x: 入力ベクトルへのポインタ (const float *)
 *   A: 重み行列へのポインタ (const float *)
 *   b: バイアスベクトルへのポインタ (const float *)
 *   y: 出力ベクトルへのポインタ (float *)
 * 戻り値: なし (void)
 */
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    for (int i = 0; i < m; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
        {
            y[i] += A[i * n + j] * x[j];
        }
        y[i] += b[i];
    }
}

// 行列を表示する関数
/*
 * 引数:
 *   m: 行数 (int)
 *   n: 列数 (int)
 *   x: 表示する行列へのポインタ (const float *)
 * 戻り値: なし (void)
 */
void print(int m, int n, const float *x)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.4f ", x[i * n + j]);
        }
        printf("\n");
    }
}

// ReLU活性化関数
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: 入力配列へのポインタ (const float *)
 *   y: 出力配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void relu(int n, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = (x[i] > 0) ? x[i] : 0;
    }
}

// ソフトマックス関数
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: 入力配列へのポインタ (const float *)
 *   y: 出力配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void softmax(int n, const float *x, float *y)
{
    float max_x = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x[i] > max_x)
            max_x = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        y[i] = expf(x[i] - max_x);
        sum += y[i];
    }

    for (int i = 0; i < n; i++)
    {
        y[i] /= sum;
    }
}

// ソフトマックス with クロスエントロピー誤差の逆伝播
/*
 * 引数:
 *   n: クラス数 (int)
 *   y: ソフトマックス関数の出力配列へのポインタ (const float *)
 *   t: 正解ラベル (unsigned char)
 *   dEdx: 勾配を格納する配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    for (int k = 0; k < n; k++)
    {
        if (k == t)
        {
            dEdx[k] = y[k] - 1.0f;
        }
        else
        {
            dEdx[k] = y[k];
        }
    }
}

// ReLU関数の逆伝播
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: ReLU関数の入力配列へのポインタ (const float *)
 *   dEdy: 上流から伝わってきた勾配配列へのポインタ (const float *)
 *   dEdx: 計算された勾配を格納する配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)
{
    for (int i = 0; i < n; i++)
    {
        dEdx[i] = (x[i] > 0) ? dEdy[i] : 0;
    }
}

// 全結合層の逆伝播
/*
 * 引数:
 *   m: 出力の次元数 (int)
 *   n: 入力の次元数 (int)
 *   x: 入力ベクトルへのポインタ (const float *)
 *   dEdy: 上流から伝わってきた勾配ベクトルへのポインタ (const float *)
 *   A: 重み行列へのポインタ (const float *)
 *   dEdA: 重み行列の勾配を格納する配列へのポインタ (float *)
 *   dEdb: バイアスの勾配を格納する配列へのポインタ (float *)
 *   dEdx: 入力に関する勾配を格納する配列へのポインタ (float *)
 * 戻り値: なし (void)
 */
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        dEdb[i] = dEdy[i];
    }

    for (int i = 0; i < n; i++)
    {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            dEdx[i] += A[j * n + i] * dEdy[j];
        }
    }
}

// 配列をシャッフルする関数
/*
 * 引数:
 *   n: 配列の要素数 (int)
 *   x: シャッフルする整数配列へのポインタ (int *)
 * 戻り値: なし (void)
 */
void shuffle(int n, int *x)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = i;
    }

    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        int temp = x[i];
        x[i] = x[j];
        x[j] = temp;
    }
}

// 交差エントロピー誤差
/*
 * 引数:
 *   y: モデルの出力確率分布配列へのポインタ (const float *)
 *   t: 正解ラベル (int)
 * 戻り値: 交差エントロピー誤差 (float)
 */
float cross_entropy_error(const float *y, int t)
{
    return -log(y[t] + 1e-7);
}

// 誤差順伝搬
/*
 * 引数:
 *   x: 入力データ (const float *)
 *   A1, b1: 第1層の重みとバイアス (const float *)
 *   A2, b2: 第2層の重みとバイアス (const float *)
 *   A3, b3: 第3層の重みとバイアス (const float *)
 * 戻り値: ソフトマックス関数の出力 (float *)
 */
float *forward6(const float *x,
                const float *A1, const float *b1,
                const float *A2, const float *b2,
                const float *A3, const float *b3)
{
    float *fc1_out = malloc(50 * sizeof(float));
    float *relu1_out = malloc(50 * sizeof(float));
    float *fc2_out = malloc(100 * sizeof(float));
    float *relu2_out = malloc(100 * sizeof(float));
    float *fc3_out = malloc(10 * sizeof(float));
    float *softmax_out = malloc(10 * sizeof(float));

    // FC1
    fc(50, 784, x, A1, b1, fc1_out);
    // ReLU1
    relu(50, fc1_out, relu1_out);
    // FC2
    fc(100, 50, relu1_out, A2, b2, fc2_out);
    // ReLU2
    relu(100, fc2_out, relu2_out);
    // FC3
    fc(10, 100, relu2_out, A3, b3, fc3_out);
    // Softmax
    softmax(10, fc3_out, softmax_out);

    free(fc1_out);
    free(relu1_out);
    free(fc2_out);
    free(relu2_out);
    free(fc3_out);

    return softmax_out;
}

// 誤差逆伝播
/*
 * 引数:
 *   x: 入力データ (const float *)
 *   t: 正解ラベル (unsigned char)
 *   A1, b1: 第1層の重みとバイアス (const float *)
 *   A2, b2: 第2層の重みとバイアス (const float *)
 *   A3, b3: 第3層の重みとバイアス (const float *)
 *   dA1, db1: 第1層の重みとバイアスの勾配 (float *)
 *   dA2, db2: 第2層の重みとバイアスの勾配 (float *)
 *   dA3, db3: 第3層の重みとバイアスの勾配 (float *)
 * 戻り値: なし (void)
 */
void backward6(const float *x, unsigned char t,
               const float *A1, const float *b1,
               const float *A2, const float *b2,
               const float *A3, const float *b3,
               float *dA1, float *db1,
               float *dA2, float *db2,
               float *dA3, float *db3)
{
    // 順伝播の中間結果を保存
    float fc1_out[50], relu1_out[50], fc2_out[100], relu2_out[100], fc3_out[10], y[10];

    // 順伝播
    fc(50, 784, x, A1, b1, fc1_out);
    relu(50, fc1_out, relu1_out);
    fc(100, 50, relu1_out, A2, b2, fc2_out);
    relu(100, fc2_out, relu2_out);
    fc(10, 100, relu2_out, A3, b3, fc3_out);
    softmax(10, fc3_out, y);

    // Softmax with Loss の逆伝播
    float dEdy[10];
    softmaxwithloss_bwd(10, y, t, dEdy);

    // FC3 の逆伝播
    float dEdrelu2[100];
    fc_bwd(10, 100, relu2_out, dEdy, A3, dA3, db3, dEdrelu2);

    // ReLU2 の逆伝播
    float dEdfc2[100];
    relu_bwd(100, fc2_out, dEdrelu2, dEdfc2);

    // FC2 の逆伝播
    float dEdrelu1[50];
    fc_bwd(100, 50, relu1_out, dEdfc2, A2, dA2, db2, dEdrelu1);

    // ReLU1 の逆伝播
    float dEdfc1[50];
    relu_bwd(50, fc1_out, dEdrelu1, dEdfc1);

    // FC1 の逆伝播
    float dEdx[784]; // 使用しないが、fc_bwd の引数として必要
    fc_bwd(50, 784, x, dEdfc1, A1, dA1, db1, dEdx);
}

// SGD (Stochastic Gradient Descent) 更新
/*
 * 引数:
 *   param: 更新するパラメータ (float *)
 *   grad: パラメータの勾配 (const float *)
 *   size: パラメータの要素数 (int)
 *   learning_rate: 学習率 (float)
 * 戻り値: なし (void)
 */
void update_sgd(float *param, float *grad, int size, float learning_rate)
{
    for (int i = 0; i < size; i++)
    {
        param[i] -= learning_rate * grad[i];
    }
}

// Adam 更新
/*
 * 引数:
 *   param: 更新するパラメータ (float *)
 *   grad: パラメータの勾配 (const float *)
 *   m: 1次モーメント (float *)
 *   v: 2次モーメント (float *)
 *   size: パラメータの要素数 (int)
 *   learning_rate: 学習率 (float)
 *   t: 現在のタイムステップ (int)
 * 戻り値: なし (void)
 */
void update_adam(float *param, float *grad, float *m, float *v, int size, float learning_rate, int t)
{
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    for (int i = 0; i < size; i++)
    {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / (1 - powf(beta1, t + 1));
        float v_hat = v[i] / (1 - powf(beta2, t + 1));
        param[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// AdaGrad 更新
/*
 * 引数:
 *   param: 更新するパラメータ (float *)
 *   grad: パラメータの勾配 (const float *)
 *   h: 累積2乗勾配 (float *)
 *   size: パラメータの要素数 (int)
 *   learning_rate: 学習率 (float)
 * 戻り値: なし (void)
 */
void update_adagrad(float *param, float *grad, float *h, int size, float learning_rate)
{
    const float epsilon = 1e-8f;

    for (int i = 0; i < size; i++)
    {
        h[i] += grad[i] * grad[i];
        param[i] -= learning_rate * grad[i] / (sqrtf(h[i]) + epsilon);
    }
}

// ニューラルネットワークのパラメータを初期化
/*
 * 引数:
 *   params: 初期化するパラメータ構造体へのポインタ (NNParameters *)
 *   width: 入力画像の幅 (int)
 *   height: 入力画像の高さ (int)
 *   initializer: 初期化方法のタイプ (InitializerType)
 * 戻り値: なし (void)
 */
void initialize_parameters(NNParameters *params, int width, int height, InitializerType initializer)
{
    params->A1 = malloc(50 * 784 * sizeof(float));
    params->b1 = malloc(50 * sizeof(float));
    params->A2 = malloc(100 * 50 * sizeof(float));
    params->b2 = malloc(100 * sizeof(float));
    params->A3 = malloc(10 * 100 * sizeof(float));
    params->b3 = malloc(10 * sizeof(float));

    optimize_init(50 * 784, params->A1, initializer, 784, 50);
    optimize_init(50, params->b1, INITIALIZER_RANDOM, 0, 0);
    optimize_init(100 * 50, params->A2, initializer, 50, 100);
    optimize_init(100, params->b2, INITIALIZER_RANDOM, 0, 0);
    optimize_init(10 * 100, params->A3, initializer, 100, 10);
    optimize_init(10, params->b3, INITIALIZER_RANDOM, 0, 0);
}

// ニューラルネットワークの計算に必要なメモリを割り当て
/*
 * 引数:
 *   memory: メモリ構造体へのポインタ (NNMemory *)
 *   params: パラメータ構造体へのポインタ (NNParameters *)
 *   optimizer: 最適化方法のタイプ (OptimizerType)
 * 戻り値: なし (void)
 */
void allocate_memory(NNMemory *memory, NNParameters *params, OptimizerType optimizer)
{
    memory->dA1 = malloc(50 * 784 * sizeof(float));
    memory->db1 = malloc(50 * sizeof(float));
    memory->dA2 = malloc(100 * 50 * sizeof(float));
    memory->db2 = malloc(100 * sizeof(float));
    memory->dA3 = malloc(10 * 100 * sizeof(float));
    memory->db3 = malloc(10 * sizeof(float));

    if (optimizer == OPTIMIZER_ADAM)
    {
        memory->mA1 = calloc(50 * 784, sizeof(float));
        memory->vA1 = calloc(50 * 784, sizeof(float));
        memory->mb1 = calloc(50, sizeof(float));
        memory->vb1 = calloc(50, sizeof(float));
        memory->mA2 = calloc(100 * 50, sizeof(float));
        memory->vA2 = calloc(100 * 50, sizeof(float));
        memory->mb2 = calloc(100, sizeof(float));
        memory->vb2 = calloc(100, sizeof(float));
        memory->mA3 = calloc(10 * 100, sizeof(float));
        memory->vA3 = calloc(10 * 100, sizeof(float));
        memory->mb3 = calloc(10, sizeof(float));
        memory->vb3 = calloc(10, sizeof(float));
    }
    else if (optimizer == OPTIMIZER_ADAGRAD)
    {
        memory->hA1 = calloc(50 * 784, sizeof(float));
        memory->hb1 = calloc(50, sizeof(float));
        memory->hA2 = calloc(100 * 50, sizeof(float));
        memory->hb2 = calloc(100, sizeof(float));
        memory->hA3 = calloc(10 * 100, sizeof(float));
        memory->hb3 = calloc(10, sizeof(float));
    }
}

// 進捗バーを表示する関数
/*
 * 引数:
 *   current: 現在の進捗 (int)
 *   total: 全体の数 (int)
 *   bar_width: 進捗バーの幅 (int)
 * 戻り値: なし (void)
 */
void print_progress_bar(int current, int total, int bar_width)
{
    float progress = (float)current / total;
    int filled_width = (int)(bar_width * progress);

    printf("\r[");
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < filled_width)
        {
            printf("=");
        }
        else
        {
            printf(" ");
        }
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
}

// 1エポックの訓練を実行
/*
 * 引数:
 *   params: パラメータ構造体へのポインタ (NNParameters *)
 *   memory: メモリ構造体へのポインタ (NNMemory *)
 *   train_x: 訓練データの入力 (float *)
 *   train_y: 訓練データのラベル (unsigned char *)
 *   train_count: 訓練データの数 (int)
 *   batch_size: バッチサイズ (int)
 *   learning_rate: 学習率 (float)
 *   optimizer: 最適化方法のタイプ (OptimizerType)
 *   index: シャッフル用インデックス配列 (int *)
 *   epoch: 現在のエポック数 (int)
 * 戻り値: なし (void)
 */
void train_epoch(NNParameters *params, NNMemory *memory, float *train_x, unsigned char *train_y,
                 int train_count, int batch_size, float learning_rate, OptimizerType optimizer,
                 int *index, int epoch)
{
    shuffle(train_count, index);

    int total_batches = train_count / batch_size;
    time_t start_time = time(NULL);

    for (int batch = 0; batch < total_batches; batch++)
    {
        init(50 * 784, 0.0f, memory->dA1);
        init(50, 0.0f, memory->db1);
        init(100 * 50, 0.0f, memory->dA2);
        init(100, 0.0f, memory->db2);
        init(10 * 100, 0.0f, memory->dA3);
        init(10, 0.0f, memory->db3);

        for (int i = 0; i < batch_size; i++)
        {
            int idx = index[batch * batch_size + i];
            float *x = train_x + idx * 784;
            unsigned char t = train_y[idx];

            float batch_dA1[50 * 784], batch_db1[50], batch_dA2[100 * 50], batch_db2[100], batch_dA3[10 * 100], batch_db3[10];

            backward6(x, t, params->A1, params->b1, params->A2, params->b2, params->A3, params->b3,
                      batch_dA1, batch_db1, batch_dA2, batch_db2, batch_dA3, batch_db3);

            add(50 * 784, batch_dA1, memory->dA1);
            add(50, batch_db1, memory->db1);
            add(100 * 50, batch_dA2, memory->dA2);
            add(100, batch_db2, memory->db2);
            add(10 * 100, batch_dA3, memory->dA3);
            add(10, batch_db3, memory->db3);
        }

        scale(50 * 784, 1.0f / batch_size, memory->dA1);
        scale(50, 1.0f / batch_size, memory->db1);
        scale(100 * 50, 1.0f / batch_size, memory->dA2);
        scale(100, 1.0f / batch_size, memory->db2);
        scale(10 * 100, 1.0f / batch_size, memory->dA3);
        scale(10, 1.0f / batch_size, memory->db3);

        if (optimizer == OPTIMIZER_SGD)
        {
            update_sgd(params->A1, memory->dA1, 50 * 784, learning_rate);
            update_sgd(params->b1, memory->db1, 50, learning_rate);
            update_sgd(params->A2, memory->dA2, 100 * 50, learning_rate);
            update_sgd(params->b2, memory->db2, 100, learning_rate);
            update_sgd(params->A3, memory->dA3, 10 * 100, learning_rate);
            update_sgd(params->b3, memory->db3, 10, learning_rate);
        }
        else if (optimizer == OPTIMIZER_ADAM)
        {
            int t = epoch * train_count / batch_size + batch;
            update_adam(params->A1, memory->dA1, memory->mA1, memory->vA1, 50 * 784, learning_rate, t);
            update_adam(params->b1, memory->db1, memory->mb1, memory->vb1, 50, learning_rate, t);
            update_adam(params->A2, memory->dA2, memory->mA2, memory->vA2, 100 * 50, learning_rate, t);
            update_adam(params->b2, memory->db2, memory->mb2, memory->vb2, 100, learning_rate, t);
            update_adam(params->A3, memory->dA3, memory->mA3, memory->vA3, 10 * 100, learning_rate, t);
            update_adam(params->b3, memory->db3, memory->mb3, memory->vb3, 10, learning_rate, t);
        }
        else if (optimizer == OPTIMIZER_ADAGRAD)
        {
            update_adagrad(params->A1, memory->dA1, memory->hA1, 50 * 784, learning_rate);
            update_adagrad(params->b1, memory->db1, memory->hb1, 50, learning_rate);
            update_adagrad(params->A2, memory->dA2, memory->hA2, 100 * 50, learning_rate);
            update_adagrad(params->b2, memory->db2, memory->hb2, 100, learning_rate);
            update_adagrad(params->A3, memory->dA3, memory->hA3, 10 * 100, learning_rate);
            update_adagrad(params->b3, memory->db3, memory->hb3, 10, learning_rate);
        }

        // 進捗バーの表示
        print_progress_bar(batch + 1, total_batches, 50);
    }

    time_t end_time = time(NULL);
    double elapsed_time = difftime(end_time, start_time);

    printf(" - %d/%d [%.2f sec]\n", total_batches, total_batches, elapsed_time);
}

// モデルの評価を実行
/*
 * 引数:
 *   params: パラメータ構造体へのポインタ (NNParameters *)
 *   test_x: テストデータの入力 (float *)
 *   test_y: テストデータのラベル (unsigned char *)
 *   test_count: テストデータの数 (int)
 *   width: 入力画像の幅 (int)
 *   height: 入力画像の高さ (int)
 *   loss: 計算された損失を格納するポインタ (float *)
 *   accuracy: 計算された精度を格納するポインタ (float *)
 * 戻り値: なし (void)
 */
void evaluate_model(NNParameters *params, float *test_x, unsigned char *test_y,
                    int test_count, int width, int height, float *loss, float *accuracy)
{
    float total_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < test_count; i++)
    {
        float *x = test_x + i * width * height;
        unsigned char t = test_y[i];

        float *y = forward6(x, params->A1, params->b1, params->A2, params->b2, params->A3, params->b3);
        total_loss += cross_entropy_error(y, t);

        int prediction = 0;
        for (int j = 1; j < 10; j++)
        {
            if (y[j] > y[prediction])
                prediction = j;
        }
        if (prediction == t)
            correct++;

        free(y);
    }

    *loss = total_loss / test_count;
    *accuracy = (float)correct / test_count;
}

// 学習したパラメータを保存
/*
 * 引数:
 *   params: 保存するパラメータ構造体へのポインタ (NNParameters *)
 *   params_name: 保存するパラメータ構造体の名前(char)
 * 戻り値: なし (void)
 */
void save_parameters(NNParameters *params, const char *params_name)
{
    char filename[256];

    snprintf(filename, sizeof(filename), "%s_fc1.dat", params_name);
    save(filename, 50, 784, params->A1, params->b1);

    snprintf(filename, sizeof(filename), "%s_fc2.dat", params_name);
    save(filename, 100, 50, params->A2, params->b2);

    snprintf(filename, sizeof(filename), "%s_fc3.dat", params_name);
    save(filename, 10, 100, params->A3, params->b3);
}

// 割り当てたメモリを解放
/*
 * 引数:
 *   memory: メモリ構造体へのポインタ (NNMemory *)
 *   params: パラメータ構造体へのポインタ (NNParameters *)
 *   optimizer: 最適化方法のタイプ (OptimizerType)
 * 戻り値: なし (void)
 */
void free_memory(NNMemory *memory, NNParameters *params, OptimizerType optimizer)
{
    free(params->A1);
    free(params->b1);
    free(params->A2);
    free(params->b2);
    free(params->A3);
    free(params->b3);

    free(memory->dA1);
    free(memory->db1);
    free(memory->dA2);
    free(memory->db2);
    free(memory->dA3);
    free(memory->db3);

    if (optimizer == OPTIMIZER_ADAM)
    {
        free(memory->mA1);
        free(memory->vA1);
        free(memory->mb1);
        free(memory->vb1);
        free(memory->mA2);
        free(memory->vA2);
        free(memory->mb2);
        free(memory->vb2);
        free(memory->mA3);
        free(memory->vA3);
        free(memory->mb3);
        free(memory->vb3);
    }
    else if (optimizer == OPTIMIZER_ADAGRAD)
    {
        free(memory->hA1);
        free(memory->hb1);
        free(memory->hA2);
        free(memory->hb2);
        free(memory->hA3);
        free(memory->hb3);
    }
}

// 6層ニューラルネットワークの学習を実行
/*
 * 引数:
 *   train_x: 訓練データの入力 (float *)
 *   train_y: 訓練データのラベル (unsigned char *)
 *   train_count: 訓練データの数 (int)
 *   test_x: テストデータの入力 (float *)
 *   test_y: テストデータのラベル (unsigned char *)
 *   test_count: テストデータの数 (int)
 *   width: 入力画像の幅 (int)
 *   height: 入力画像の高さ (int)
 *   optimizer: 最適化方法のタイプ (OptimizerType)
 *   initializer: 初期化方法のタイプ (InitializerType)
 *   params_name: 保存するパラメータ構造体の名前(const char *)
 * 戻り値: なし (void)
 */
void train_nn6(float *train_x, unsigned char *train_y, int train_count,
               float *test_x, unsigned char *test_y, int test_count,
               int width, int height, OptimizerType optimizer, InitializerType initializer, const char *params_name)
{
    // ハイパーパラーメーターの設定
    const int epochs = 10;
    const int batch_size = 100;
    const float learning_rate = 0.01f;

    // パラメータの初期化
    NNParameters params;
    initialize_parameters(&params, width, height, initializer);

    // メモリの割り当て
    NNMemory memory;
    allocate_memory(&memory, &params, optimizer);

    int *index = malloc(train_count * sizeof(int));

    // エポックごとの学習
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        train_epoch(&params, &memory, train_x, train_y, train_count, batch_size, learning_rate, optimizer, index, epoch);

        float loss, accuracy;
        evaluate_model(&params, test_x, test_y, test_count, width, height, &loss, &accuracy);

        printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%\n", epoch + 1, loss, accuracy * 100);
    }

    // 学習したパラメータの保存
    save_parameters(&params, params_name);

    // メモリの開放
    free_memory(&memory, &params, optimizer);
    free(index);
}

// メイン関数
/*
 * 引数:
 *   argc: コマンドライン引数の数 (int)
 *   argv: コマンドライン引数の配列 (char *[])
 * 戻り値: プログラムの終了状態 (int)
 */
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [train|infer] [args...]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0)
    {
        // 学習モード
        srand((unsigned int)time(NULL)); 

        float *train_x = NULL;
        unsigned char *train_y = NULL;
        int train_count = -1;
        float *test_x = NULL;
        unsigned char *test_y = NULL;
        int test_count = -1;
        int width = -1;
        int height = -1;

        // mnist読み込み
        load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);

        // パラメータの初期化
        float *A1 = malloc(50 * 784 * sizeof(float));
        float *b1 = malloc(50 * sizeof(float));
        float *A2 = malloc(100 * 50 * sizeof(float));
        float *b2 = malloc(100 * sizeof(float));
        float *A3 = malloc(10 * 100 * sizeof(float));
        float *b3 = malloc(10 * sizeof(float));

        OptimizerType optimizer = OPTIMIZER_SGD;
        InitializerType initializer = INITIALIZER_RANDOM;
        const char *params_name = "default"; // デフォルト値を設定

        if (argc > 2)
        {
            if (strcmp(argv[2], "adam") == 0)
            {
                optimizer = OPTIMIZER_ADAM;
            }
            else if (strcmp(argv[2], "adagrad") == 0)
            {
                optimizer = OPTIMIZER_ADAGRAD;
            }
        }

        if (argc > 3)
        {
            if (strcmp(argv[3], "he") == 0)
            {
                initializer = INITIALIZER_HE;
            }
            else if (strcmp(argv[3], "xavier") == 0)
            {
                initializer = INITIALIZER_XAVIER;
            }
        }

        if (argc > 4)
        {
            params_name = argv[4]; // 4番目の引数をparams_nameとして使用
        }

        train_nn6(train_x, train_y, train_count, test_x, test_y, test_count, width, height, optimizer, initializer, params_name);

        // メモリの解放
        free(train_x);
        free(train_y);
        free(test_x);
        free(test_y);
        free(A1);
        free(b1);
        free(A2);
        free(b2);
        free(A3);
        free(b3);
    }
    else if (strcmp(argv[1], "infer") == 0)
    {
        // 推論モード
        if (argc != 6)
        {
            fprintf(stderr, "Usage: %s infer <fc1_file> <fc2_file> <fc3_file> <image_file>\n", argv[0]);
            return 1;
        }

        float *A1 = malloc(sizeof(float) * 784 * 50);
        float *b1 = malloc(sizeof(float) * 50);
        float *A2 = malloc(sizeof(float) * 50 * 100);
        float *b2 = malloc(sizeof(float) * 100);
        float *A3 = malloc(sizeof(float) * 100 * 10);
        float *b3 = malloc(sizeof(float) * 10);

        // パラメータの読み込み
        load(argv[2], 50, 784, A1, b1);
        load(argv[3], 100, 50, A2, b2);
        load(argv[4], 10, 100, A3, b3);

        // 画像の読み込み
        float *x = load_mnist_bmp(argv[5]);
        if (x == NULL)
        {
            fprintf(stderr, "Failed to load image file: %s\n", argv[5]);
            return 1;
        }

        // 推論の実行
        float *y = forward6(x, A1, b1, A2, b2, A3, b3);

        // 結果の表示
        int prediction = 0;
        for (int i = 1; i < 10; i++)
        {
            if (y[i] > y[prediction])
            {
                prediction = i;
            }
        }
        printf("Predicted digit: %d\n", prediction);

        // メモリの解放
        free(A1);
        free(b1);
        free(A2);
        free(b2);
        free(A3);
        free(b3);
        free(x);
        free(y);
    }
    else
    {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        return 1;
    }

    return 0;
}