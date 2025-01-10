#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CHILDREN 256   // ASCII想定の最大子ノード数
#define RESERVOIR_SIZE 64  // リザバーの次元数 (論文例: 256 など)
#define MAX_DEPTH 16       // Trie の固定深度 (論文例: 16 や 64)
#define ALPHA 0.85f        // 減衰係数
#define RHO 0.9f           // スペクトル半径 (簡易的にこの係数でスケーリング)

// -------------------------
// 乱数まわりのヘルパー
// -------------------------
static float rand_float(void) {
    // -1.0 ~ +1.0 の一様乱数
    return (float)rand() / (float)(RAND_MAX/2) - 1.0f;
}

// -------------------------
// tanh 活性化関数
// -------------------------
static float activate_tanh(float x) {
    // 数値安定性を考慮した簡易 tanh
    const float e1 = expf(x);
    const float e2 = expf(-x);
    return (e1 - e2) / (e1 + e2);
}

// -------------------------
// 行列 x ベクトル積 (サイズ: RESERVOIR_SIZE x RESERVOIR_SIZE)
// out = W * in
// -------------------------
static void matvec(const float* W, const float* in, float* out) {
    for (int i = 0; i < RESERVOIR_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < RESERVOIR_SIZE; j++) {
            sum += W[i * RESERVOIR_SIZE + j] * in[j];
        }
        out[i] = sum;
    }
}

// -------------------------
// Trieノード構造体
// -------------------------
typedef struct TrieNode {
    struct TrieNode* children[MAX_CHILDREN];
    // 固定深度管理 (rootのdepth=0, 1, 2, ..., D-1)
    int depth;
    // 子があるかどうかのフラグ
    int is_leaf;
} TrieNode;

// -------------------------
// Trieノード作成
// -------------------------
TrieNode* create_trie_node(int depth) {
    TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
    node->depth = depth;
    node->is_leaf = 0;
    for(int i = 0; i < MAX_CHILDREN; i++) {
        node->children[i] = NULL;
    }
    return node;
}

// -------------------------
// Trie への挿入 (深度MAX_DEPTHで打ち切る)
// -------------------------
void trie_insert(TrieNode* root, const char* str) {
    TrieNode* cur = root;
    int length = (int)strlen(str);
    for(int i = 0; i < length && i < MAX_DEPTH; i++) {
        unsigned char c = (unsigned char)str[i];
        if(cur->children[c] == NULL) {
            cur->children[c] = create_trie_node(cur->depth + 1);
        }
        cur = cur->children[c];
    }
    cur->is_leaf = 1;
}

// ---------------------------------------------------------
// リザバー用の重み行列 W^(l) を深度ごとに用意
//   reservoir_weights[l][ i*RESERVOIR_SIZE + j ]
//   => depth l の RESERVOIR_SIZE×RESERVOIR_SIZE 行列
// ---------------------------------------------------------
static float** reservoir_weights = NULL;

// -------------------------
// リザバー重みの初期化
//  - まず乱数で埋める
//  - スペクトル半径を RHO 程度にするため、
//    固有値計算は省略し、雑に(1/norm)でスケーリング
// -------------------------
void init_reservoir_weights(int depth_count) {
    reservoir_weights = (float**)malloc(sizeof(float*) * depth_count);
    srand((unsigned int)time(NULL));

    for(int l = 0; l < depth_count; l++) {
        reservoir_weights[l] = (float*)malloc(sizeof(float) * RESERVOIR_SIZE * RESERVOIR_SIZE);
        // 乱数で初期化
        float norm_sum = 0.0f;
        for(int i = 0; i < RESERVOIR_SIZE * RESERVOIR_SIZE; i++) {
            reservoir_weights[l][i] = rand_float();
            norm_sum += fabsf(reservoir_weights[l][i]);
        }
        // 平均絶対値でスケーリングする簡易版
        float avg_abs = norm_sum / (RESERVOIR_SIZE * RESERVOIR_SIZE);
        float scale = (avg_abs > 1e-5f)? (RHO / avg_abs) : 1.0f;
        for(int i = 0; i < RESERVOIR_SIZE * RESERVOIR_SIZE; i++) {
            reservoir_weights[l][i] *= scale;
        }
    }
}

// -------------------------
// リザバー状態を1ステップ更新する
//  h_{l+1} = alpha * tanh(W^(l) * h_l + noise)
// -------------------------
void reservoir_update(const float* Wl, float* h_inout) {
    static float tmp[RESERVOIR_SIZE];
    // W^(l) * h(l)
    matvec(Wl, h_inout, tmp);
    // ノイズを加える (非常に小さい値)
    for(int i = 0; i < RESERVOIR_SIZE; i++) {
        tmp[i] += 0.01f * rand_float(); 
    }
    // tanh + alpha
    for(int i = 0; i < RESERVOIR_SIZE; i++) {
        float z = activate_tanh(tmp[i]);
        h_inout[i] = ALPHA * z;
    }
}

// -------------------------
// (1) 文字列を1つ与え、Trieを深さ方向に進む
// (2) 各深度ごとにリザバー状態を更新
// => 最終的な状態ベクトル h(D) を得る
// -------------------------
void trie_reservoir_forward(TrieNode* root, const char* input, float* h_state) {
    TrieNode* cur = root;
    int length = (int)strlen(input);

    // リザバー状態 h_state は呼び出し前にゼロクリアしておく想定

    for(int i = 0; i < length && i < MAX_DEPTH; i++) {
        unsigned char c = (unsigned char)input[i];
        if(cur->children[c] == NULL) {
            // ノードが存在しなければ中断 (実運用なら生成 or 例外処理)
            break;
        }
        // depth l に応じた重みで更新
        int l = cur->depth;  // 0,1,2... (最大MAX_DEPTH-1)
        reservoir_update(reservoir_weights[l], h_state);

        cur = cur->children[c];
    }

    // 入力を最後まで/最大深度まで辿った時点で h_state が「最終状態」
    // ここでは何もしない
}

// -------------------------
// リードアウト部：単純な全結合＋softmax想定
//   out_dim = 語彙数 (サンプルなので少数にしている)
// -------------------------
#define OUT_DIM 4   // 出力次元(例: 4語彙だとする)
static float readout_weights[OUT_DIM][RESERVOIR_SIZE]; // 語彙数 × リザバー次元

// 初期化
void init_readout() {
    for(int i = 0; i < OUT_DIM; i++) {
        for(int j = 0; j < RESERVOIR_SIZE; j++) {
            readout_weights[i][j] = 0.01f * rand_float();
        }
    }
}

// 全結合 + softmax
void readout_forward(const float* h_state, float* out_probs) {
    // z = W * h_state
    float sum_exp = 0.0f;
    for(int i = 0; i < OUT_DIM; i++) {
        float z = 0.0f;
        for(int j = 0; j < RESERVOIR_SIZE; j++) {
            z += readout_weights[i][j] * h_state[j];
        }
        // ここでは簡易に exp(z)
        out_probs[i] = expf(z);
        sum_exp += out_probs[i];
    }
    // ソフトマックス正規化
    for(int i = 0; i < OUT_DIM; i++) {
        out_probs[i] /= sum_exp;
    }
}

// リードアウト部の単純学習 (クロスエントロピー誤差に対する勾配下降の例)
void readout_train(const float* h_state, int gold_index, float lr) {
    // 順伝搬
    float probs[OUT_DIM];
    readout_forward(h_state, probs);

    // 勾配 = (pred - onehot(gold)) * h_state
    for(int i = 0; i < OUT_DIM; i++) {
        float grad = probs[i];
        if(i == gold_index) {
            grad -= 1.0f;
        }
        // パラメータ更新
        for(int j = 0; j < RESERVOIR_SIZE; j++) {
            readout_weights[i][j] -= lr * grad * h_state[j];
        }
    }
}

// -------------------------
// メイン関数
// -------------------------
int main(void) {
    // 1. Trie 構築 (サンプル文字列をいくつか挿入)
    TrieNode* root = create_trie_node(0);
    trie_insert(root, "hello");
    trie_insert(root, "help");
    trie_insert(root, "helium");
    trie_insert(root, "cat");
    trie_insert(root, "dog");
    // ... 必要に応じて単語や文を追加

    // 2. リザバー重み初期化
    init_reservoir_weights(MAX_DEPTH);

    // 3. リードアウト部初期化
    init_readout();

    // 4. 簡易的なトレーニング例
    //
    //   例: 入力文字列 -> Trie+Reservoir Forward -> リードアウトForward -> 誤差逆伝搬
    //   gold_index は (OUT_DIM=4 語彙のどれか) という想定
    //
    //   ここでは "hello" を入力して gold_index=0 (「hello」ラベル) を教師データにするなど
    //
    float learning_rate = 0.01f;
    for(int epoch = 0; epoch < 100; epoch++) {
        // 例として "hello" で学習
        float h_state[RESERVOIR_SIZE];
        memset(h_state, 0, sizeof(h_state));  // 初期状態を0でリセット
        trie_reservoir_forward(root, "hello", h_state);
        readout_train(h_state, 0, learning_rate);  // gold_index=0

        // 例として "cat" で学習
        memset(h_state, 0, sizeof(h_state));
        trie_reservoir_forward(root, "cat", h_state);
        readout_train(h_state, 1, learning_rate);  // gold_index=1

        // 例として "dog" で学習
        memset(h_state, 0, sizeof(h_state));
        trie_reservoir_forward(root, "dog", h_state);
        readout_train(h_state, 2, learning_rate);  // gold_index=2

        // 例として "help" で学習
        memset(h_state, 0, sizeof(h_state));
        trie_reservoir_forward(root, "help", h_state);
        readout_train(h_state, 3, learning_rate);  // gold_index=3

        // 適当に学習率を下げるなど
        if(epoch % 20 == 19) {
            learning_rate *= 0.9f;
        }
    }

    // 5. 学習結果の確認: "hello" を入力して予測を見る
    {
        float h_state[RESERVOIR_SIZE];
        memset(h_state, 0, sizeof(h_state));
        trie_reservoir_forward(root, "hello", h_state);

        float probs[OUT_DIM];
        readout_forward(h_state, probs);

        printf("Input: 'hello' -> Output Probs: ");
        for(int i = 0; i < OUT_DIM; i++) {
            printf("%.3f ", probs[i]);
        }
        printf("\n");
    }

    // 6. 後始末 (メモリ解放など)
    // 実際には Trie ノードの再帰解放が必要
    // サンプルでは省略

    if(reservoir_weights) {
        for(int l = 0; l < MAX_DEPTH; l++) {
            if(reservoir_weights[l]) free(reservoir_weights[l]);
        }
        free(reservoir_weights);
    }
    // TrieNode の解放(省略) ...

    return 0;
}
