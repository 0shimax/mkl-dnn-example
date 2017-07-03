# 用語
- dst： 出力側（のピクセル値？）を意味する
- md： memory discriminator
- pd： primitive discriminator
- tz： tensor size

# 注意点
- 要素どうしの計算では`eltwise_forward`などを使用する

# 処理の流れ（forward）
- forwardとbackwordを別々に記述する必要がある
    - 同じようなコードを2回書かないといけない...
- backwordの記述でもサイズを再定義する必要がある

## エンジンの定義
- 例： `auto cpu_engine = engine(engine::cpu, 0);`

## 入出力
- `std::vector`で入出力データのvectorのメモリを確保

## 一層目
- `memory::dims`で各層の入出力、パラメタのサイズを定義
    - `memory::dims`はMKL-DNNに最適化するためのAPI
    - Conv.層の場合は
        - conv_src_tz：入力側shape
        - conv_weights_tz：weight shape
        - conv_bias_tz： bias shape
        - conv_dst_tz： 出力側 shape
        - conv_strides： stride shape({3, 3}とか)
- `std::vector`でweight, biasなどのパラメタを格納するメモリを確保
- パラメタの初期化
    - exampleではfor文で回して入れてる
- 層で保持するデータのメモリ確保
    - MKL-DNNの`memory`APIを使ってconv_user_src_memoryのように確保
    - ~~おそらくbackwordなどで使用するために確保しておく~~
    - backwordでも同じように定義しているため、作業用のメモリの確保なのかもしれない
- 各層のmemory descriptorの定義
    - user_memory_descriptorと同じような処理をもう一度書く
    - ただし、アウトプット側のdescriptorも必要
- primitive descriptorの定義
    - まず`convolution_forward::desc`などで、層のパラメタ設定などをする
    - `convolution_forward::primitive_desc`で設定を反映し、primitive descriptorのインスタンス作成
- shapeが合っていない場合はreorderする
    - 必要なんだろうか？謎い
- 出力側のメモリ確保
    - 例：`memory(conv_pd.dst_primitive_desc())`
- フォワードの定義（データが入ってくるまでは実行されない=遅延評価）
    - 例：convolution_forward(conv_pd, conv_src_memory, conv_weights_memory, conv_user_bias_memory, conv_dst_memory);

## 2層目
- フォワードの定義
    - 入力ソースの引数は前の層の出力メモリ（MKL-DNNの`memory`オブジェクト）
    - 例：eltwise_forward(relu_pd, **conv_dst_memory** , relu_dst_memory);

## ネットワーク定義
- `net_fwd.push_back(conv);`, `net_fwd.push_back(relu);`などで積み重ねていく
