# LLM from scratch

## Introduction

- **Transformer-Decoder-only**： 實作 Transformer 的 Decoder-only 模型，用於根據給定 prompt 生成後續。


## Usage

### Transformer-Decoder-only

- 可修改 `config.json` 中的參數，調整模型訓練的參數。

1. 安裝 requirements

    ```bash
    pip install -r requirements.txt
    ```

2. 訓練模型

    ```bash
    python src/train.py
    ```
    or 
    ```bash
    ./train.bat # for windows
    ```

3. 測試模型
    - 根據給定 prompt 生成後續
    
    ```bash
    python src/generation.py
    ```

## Reference

- [https://github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)