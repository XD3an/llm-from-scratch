# LLM from scratch

## Usage

1. 安裝 requirements

    ```bash
    pip install -r requirements.txt
    ```

2. 訓練模型

    ```bash
    python src/train.py
    ```
    or 
    ```
    ./train.bat # for windows
    ```

3. 測試模型
    - 根據給定 prompt 生成後續
    
    ```bash
    python src/generation.py
    ```

## Reference

- [https://github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)