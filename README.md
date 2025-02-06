# TAO_CT
## 使用方法
1. 安装依赖：
    ```
    pip install -r requirements.txt
    ```
2. 我们准备的已训练模型文件位置：
    ```
    MC_Net_main/PTH6/seg+cls_3000.pth
    classify/best/7pth/...
    ```
3. 数据存放：
    ```
    dataset/
    │
    ├── origin/
    │   ├── 1.nii.gz
    │   ├── 2.nii.gz
    │   └── ...
    │
    └── label/
        ├── 1.nii.gz
        ├── 2.nii.gz
        └── ...
    ```
4. 运行分割训练：
    ```
    MC_Net_main/train4.py
    ```
5. 运行多任务分类训练：
    ```
    classify/best/train1.py
    ```

