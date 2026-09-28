# C 整数参考(P4.4)

独立实现,与 `reference/python/int_reference.py` 只共享两样东西:P4PKB01 包格式(定点合同)与 golden vectors;不共享任何代码。

## 构建

```
gcc -std=c11 -O2 -Wall -Wextra -o int_reference.exe int_reference.c
```

gcc 13.2.0(MinGW-w64)验证通过,无警告;代码保持可交叉编译子集(仅 libc stdio/stdlib/string,无 POSIX 扩展)。有符号右移依赖算术移位(gcc/MSVC 均保证),golden vectors 中的负数舍入向量用于在编译器变更时捕获差异。

## 运行

```
./int_reference.exe --pack <model_pack.bin> --input <input.bin> --output-dir <dir>
```

对每个算术节点(conv/requantize/add/maxpool)写出原始 int8 字节 `<node>.bin`。验收入口:

```
python -m tools.verify_golden_vectors
```

对 tests/golden 全部向量做三方对拍:冻结期望(Python 参考生成)↔ Python 重算 ↔ C 重算,逐节点哈希比较;任一不一致即失败(P4 验收条件)。
