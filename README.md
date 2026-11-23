# Census Archive Service

四叉树普查档案的最小读取取数服务。FastAPI + SQLite，`docker compose up` 即可启动。

## 启动

```bash
docker compose up --build
# 文档: http://localhost:8000/docs
```

数据卷 `census-data` 挂在 `/data/archive.sqlite3`；可用 `CENSUS_PORT` 改端口。

本地开发：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -q
```

## 数据模型

网格逐层四分：根格号为空串 `""`，子格在父格后追加一位 `0`–`3`，最多 12 位。
例如 `""` → `"3"` → `"31"` → `"310"`。格号只含 `0`–`3`，长度即层级。

每个统计期只有一批档案。提交整批在**单个事务**内完成（`BEGIN IMMEDIATE`）：

- 不带 `replace` 提交已存在的统计期 → `409`，原批不动；
- 带 `replace=true` → 删旧行 + 写新行在同一事务，失败自动回滚，原批不动；
- 校验失败（重复格号、非法数字、>12 位、负数人数）→ `422`，原批不动。

每次成功交存（初次、换交、格号更正）都会生成新的**批号** `batch_id`。
`GET /archive/{period}` 与交存响应都会返回当前批号；档案详情可据此核对。
旧版数据库（`batches` 表无 `batch_id` 列）在首次连接时自动就地升级，各旧批
获得独立批号，可直接使用，无需单独迁移。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/archive` | 整批交存/换交 |
| `GET` | `/archive/{period}` | 批次元数据（含当前批号） |
| `POST` | `/archive/{period}/correct` | 格号更正（同层格号循环换位） |
| `POST` | `/census/query` | 求覆盖并集人数 + 最简算式 |
| `GET` | `/healthz` | 健康检查 |

`/correction`、`/archive/{period}/correction` 也可用；请求体里给了 `period`
时也可省略路径中的统计期。

### 交存

```bash
curl -X POST localhost:8000/archive -H 'content-type: application/json' -d '{
  "period": "2024",
  "cells": [{"cell": "0", "population": 120}, {"cell": "01", "population": 45}]
}'
```

换交同一统计期时加 `"replace": true`。

### 格号更正

几个片区填错了普查格号时，可在**同一层**的 2～8 个格号之间做一一换位
（可循环）。一条档案 `旧格号 + 后缀` 更正为 `新格号 + 后缀`，人数随档案移动、
后缀原样保留：

```bash
# 读取当前批号（档案详情与交存响应里也有）
SEEN=$(curl -s localhost:8000/archive/2024 | python3 -c 'import sys,json;print(json.load(sys.stdin)["batch_id"])')

# 显式置换 {"旧": "新"}，也可用 "swaps" / "from_cells"+"to_cells" /
# 有序 "cells"（按顺序循环）
curl -X POST localhost:8000/archive/2024/correct -H 'content-type: application/json' -d "{
  \"seen_batch_id\": \"$SEEN\",
  \"permutation\": {\"01\": \"02\", \"02\": \"01\"}
}"
```

- 请求须提交**统计期**与**所见批号** `seen_batch_id`。入库时若当前批号已与所见
  批号不同（期间发生过换交或另一次更正）→ `409`，拒绝更正，**当前批继续服务**。
- 已存上级格的覆盖来源变了（跨父格换位），按换位后落入该格的原区域，从更正前
  的档案精确重算。所需片区人数若未单独交存，由求解器在更正前的档案上精确导出
  （只展开已交存格号及其前缀，绝不枚举最细格）。
- 同一父格下的片区换位，即使取不到各片区单独人数也能办理：该父格的覆盖来源不
  变，人数保持原值。
- 缺档时报 `missing_cells`（及具体缺档区域 `missing_regions`），重算为负数时报
  `negative_cells`，均指向相应**上级格**；这些情况下原批不动。
- 更正成功生成**新批号**，格号、上级人数和新批号在同一事务内一同生效，`/census/query`
  随即按新批取数。响应里 `moves` 列出档案移动，`recomputed` 列出被重算的上级格。
- 更正失败时继续使用当前批；失败不会产生新批号。

### 取数

```bash
curl -X POST localhost:8000/census/query -H 'content-type: application/json' -d '{
  "period": "2024",
  "cells": ["00", "02", "03"]
}'
```

```json
{
  "period": "2024",
  "cells": ["00", "02", "03"],
  "expression": "pop(\"0\") - pop(\"01\")",
  "terms": [
    {"cell": "0", "coefficient": 1},
    {"cell": "01", "coefficient": -1}
  ],
  "retrieved_cells": 2,
  "total_population": 75
}
```

- 查询是一组格号，结果按**覆盖并集**计算，重叠格不会重复计数。
- 算式系数可正可负，且只对目录中存在的格取数；返回的 `terms` 就是需要读取的格，
  `retrieved_cells` 为其个数。
- 算式**与人口分布无关**：对任意把人数分配到最细格的方案，算式恒等于目标区域人数。
- 目录与人数读自同一 SQLite 读事务（`snapshot`），并发换交也不会混批。
- 无解（请求区域没有可覆盖的档案）时返回 `422` 并列出 `missing_blocks`，提示缺档。

## 求解器

`app/solver.py` 在格号的**前缀树**上做两状态 DP，只在交存的格号与查询格号及其前缀处展开，
绝不枚举 `4^12` 个细格，也不需要读取任何人数即可确定算式。

设 `X(v)` 为根到 `v` 路径上的系数前缀和，则格 `v` 的系数为 `X(v) - X(parent)`，
细格约束为：目标区域内 `X=1`，区域外 `X=0`。任意整数标号钳到 `{0,1}` 不会增加系数非零的格，
故最优解的系数都在 `{-1,0,1}`，DP 每个节点只需记"进入标号"这一位状态，代价为非零系数个数。
无档案可用的子树会被 O(1) 折叠，并作为缺档块报告。

格号更正复用同一个求解器来核算上级格：需要某个片区在**更正前**的精确人数时，
对更正前的目录调用 `analyze` 得到表达式再取值，同样不展开最细格。
