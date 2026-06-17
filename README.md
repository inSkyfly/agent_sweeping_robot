# 智扫通机器人智能客服

基于 LangChain + LangGraph + Chroma 的 RAG 智能客服 Agent，使用 Streamlit 提供 Web 聊天界面。支持知识库问答、工具调用（天气、用户数据等）及使用报告生成。

## 环境要求

- Python **3.10+**（推荐 **3.11** 或 **3.12**）
- 阿里云 DashScope API Key（通义千问大模型 + 向量嵌入）

## 快速开始

### 1. 克隆 / 进入项目目录

```powershell
cd "d:\AiAgent\AI大模型RAG与智能体开发_Agent项目"
```

### 2. 创建虚拟环境（推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

> 若 PowerShell 提示无法执行脚本，先运行：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. 安装依赖

```powershell
pip install -r requirements.txt
```

### 4. 配置 API Key

在 [阿里云百炼 / DashScope 控制台](https://help.aliyun.com/document_detail/611472.html) 申请 API Key，任选一种方式配置：

**方式一：`.env` 文件（推荐）**

```powershell
copy .env.example .env
```

编辑 `.env`，填入你的密钥：

```text
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**方式二：环境变量**

当前终端会话：

```powershell
$env:DASHSCOPE_API_KEY = "你的API密钥"
```

永久生效（用户级）：

```powershell
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "你的API密钥", "User")
```

设置永久变量后需重新打开终端。

### 5. 初始化知识库（可选）

首次运行会将 `data/` 目录下的 `.txt` / `.pdf` 文件写入 Chroma 向量库：

```powershell
python -m rag.vector_store
```

也可跳过此步，启动应用后 RAG 服务会自动加载知识库。

### 6. 启动应用

```powershell
streamlit run app.py
```

启动成功后，终端会显示类似：

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

浏览器将打开 **智扫通机器人智能客服** 聊天界面。

> **首次运行提示**：若出现 `Welcome to Streamlit!` 并询问邮箱，直接按 **Enter**（留空即可），服务会继续启动。项目已包含 `.streamlit/config.toml`，后续不会再弹出该提示。

## 命令行测试

不启动 Web 界面，直接测试 Agent：

```powershell
python -m agent.react_agent
```

测试 RAG 检索与总结：

```powershell
python -m rag.rag_service
```

## 项目结构

```
├── app.py                  # Streamlit 入口
├── agent/
│   ├── react_agent.py      # ReAct Agent 定义
│   └── tools/              # Agent 工具与中间件
├── rag/
│   ├── vector_store.py     # Chroma 向量库管理
│   └── rag_service.py      # RAG 检索 + 总结链
├── model/
│   └── factory.py          # 通义千问 / 嵌入模型工厂
├── config/
│   ├── rag.yml             # 模型名称配置
│   ├── chroma.yml          # 向量库与分片参数
│   ├── agent.yml           # Agent 外部数据路径
│   └── prompts.yml         # 提示词文件路径
├── prompts/                # 系统 / RAG / 报告提示词模板
├── data/                   # 知识库文档（txt / pdf）
├── data/external/          # 用户报告 CSV 数据
├── chroma_db/              # 向量库持久化目录（运行后生成）
├── logs/                   # 运行日志
└── requirements.txt        # Python 依赖
```

## 配置说明

| 文件 | 说明 |
|------|------|
| `config/rag.yml` | 对话模型 `qwen3-max`，嵌入模型 `text-embedding-v4` |
| `config/chroma.yml` | 向量库路径、检索数量、文本分片参数 |
| `config/agent.yml` | 外部 CSV 数据路径（报告生成） |
| `config/prompts.yml` | 系统提示词、RAG 提示词、报告提示词路径 |

## 常见问题

### Streamlit 只显示 Welcome 就退出了

这是首次运行的邮箱订阅提示，不是报错。在 `Email:` 后**直接按 Enter**（留空），等待出现 `Local URL: http://localhost:8501` 即表示成功。也可手动访问该地址。

### IP 定位城市不准确

公网 IP 定位本身存在误差（运营商机房、宽带归属地、VPN 等都会导致偏差）。可任选一种方式：

1. **侧边栏手动填写城市**（推荐）：启动后在页面左侧「所在城市」输入框填写，例如 `深圳`
2. **配置高德 Key**：在 `.env` 中设置 `AMAP_API_KEY`，国内定位更准
3. **对话中直接说明城市**：例如「我在杭州，今天适合扫地吗？」

### API 认证失败 / `Did not find dashscope_api_key`

说明未配置 `DASHSCOPE_API_KEY`。请复制 `.env.example` 为 `.env` 并填入密钥，或在终端执行 `$env:DASHSCOPE_API_KEY = "你的密钥"` 后重新运行 `streamlit run app.py`。

### `ModuleNotFoundError`

确认已激活虚拟环境，并执行 `pip install -r requirements.txt`。

### API 认证失败

检查 `DASHSCOPE_API_KEY` 是否正确，账户是否开通对应模型权限。

### Python 3.14 安装报错

部分依赖可能尚未完全支持 Python 3.14，建议改用 Python 3.11 或 3.12 创建虚拟环境。

### 向量库异常 / 需要重建索引

删除 `chroma_db/` 目录和 `md5.text` 文件，然后重新执行：

```powershell
python -m rag.vector_store
```

## 技术栈

- **LLM**：阿里云通义千问（DashScope）
- **Agent 框架**：LangChain + LangGraph
- **向量数据库**：Chroma
- **Web 界面**：Streamlit
- **IP 定位**：ip9.com.cn / 太平洋 IP 库（可选高德 Key，无需 Key 也可用手动指定城市）
- **天气数据**：Open-Meteo（免费，无需 Key）
