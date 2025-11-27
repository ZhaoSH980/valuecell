# 配置层 (Config)

## 概述

配置层负责管理系统的所有配置信息，包括智能体配置、模型提供商配置、系统常量等。采用 YAML 文件存储配置，支持热加载和多环境配置。

## 目录结构

```
config/
├── __init__.py
├── constants.py    # 常量定义
├── loader.py       # 配置加载器
└── manager.py      # 配置管理器

configs/            # 配置文件目录（位于 python/ 目录下）
├── config.yaml     # 主配置文件
├── agents/         # 智能体配置
│   ├── news_agent.yaml
│   ├── research_agent.yaml
│   └── super_agent.yaml
├── agent_cards/    # Agent Card 配置
│   ├── grid_agent.json
│   ├── investment_research_agent.json
│   ├── news_agent.json
│   └── prompt_strategy_agent.json
├── providers/      # 模型提供商配置
│   ├── openai.yaml
│   ├── azure.yaml
│   ├── google.yaml
│   ├── deepseek.yaml
│   ├── openrouter.yaml
│   ├── siliconflow.yaml
│   └── openai-compatible.yaml
└── locales/        # 国际化文件
    ├── language_list.json
    ├── en-US.json
    ├── en-GB.json
    ├── zh-Hans.json
    └── zh-Hant.json
```

## 核心文件详解

### 1. constants.py - 常量定义

**位置**：`config/constants.py`

#### 配置路径常量

```python
# 默认配置目录
DEFAULT_CONFIG_DIR = "configs"

# 配置文件名
MAIN_CONFIG_FILE = "config.yaml"
AGENT_CARDS_DIR = "agent_cards"
AGENTS_DIR = "agents"
PROVIDERS_DIR = "providers"
LOCALES_DIR = "locales"

# 环境变量前缀
ENV_PREFIX = "VALUECELL_"
```

#### 默认值常量

```python
# 默认语言
DEFAULT_LANGUAGE = "en-US"

# 默认时区
DEFAULT_TIMEZONE = "UTC"

# 默认模型
DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_MODEL_NAME = "gpt-4"
```

### 2. loader.py - 配置加载器

**位置**：`config/loader.py`

#### ConfigLoader 类

**功能**：
- 加载 YAML 配置文件
- 加载 JSON 配置文件
- 解析环境变量
- 配置验证

**核心方法**：

```python
class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        初始化加载器。
        
        参数：
        - config_dir: 配置目录路径，默认为 ./configs
        """
        self.config_dir = config_dir or self._get_default_config_dir()
    
    def _get_default_config_dir(self) -> Path:
        """获取默认配置目录"""
        # 查找顺序：
        # 1. 环境变量 VALUECELL_CONFIG_DIR
        # 2. 项目根目录/configs
        # 3. 当前目录/configs
        env_dir = os.getenv("VALUECELL_CONFIG_DIR")
        if env_dir:
            return Path(env_dir)
        
        # 尝试找到项目根目录
        current = Path(__file__).resolve()
        for parent in current.parents:
            config_path = parent / "configs"
            if config_path.exists():
                return config_path
        
        return Path("./configs")
    
    def load_yaml(self, file_path: str | Path) -> Dict:
        """
        加载 YAML 文件。
        
        参数：
        - file_path: 文件路径（相对于 config_dir 或绝对路径）
        
        返回：
        - Dict: 解析后的配置字典
        """
        import yaml
        
        if not Path(file_path).is_absolute():
            file_path = self.config_dir / file_path
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 解析环境变量占位符
        return self._resolve_env_vars(config)
    
    def load_json(self, file_path: str | Path) -> Dict:
        """
        加载 JSON 文件。
        
        参数：
        - file_path: 文件路径
        
        返回：
        - Dict: 解析后的配置字典
        """
        import json
        
        if not Path(file_path).is_absolute():
            file_path = self.config_dir / file_path
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return self._resolve_env_vars(config)
    
    def _resolve_env_vars(self, config: Any) -> Any:
        """
        递归解析配置中的环境变量占位符。
        
        支持格式：
        - ${ENV_VAR}
        - ${ENV_VAR:default_value}
        """
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._resolve_env_var_string(config)
        else:
            return config
    
    def _resolve_env_var_string(self, value: str) -> str:
        """解析字符串中的环境变量"""
        import re
        
        # 匹配 ${VAR} 或 ${VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value or "")
        
        return re.sub(pattern, replace, value)
```

### 3. manager.py - 配置管理器

**位置**：`config/manager.py`

#### ConfigManager 类

**功能**：
- 统一的配置访问接口
- 配置缓存
- 配置热加载（预留）
- 默认值管理

**核心方法**：

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.loader = ConfigLoader(config_dir)
        self._cache: Dict[str, Any] = {}
        self._load_main_config()
    
    def _load_main_config(self):
        """加载主配置文件"""
        try:
            self.main_config = self.loader.load_yaml("config.yaml")
        except FileNotFoundError:
            logger.warning("Main config file not found, using defaults")
            self.main_config = {}
    
    def get_agent_config(self, agent_name: str) -> Dict:
        """
        获取智能体配置。
        
        参数：
        - agent_name: 智能体名称
        
        返回：
        - Dict: 智能体配置
        """
        cache_key = f"agent:{agent_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            config_path = f"agents/{agent_name}.yaml"
            config = self.loader.load_yaml(config_path)
            self._cache[cache_key] = config
            return config
        except FileNotFoundError:
            logger.warning(f"Agent config not found: {agent_name}")
            return self._get_default_agent_config(agent_name)
    
    def get_provider_config(self, provider_name: str) -> Dict:
        """
        获取模型提供商配置。
        
        参数：
        - provider_name: 提供商名称
        
        返回：
        - Dict: 提供商配置
        """
        cache_key = f"provider:{provider_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            config_path = f"providers/{provider_name}.yaml"
            config = self.loader.load_yaml(config_path)
            self._cache[cache_key] = config
            return config
        except FileNotFoundError:
            logger.warning(f"Provider config not found: {provider_name}")
            return {}
    
    def get_locale(self, language: str) -> Dict:
        """
        获取语言翻译。
        
        参数：
        - language: 语言代码（如 zh-Hans）
        
        返回：
        - Dict: 翻译字典
        """
        cache_key = f"locale:{language}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            config_path = f"locales/{language}.json"
            config = self.loader.load_json(config_path)
            self._cache[cache_key] = config
            return config
        except FileNotFoundError:
            logger.warning(f"Locale not found: {language}")
            return {}
    
    def get_all_languages(self) -> List[Dict]:
        """获取所有支持的语言列表"""
        try:
            return self.loader.load_json("locales/language_list.json")
        except FileNotFoundError:
            return [{"code": "en-US", "name": "English"}]
    
    def _get_default_agent_config(self, agent_name: str) -> Dict:
        """获取默认智能体配置"""
        return {
            "name": agent_name,
            "model": {
                "provider": DEFAULT_MODEL_PROVIDER,
                "name": DEFAULT_MODEL_NAME,
                "temperature": 0.7,
                "max_tokens": 4000
            }
        }
    
    def reload_config(self, config_type: str, name: str):
        """
        重新加载配置（热加载）。
        
        参数：
        - config_type: 配置类型（agent/provider/locale）
        - name: 配置名称
        """
        cache_key = f"{config_type}:{name}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # 重新加载
        if config_type == "agent":
            return self.get_agent_config(name)
        elif config_type == "provider":
            return self.get_provider_config(name)
        elif config_type == "locale":
            return self.get_locale(name)
    
    def clear_cache(self):
        """清空配置缓存"""
        self._cache.clear()
        self._load_main_config()
```

#### 单例模式

```python
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
```

## 配置文件格式

### 主配置文件 (config.yaml)

```yaml
# 应用配置
app:
  name: ValueCell
  version: 0.1.0
  environment: production  # production / development / test

# API 配置
api:
  host: ${API_HOST:0.0.0.0}
  port: ${API_PORT:8000}
  debug: ${API_DEBUG:false}
  cors_origins:
    - "*"

# 数据库配置
database:
  url: ${DATABASE_URL:sqlite+aiosqlite:///./valuecell.db}
  echo: false  # 是否打印 SQL

# 日志配置
logging:
  level: ${LOG_LEVEL:INFO}
  format: "{time} | {level} | {message}"
  rotation: "100 MB"
  retention: "30 days"

# 默认配置
defaults:
  language: en-US
  timezone: UTC
  currency: USD
```

### 智能体配置 (agents/*.yaml)

**示例**：`agents/research_agent.yaml`

```yaml
name: research_agent
display_name: Research Agent

# 模型配置
model:
  provider: openai
  name: gpt-4
  temperature: 0.7
  max_tokens: 4000
  stream: true

# 工具配置
tools:
  - fetch_periodic_sec_filings
  - fetch_event_sec_filings
  - fetch_ashare_filings
  - web_search

# 知识库配置
knowledge:
  enabled: true
  vector_db: lancedb
  index_path: ./data/knowledge/research
  search_limit: 5

# 上下文配置
context:
  add_datetime: true
  add_history: true
  num_history_runs: 3
  enable_session_summaries: true

# 自定义参数
custom:
  max_filing_pages: 10
  web_search_timeout: 30
```

### 模型提供商配置 (providers/*.yaml)

**示例**：`providers/openai.yaml`

```yaml
provider: openai
display_name: OpenAI

# API 配置
api:
  key_env: OPENAI_API_KEY
  base_url_env: OPENAI_BASE_URL
  default_base_url: https://api.openai.com/v1

# 支持的模型
models:
  - name: gpt-4
    max_tokens: 8192
    supports_vision: false
    supports_function_calling: true
  
  - name: gpt-4-turbo
    max_tokens: 128000
    supports_vision: true
    supports_function_calling: true
  
  - name: gpt-3.5-turbo
    max_tokens: 4096
    supports_vision: false
    supports_function_calling: true

# 默认参数
defaults:
  temperature: 0.7
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

### Agent Card 配置 (agent_cards/*.json)

**示例**：`agent_cards/research_agent.json`

```json
{
  "name": "research_agent",
  "display_name": "投资研究智能体",
  "description": "分析 SEC 文件、公司财报和市场数据，提供深度投资研究报告",
  "version": "1.0.0",
  "enabled": true,
  "capabilities": {
    "streaming": true,
    "push_notifications": false
  },
  "default_input_modes": ["text"],
  "default_output_modes": ["text", "markdown"],
  "metadata": {
    "category": "analysis",
    "tags": ["research", "SEC", "financial-analysis", "fundamental"],
    "author": "ValueCell Team",
    "icon": "📊",
    "color": "#3B82F6"
  }
}
```

### 国际化配置 (locales/*.json)

**示例**：`locales/zh-Hans.json`

```json
{
  "common": {
    "welcome": "欢迎使用 ValueCell",
    "loading": "加载中...",
    "error": "错误",
    "success": "成功"
  },
  "agents": {
    "research_agent": "研究智能体",
    "news_agent": "新闻智能体",
    "grid_agent": "网格交易智能体"
  },
  "errors": {
    "agent_not_found": "智能体未找到",
    "invalid_input": "无效输入",
    "network_error": "网络错误"
  },
  "strategy": {
    "start": "启动策略",
    "stop": "停止策略",
    "status": {
      "running": "运行中",
      "stopped": "已停止",
      "paused": "已暂停"
    }
  }
}
```

**语言列表**：`locales/language_list.json`

```json
[
  {
    "code": "en-US",
    "name": "English (US)",
    "native_name": "English"
  },
  {
    "code": "zh-Hans",
    "name": "Chinese (Simplified)",
    "native_name": "简体中文"
  },
  {
    "code": "zh-Hant",
    "name": "Chinese (Traditional)",
    "native_name": "繁體中文"
  },
  {
    "code": "ja-JP",
    "name": "Japanese",
    "native_name": "日本語"
  }
]
```

## 环境变量

### 优先级

1. 实际环境变量
2. `.env` 文件
3. 配置文件中的默认值

### 常用环境变量

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# 数据库
DATABASE_URL=sqlite+aiosqlite:///./valuecell.db

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Google
GOOGLE_API_KEY=...

# DeepSeek
DEEPSEEK_API_KEY=...

# OKX 交易
OKX_API_KEY=...
OKX_API_SECRET=...
OKX_API_PASSPHRASE=...
OKX_NETWORK=paper
OKX_ALLOW_LIVE_TRADING=false

# SEC EDGAR
SEC_EMAIL=your@email.com

# 日志
LOG_LEVEL=INFO
AGENT_DEBUG_MODE=false

# 配置目录
VALUECELL_CONFIG_DIR=/path/to/configs
```

## 使用示例

### 获取智能体配置

```python
from valuecell.config.manager import get_config_manager

manager = get_config_manager()
config = manager.get_agent_config("research_agent")

print(f"Model: {config['model']['name']}")
print(f"Tools: {config['tools']}")
```

### 获取翻译

```python
manager = get_config_manager()
translations = manager.get_locale("zh-Hans")

welcome_msg = translations["common"]["welcome"]
print(welcome_msg)  # "欢迎使用 ValueCell"
```

### 环境变量占位符

在 YAML 中使用：

```yaml
api:
  host: ${API_HOST:0.0.0.0}
  key: ${SECRET_KEY:default_key}
```

- `${API_HOST:0.0.0.0}`：使用环境变量 `API_HOST`，如果不存在则使用 `0.0.0.0`
- `${SECRET_KEY}`：使用环境变量 `SECRET_KEY`，如果不存在则为空字符串

## 配置验证

### 验证器类

```python
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """智能体配置验证模型"""
    name: str
    model: dict
    tools: List[str] = []
    knowledge: Optional[dict] = None

def validate_agent_config(config: Dict) -> AgentConfig:
    """验证智能体配置"""
    return AgentConfig(**config)
```

## 配置迁移

### 版本管理

```yaml
# 在配置文件顶部标记版本
_version: "1.0"
```

### 迁移脚本

```python
def migrate_config_v1_to_v2(old_config: Dict) -> Dict:
    """从 v1 配置迁移到 v2"""
    new_config = old_config.copy()
    # 执行迁移逻辑
    new_config["_version"] = "2.0"
    return new_config
```

## 最佳实践

### 1. 敏感信息
- 永远不要在配置文件中硬编码密钥
- 使用环境变量
- 使用 `.env` 文件（不提交到版本控制）

### 2. 默认值
- 提供合理的默认值
- 使用环境变量占位符的默认值语法

### 3. 配置组织
- 按功能分类配置文件
- 保持配置文件简洁
- 使用注释说明复杂配置

### 4. 版本控制
- 提交 `.env.example` 而非 `.env`
- 文档化所有配置项
- 标记配置版本

---

**相关文档**：
- [工具层](./06_utils.md)
- [适配器层](./04_adapters.md)
- [服务器层](./01_server.md)

