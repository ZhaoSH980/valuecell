# 工具层 (Utils)

## 概述

工具层提供系统中各模块共用的工具函数和辅助类，包括数据库连接、环境变量处理、国际化、时间处理、UUID 生成等。

## 目录结构

```
utils/
├── __init__.py
├── db.py               # 数据库工具
├── env.py              # 环境变量工具
├── i18n_utils.py       # 国际化工具
├── model.py            # 模型工具
├── path.py             # 路径工具
├── port.py             # 端口工具
├── ts.py               # 时间序列工具
├── user_profile_utils.py  # 用户配置工具
└── uuid.py             # UUID 生成工具
```

## 核心文件详解

### 1. uuid.py - UUID 生成工具

**位置**：`utils/uuid.py`

#### 功能
- 生成各类 ID
- 保证唯一性
- 支持自定义前缀

#### 核心函数

```python
import uuid
from datetime import datetime

def generate_uuid() -> str:
    """
    生成标准 UUID。
    
    返回：
    - str: UUID 字符串（无短横线）
    """
    return uuid.uuid4().hex

def generate_conversation_id() -> str:
    """
    生成对话 ID。
    
    格式：conv_{uuid}
    """
    return f"conv_{generate_uuid()}"

def generate_thread_id() -> str:
    """
    生成线程 ID。
    
    格式：thread_{uuid}
    """
    return f"thread_{generate_uuid()}"

def generate_task_id() -> str:
    """
    生成任务 ID。
    
    格式：task_{uuid}
    """
    return f"task_{generate_uuid()}"

def generate_item_id() -> str:
    """
    生成消息项 ID。
    
    格式：item_{uuid}
    """
    return f"item_{generate_uuid()}"

def generate_strategy_id() -> str:
    """
    生成策略 ID。
    
    格式：strategy_{uuid}
    """
    return f"strategy_{generate_uuid()}"

def generate_timestamp_id(prefix: str = "") -> str:
    """
    生成基于时间戳的 ID。
    
    参数：
    - prefix: ID 前缀
    
    返回：
    - str: {prefix}{timestamp}_{random}
    
    示例：task_20250126_abc123
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_suffix}"
    return f"{timestamp}_{random_suffix}"
```

### 2. env.py - 环境变量工具

**位置**：`utils/env.py`

#### 功能
- 获取环境变量
- 类型转换
- 默认值处理
- 系统环境路径管理

#### 核心函数

```python
import os
from pathlib import Path
from typing import Optional

def get_env(key: str, default: str = "") -> str:
    """
    获取环境变量（字符串）。
    
    参数：
    - key: 环境变量名
    - default: 默认值
    
    返回：
    - str: 环境变量值
    """
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    获取布尔类型环境变量。
    
    true 值：true, 1, yes, on
    false 值：false, 0, no, off
    
    参数：
    - key: 环境变量名
    - default: 默认值
    
    返回：
    - bool: 布尔值
    """
    value = os.getenv(key, "").lower()
    if not value:
        return default
    return value in ("true", "1", "yes", "on")

def get_env_int(key: str, default: int = 0) -> int:
    """
    获取整数类型环境变量。
    
    参数：
    - key: 环境变量名
    - default: 默认值
    
    返回：
    - int: 整数值
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """获取浮点数类型环境变量"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

def agent_debug_mode_enabled() -> bool:
    """
    检查智能体调试模式是否启用。
    
    环境变量：AGENT_DEBUG_MODE
    
    返回：
    - bool: 是否启用
    """
    return get_env_bool("AGENT_DEBUG_MODE", False)

def get_system_env_dir() -> Path:
    """
    获取系统环境配置目录。
    
    位置：~/.valuecell/
    
    返回：
    - Path: 配置目录路径
    """
    return Path.home() / ".valuecell"

def get_system_env_path() -> Path:
    """
    获取系统环境文件路径。
    
    位置：~/.valuecell/.env
    
    返回：
    - Path: .env 文件路径
    """
    return get_system_env_dir() / ".env"

def ensure_system_env_dir():
    """确保系统环境目录存在"""
    env_dir = get_system_env_dir()
    env_dir.mkdir(parents=True, exist_ok=True)
```

### 3. i18n_utils.py - 国际化工具

**位置**：`utils/i18n_utils.py`

#### 功能
- 获取当前语言
- 获取当前时区
- 翻译文本

#### 核心函数

```python
from contextvars import ContextVar
from typing import Optional

# 上下文变量（线程安全）
_current_language: ContextVar[Optional[str]] = ContextVar(
    "current_language", default=None
)
_current_timezone: ContextVar[Optional[str]] = ContextVar(
    "current_timezone", default=None
)

def get_current_language() -> str:
    """
    获取当前语言。
    
    优先级：
    1. 上下文变量
    2. 环境变量 LANGUAGE
    3. 默认值 en-US
    
    返回：
    - str: 语言代码
    """
    lang = _current_language.get()
    if lang:
        return lang
    
    from valuecell.utils.env import get_env
    from valuecell.config.constants import DEFAULT_LANGUAGE
    
    return get_env("LANGUAGE", DEFAULT_LANGUAGE)

def set_current_language(language: str):
    """设置当前语言（在当前上下文中）"""
    _current_language.set(language)

def get_current_timezone() -> str:
    """
    获取当前时区。
    
    优先级：
    1. 上下文变量
    2. 环境变量 TIMEZONE
    3. 默认值 UTC
    
    返回：
    - str: 时区字符串
    """
    tz = _current_timezone.get()
    if tz:
        return tz
    
    from valuecell.utils.env import get_env
    from valuecell.config.constants import DEFAULT_TIMEZONE
    
    return get_env("TIMEZONE", DEFAULT_TIMEZONE)

def set_current_timezone(timezone: str):
    """设置当前时区"""
    _current_timezone.set(timezone)

def translate(key: str, language: Optional[str] = None, **kwargs) -> str:
    """
    翻译文本。
    
    参数：
    - key: 翻译键（如 "common.welcome"）
    - language: 语言代码（可选，默认使用当前语言）
    - **kwargs: 格式化参数
    
    返回：
    - str: 翻译后的文本
    
    示例：
    translate("common.welcome")
    translate("errors.user_not_found", username="Alice")
    """
    from valuecell.config.manager import get_config_manager
    
    if language is None:
        language = get_current_language()
    
    manager = get_config_manager()
    translations = manager.get_locale(language)
    
    # 支持嵌套键：common.welcome
    keys = key.split(".")
    value = translations
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            value = None
            break
    
    if value is None:
        return key  # 返回键本身作为降级
    
    # 格式化
    if kwargs:
        try:
            return value.format(**kwargs)
        except KeyError:
            return value
    
    return value
```

### 4. ts.py - 时间序列工具

**位置**：`utils/ts.py`

#### 功能
- 时间格式转换
- 时区处理
- 时间计算

#### 核心函数

```python
from datetime import datetime, timedelta, timezone
import pytz

def now_utc() -> datetime:
    """
    获取当前 UTC 时间（带时区信息）。
    
    返回：
    - datetime: UTC 时间
    """
    return datetime.now(timezone.utc)

def to_utc(dt: datetime) -> datetime:
    """
    转换为 UTC 时间。
    
    参数：
    - dt: 本地时间
    
    返回：
    - datetime: UTC 时间
    """
    if dt.tzinfo is None:
        # 假设是 UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def to_timezone(dt: datetime, tz_str: str) -> datetime:
    """
    转换到指定时区。
    
    参数：
    - dt: 时间
    - tz_str: 时区字符串（如 'Asia/Shanghai'）
    
    返回：
    - datetime: 指定时区的时间
    """
    tz = pytz.timezone(tz_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)

def format_datetime(
    dt: datetime,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    tz_str: Optional[str] = None
) -> str:
    """
    格式化时间。
    
    参数：
    - dt: 时间
    - format_str: 格式字符串
    - tz_str: 目标时区
    
    返回：
    - str: 格式化后的字符串
    """
    if tz_str:
        dt = to_timezone(dt, tz_str)
    return dt.strftime(format_str)

def parse_datetime(
    dt_str: str,
    format_str: str = "%Y-%m-%d %H:%M:%S"
) -> datetime:
    """
    解析时间字符串。
    
    参数：
    - dt_str: 时间字符串
    - format_str: 格式字符串
    
    返回：
    - datetime: 时间对象
    """
    return datetime.strptime(dt_str, format_str)

def calculate_duration(start: datetime, end: datetime) -> timedelta:
    """
    计算时间间隔。
    
    参数：
    - start: 开始时间
    - end: 结束时间
    
    返回：
    - timedelta: 时间间隔
    """
    return end - start

def add_time(dt: datetime, **kwargs) -> datetime:
    """
    增加时间。
    
    参数：
    - dt: 基准时间
    - **kwargs: days, hours, minutes, seconds 等
    
    返回：
    - datetime: 新时间
    
    示例：
    add_time(now, days=1, hours=2)
    """
    delta = timedelta(**kwargs)
    return dt + delta
```

### 5. model.py - 模型工具

**位置**：`utils/model.py`

#### 功能
- 获取模型实例
- 模型配置管理

#### 核心函数

```python
from typing import Any, Optional

def get_model_for_agent(agent_name: str) -> Any:
    """
    为智能体获取模型实例。
    
    参数：
    - agent_name: 智能体名称
    
    返回：
    - Model: 模型实例（如 OpenAIChat）
    """
    from valuecell.adapters.models import create_model_for_agent
    return create_model_for_agent(agent_name)

def get_model(
    provider: str = "openai",
    name: str = "gpt-4",
    **kwargs
) -> Any:
    """
    直接获取模型实例。
    
    参数：
    - provider: 提供商
    - name: 模型名称
    - **kwargs: 额外参数
    
    返回：
    - Model: 模型实例
    """
    from valuecell.adapters.models.factory import create_model
    return create_model(provider, name, **kwargs)
```

### 6. path.py - 路径工具

**位置**：`utils/path.py`

#### 功能
- 获取项目路径
- 获取配置路径
- 获取数据路径

#### 核心函数

```python
from pathlib import Path

def get_project_root() -> Path:
    """
    获取项目根目录。
    
    返回：
    - Path: 项目根目录路径
    """
    # 从当前文件向上查找，直到找到包含 pyproject.toml 的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()

def get_config_dir() -> Path:
    """
    获取配置目录。
    
    返回：
    - Path: 配置目录路径
    """
    return get_project_root() / "configs"

def get_agent_card_path() -> Path:
    """
    获取 Agent Card 配置目录。
    
    返回：
    - Path: Agent Card 目录路径
    """
    return get_config_dir() / "agent_cards"

def get_data_dir() -> Path:
    """
    获取数据目录。
    
    返回：
    - Path: 数据目录路径
    """
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_logs_dir() -> Path:
    """
    获取日志目录。
    
    返回：
    - Path: 日志目录路径
    """
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def get_cache_dir() -> Path:
    """
    获取缓存目录。
    
    返回：
    - Path: 缓存目录路径
    """
    cache_dir = get_project_root() / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir
```

### 7. user_profile_utils.py - 用户配置工具

**位置**：`utils/user_profile_utils.py`

#### 功能
- 获取用户配置元数据
- 用户上下文管理

#### 核心函数

```python
from contextvars import ContextVar
from typing import Dict, Optional

# 用户配置上下文
_user_profile: ContextVar[Optional[Dict]] = ContextVar(
    "user_profile", default=None
)

def set_user_profile_metadata(profile: Dict):
    """
    设置用户配置到上下文。
    
    参数：
    - profile: 用户配置字典
    """
    _user_profile.set(profile)

def get_user_profile_metadata() -> Optional[Dict]:
    """
    获取用户配置。
    
    返回：
    - Dict: 用户配置字典
    """
    return _user_profile.get()

def get_user_preference(key: str, default: Any = None) -> Any:
    """
    获取用户偏好设置。
    
    参数：
    - key: 偏好键
    - default: 默认值
    
    返回：
    - Any: 偏好值
    """
    profile = get_user_profile_metadata()
    if profile is None:
        return default
    
    preferences = profile.get("preferences", {})
    return preferences.get(key, default)
```

### 8. port.py - 端口工具

**位置**：`utils/port.py`

#### 功能
- 检查端口可用性
- 查找可用端口

#### 核心函数

```python
import socket

def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """
    检查端口是否可用。
    
    参数：
    - port: 端口号
    - host: 主机地址
    
    返回：
    - bool: 是否可用
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(
    start_port: int = 8000,
    end_port: int = 9000,
    host: str = "0.0.0.0"
) -> Optional[int]:
    """
    查找可用端口。
    
    参数：
    - start_port: 起始端口
    - end_port: 结束端口
    - host: 主机地址
    
    返回：
    - int: 可用端口号，如果未找到则返回 None
    """
    for port in range(start_port, end_port + 1):
        if is_port_available(port, host):
            return port
    return None
```

### 9. db.py - 数据库工具

**位置**：`utils/db.py`

#### 功能
- 数据库URL解析
- 连接池管理

#### 核心函数

```python
from typing import Tuple

def parse_database_url(url: str) -> Tuple[str, str]:
    """
    解析数据库 URL。
    
    参数：
    - url: 数据库 URL
    
    返回：
    - Tuple[str, str]: (驱动, 路径)
    
    示例：
    parse_database_url("sqlite+aiosqlite:///./app.db")
    -> ("sqlite+aiosqlite", "./app.db")
    """
    if ":///" in url:
        driver, path = url.split(":///", 1)
        return (driver, path)
    elif "://" in url:
        driver, rest = url.split("://", 1)
        return (driver, rest)
    return ("", url)

def get_database_type(url: str) -> str:
    """
    获取数据库类型。
    
    参数：
    - url: 数据库 URL
    
    返回：
    - str: 数据库类型（sqlite, mysql, postgresql 等）
    """
    driver, _ = parse_database_url(url)
    return driver.split("+")[0] if "+" in driver else driver
```

## 使用示例

### UUID 生成

```python
from valuecell.utils.uuid import generate_conversation_id, generate_task_id

conv_id = generate_conversation_id()  # "conv_abc123..."
task_id = generate_task_id()  # "task_def456..."
```

### 环境变量

```python
from valuecell.utils.env import get_env_bool, get_env_int

debug = get_env_bool("DEBUG", False)
port = get_env_int("PORT", 8000)
```

### 国际化

```python
from valuecell.utils.i18n_utils import translate, set_current_language

set_current_language("zh-Hans")
msg = translate("common.welcome")  # "欢迎使用 ValueCell"
```

### 时间处理

```python
from valuecell.utils.ts import now_utc, to_timezone, add_time

now = now_utc()
shanghai_time = to_timezone(now, "Asia/Shanghai")
tomorrow = add_time(now, days=1)
```

### 路径获取

```python
from valuecell.utils.path import get_config_dir, get_logs_dir

config_dir = get_config_dir()
logs_dir = get_logs_dir()
```

## 最佳实践

### 1. 上下文变量
- 使用 `contextvars` 保证线程安全
- 适用于语言、时区、用户配置等

### 2. 默认值
- 所有工具函数提供合理默认值
- 优雅降级

### 3. 错误处理
- 工具函数捕获异常
- 返回默认值而非抛出异常

### 4. 类型提示
- 所有函数提供类型提示
- 提高代码可读性和 IDE 支持

---

**相关文档**：
- [配置层](./05_config.md)
- [数据库层](./07_database.md)
- [核心层](./02_core.md)

