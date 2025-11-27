# 服务器层 (Server)

## 概述

服务器层是 ValueCell 后端的 HTTP 接口层，基于 FastAPI 框架构建，负责：
- 接收和验证 HTTP 请求
- 路由到相应的业务服务
- 数据库持久化
- 返回标准化的响应

## 目录结构

```
server/
├── api/                    # API 接口层
│   ├── app.py             # FastAPI 应用工厂
│   ├── exceptions.py      # 异常处理器
│   ├── routers/           # API 路由模块
│   │   ├── agent.py       # 智能体相关 API
│   │   ├── agent_stream.py # 智能体流式 API
│   │   ├── conversation.py # 对话管理 API
│   │   ├── i18n.py        # 国际化 API
│   │   ├── models.py      # AI 模型配置 API
│   │   ├── strategy_api.py # 策略 API（聚合）
│   │   ├── strategy.py    # 策略管理 API
│   │   ├── strategy_agent.py # 策略智能体 API
│   │   ├── strategy_prompts.py # 策略提示 API
│   │   ├── system.py      # 系统信息 API
│   │   ├── task.py        # 任务管理 API
│   │   ├── trading.py     # 交易执行 API
│   │   ├── user_profile.py # 用户配置 API
│   │   └── watchlist.py   # 观察列表 API
│   └── schemas/           # Pydantic Schema 定义
│       ├── agent.py       # 智能体 Schema
│       ├── base.py        # 基础响应 Schema
│       ├── conversation.py # 对话 Schema
│       ├── i18n.py        # 国际化 Schema
│       ├── models.py      # 模型配置 Schema
│       ├── strategy.py    # 策略 Schema
│       ├── system.py      # 系统信息 Schema
│       ├── task.py        # 任务 Schema
│       ├── user_profile.py # 用户配置 Schema
│       └── watchlist.py   # 观察列表 Schema
├── config/                # 服务器配置
│   ├── i18n.py           # 国际化配置
│   └── settings.py       # 应用设置
├── db/                    # 数据库层
│   ├── connection.py     # 数据库连接
│   ├── init_db.py        # 数据库初始化
│   ├── models/           # SQLAlchemy 模型
│   │   ├── base.py       # 基础模型类
│   │   ├── agent.py      # 智能体模型
│   │   ├── asset.py      # 资产模型
│   │   ├── strategy.py   # 策略模型
│   │   ├── strategy_detail.py # 策略详情
│   │   ├── strategy_holding.py # 持仓记录
│   │   ├── strategy_portfolio.py # 投资组合
│   │   ├── strategy_instruction.py # 交易指令
│   │   ├── strategy_compose_cycle.py # 组合周期
│   │   ├── strategy_prompt.py # 策略提示
│   │   ├── user_profile.py # 用户配置
│   │   └── watchlist.py  # 观察列表
│   └── repositories/     # 仓储模式实现
│       ├── agent_repository.py
│       ├── strategy_repository.py
│       ├── user_profile_repository.py
│       └── watchlist_repository.py
├── services/             # 业务服务层
│   ├── agent_service.py  # 智能体服务
│   ├── agent_stream_service.py # 流式服务
│   ├── conversation_service.py # 对话服务
│   ├── i18n_service.py   # 国际化服务
│   ├── strategy_service.py # 策略服务
│   ├── strategy_persistence.py # 策略持久化
│   ├── strategy_autoresume.py # 策略自动恢复
│   ├── task_service.py   # 任务服务
│   ├── user_profile_service.py # 用户服务
│   └── assets/          # 资产服务
│       └── asset_service.py
└── main.py              # 应用入口
```

## 核心文件详解

### 1. main.py - 应用入口

**位置**：`server/main.py`

**功能**：
- 应用程序主入口
- 设置 UTF-8 编码
- 启动 Uvicorn 服务器

**关键代码**：
```python
def main():
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
    )
```

### 2. api/app.py - FastAPI 应用工厂

**位置**：`server/api/app.py`

**功能**：
- 创建和配置 FastAPI 应用实例
- 初始化数据库
- 配置数据适配器
- 注册中间件和异常处理器
- 注册所有 API 路由

**关键组件**：

#### 应用生命周期管理
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    - 初始化数据库表
    - 配置 Yahoo Finance 适配器
    - 配置 AKShare 适配器
    
    yield
    
    # 关闭时
    - 清理资源
```

#### 中间件配置
- **CORS 中间件**：配置跨域访问策略
- 支持所有 HTTP 方法和头部

#### 异常处理器
- `APIException`：自定义 API 异常
- `RequestValidationError`：请求验证异常
- `Exception`：通用异常处理

#### 路由注册（按功能分组）
- **基础路由**：`/`, `/api/v1/healthz`
- **国际化**：`/api/v1/i18n/*`
- **系统信息**：`/api/v1/system/*`
- **模型配置**：`/api/v1/models/*`
- **观察列表**：`/api/v1/watchlist/*`
- **对话管理**：`/api/v1/conversations/*`
- **用户配置**：`/api/v1/user-profile/*`
- **智能体流**：`/api/v1/agent-stream/*`
- **策略 API**：`/api/v1/strategies/*`
- **智能体**：`/api/v1/agents/*`
- **任务**：`/api/v1/tasks/*`
- **交易**：`/api/v1/trading/*`（可选）

### 3. api/exceptions.py - 异常处理

**位置**：`server/api/exceptions.py`

**功能**：统一的异常定义和处理

**关键类**：
- `APIException`：基础 API 异常类
  - `status_code`: HTTP 状态码
  - `code`: 业务错误码
  - `message`: 错误消息
  - `details`: 详细信息

**异常处理器**：
- `api_exception_handler`：处理自定义 API 异常
- `validation_exception_handler`：处理请求验证错误
- `general_exception_handler`：处理未捕获的异常

### 4. api/schemas/base.py - 基础响应模型

**位置**：`server/api/schemas/base.py`

**功能**：定义标准化的 API 响应格式

**核心 Schema**：

```python
class SuccessResponse(BaseModel, Generic[T]):
    """统一成功响应格式"""
    code: int = 200
    msg: str = "Success"
    data: Optional[T] = None
    
class ErrorResponse(BaseModel):
    """统一错误响应格式"""
    code: int
    msg: str
    details: Optional[Any] = None
```

## API 路由详解

### 1. agent.py - 智能体管理

**端点**：
- `GET /api/v1/agents` - 获取所有智能体列表
- `GET /api/v1/agents/{agent_id}` - 获取智能体详情
- `POST /api/v1/agents` - 创建智能体（预留）
- `PUT /api/v1/agents/{agent_id}` - 更新智能体
- `DELETE /api/v1/agents/{agent_id}` - 删除智能体

**功能**：
- 智能体元数据管理
- Agent Card 信息查询
- 智能体能力展示

### 2. agent_stream.py - 智能体流式通信

**端点**：
- `POST /api/v1/agent-stream` - 流式处理用户输入

**功能**：
- 接收用户输入
- 通过 AgentOrchestrator 处理
- 流式返回响应（SSE）
- 支持 Human-in-the-Loop

**请求格式**：
```json
{
  "user_id": "user123",
  "message": "分析 AAPL 股票",
  "conversation_id": "conv456",
  "metadata": {
    "timezone": "Asia/Shanghai",
    "language": "zh-Hans"
  }
}
```

**响应格式**：Server-Sent Events (SSE)
```
data: {"event": "message", "content": "正在分析..."}
data: {"event": "tool_call", "tool": "get_stock_price"}
data: {"event": "final", "content": "分析完成"}
```

### 3. conversation.py - 对话管理

**端点**：
- `GET /api/v1/conversations` - 获取对话列表
- `GET /api/v1/conversations/{conversation_id}` - 获取对话详情
- `GET /api/v1/conversations/{conversation_id}/items` - 获取对话消息
- `POST /api/v1/conversations` - 创建新对话
- `DELETE /api/v1/conversations/{conversation_id}` - 删除对话

**功能**：
- 对话历史管理
- 消息检索
- 对话状态跟踪

### 4. strategy_api.py - 策略聚合 API

**端点**：
- `GET /api/v1/strategies` - 获取策略列表
- `GET /api/v1/strategies/{strategy_id}` - 获取策略详情
- `POST /api/v1/strategies` - 创建策略（通过智能体）
- `PUT /api/v1/strategies/{strategy_id}` - 更新策略
- `DELETE /api/v1/strategies/{strategy_id}` - 删除策略
- `POST /api/v1/strategies/{strategy_id}/start` - 启动策略
- `POST /api/v1/strategies/{strategy_id}/stop` - 停止策略
- `GET /api/v1/strategies/{strategy_id}/holdings` - 获取持仓
- `GET /api/v1/strategies/{strategy_id}/performance` - 获取绩效

**功能**：
- 策略全生命周期管理
- 策略执行控制
- 持仓和绩效监控
- 与策略智能体集成

### 5. watchlist.py - 观察列表

**端点**：
- `GET /api/v1/watchlist` - 获取观察列表
- `POST /api/v1/watchlist` - 添加资产到观察列表
- `DELETE /api/v1/watchlist/{asset_id}` - 从观察列表移除
- `GET /api/v1/watchlist/prices` - 批量获取价格

**功能**：
- 用户关注资产管理
- 实时价格监控

### 6. models.py - AI 模型配置

**端点**：
- `GET /api/v1/models/providers` - 获取支持的模型提供商
- `GET /api/v1/models/providers/{provider}/models` - 获取提供商的模型列表
- `POST /api/v1/models/test` - 测试模型连接

**功能**：
- 模型提供商管理（OpenAI、Azure、DeepSeek 等）
- 模型配置和测试

### 7. system.py - 系统信息

**端点**：
- `GET /api/v1/system/info` - 获取系统信息
- `GET /api/v1/system/health` - 健康检查
- `GET /api/v1/system/version` - 获取版本信息

**功能**：
- 系统状态监控
- 版本管理

### 8. task.py - 任务管理

**端点**：
- `GET /api/v1/tasks` - 获取任务列表
- `GET /api/v1/tasks/{task_id}` - 获取任务详情
- `POST /api/v1/tasks/{task_id}/cancel` - 取消任务
- `POST /api/v1/tasks/{task_id}/resume` - 恢复任务

**功能**：
- 异步任务管理
- 定时任务调度
- 任务状态追踪

### 9. i18n.py - 国际化

**端点**：
- `GET /api/v1/i18n/languages` - 获取支持的语言列表
- `GET /api/v1/i18n/translations/{language}` - 获取指定语言的翻译

**功能**：
- 多语言支持
- 动态翻译加载

### 10. user_profile.py - 用户配置

**端点**：
- `GET /api/v1/user-profile` - 获取用户配置
- `PUT /api/v1/user-profile` - 更新用户配置

**功能**：
- 用户偏好设置
- 语言、时区配置

### 11. trading.py - 交易执行（可选）

**端点**：
- `POST /api/v1/trading/order` - 提交订单
- `GET /api/v1/trading/orders` - 获取订单列表
- `GET /api/v1/trading/positions` - 获取持仓

**功能**：
- 实盘/模拟交易
- 订单管理
- 持仓查询

## 数据库层详解

### 1. connection.py - 数据库连接

**功能**：
- 创建异步数据库引擎
- 会话管理
- 连接池配置

**关键函数**：
```python
def get_engine() -> AsyncEngine
def get_session() -> AsyncSession
async def init_db()
```

### 2. init_db.py - 数据库初始化

**功能**：
- 创建所有表
- 初始化基础数据
- 数据库迁移（预留）

### 3. models/ - SQLAlchemy 模型

#### base.py
- `Base`：所有模型的基类
- 通用字段：`id`, `created_at`, `updated_at`
- 软删除支持：`is_deleted`

#### agent.py - 智能体模型
```python
class Agent(Base):
    __tablename__ = "agents"
    
    name: str               # 智能体名称
    display_name: str       # 显示名称
    description: str        # 描述
    capabilities: JSON      # 能力配置
    config: JSON            # 配置信息
    enabled: bool           # 是否启用
```

#### strategy.py - 策略模型
```python
class Strategy(Base):
    __tablename__ = "strategies"
    
    user_id: str           # 用户 ID
    name: str              # 策略名称
    agent_name: str        # 关联智能体
    config: JSON           # 策略配置
    status: str            # 状态：draft/active/paused/stopped
    start_time: datetime   # 开始时间
    end_time: datetime     # 结束时间
```

#### strategy_holding.py - 持仓记录
```python
class StrategyHolding(Base):
    __tablename__ = "strategy_holdings"
    
    strategy_id: str       # 策略 ID
    symbol: str            # 交易对
    quantity: Decimal      # 数量
    avg_price: Decimal     # 平均价格
    current_price: Decimal # 当前价格
    unrealized_pnl: Decimal # 未实现盈亏
```

#### strategy_portfolio.py - 投资组合
```python
class StrategyPortfolio(Base):
    __tablename__ = "strategy_portfolios"
    
    strategy_id: str       # 策略 ID
    total_value: Decimal   # 总价值
    cash: Decimal          # 现金
    invested: Decimal      # 已投资
    pnl: Decimal          # 盈亏
```

#### strategy_instruction.py - 交易指令
```python
class StrategyInstruction(Base):
    __tablename__ = "strategy_instructions"
    
    strategy_id: str       # 策略 ID
    symbol: str            # 交易对
    action: str            # buy/sell
    quantity: Decimal      # 数量
    price: Decimal         # 价格
    status: str            # pending/executed/cancelled
```

#### watchlist.py - 观察列表
```python
class Watchlist(Base):
    __tablename__ = "watchlist"
    
    user_id: str           # 用户 ID
    symbol: str            # 标的代码
    name: str              # 名称
    exchange: str          # 交易所
    asset_type: str        # 资产类型
```

#### user_profile.py - 用户配置
```python
class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    user_id: str           # 用户 ID（主键）
    timezone: str          # 时区
    language: str          # 语言
    preferences: JSON      # 其他偏好
```

### 4. repositories/ - 仓储模式

仓储模式封装数据访问逻辑，提供统一的接口。

#### agent_repository.py
```python
class AgentRepository:
    async def get_all() -> List[Agent]
    async def get_by_id(agent_id: str) -> Agent
    async def get_by_name(name: str) -> Agent
    async def create(agent: Agent) -> Agent
    async def update(agent: Agent) -> Agent
    async def delete(agent_id: str) -> bool
```

#### strategy_repository.py
```python
class StrategyRepository:
    async def get_all(user_id: str) -> List[Strategy]
    async def get_by_id(strategy_id: str) -> Strategy
    async def create(strategy: Strategy) -> Strategy
    async def update(strategy: Strategy) -> Strategy
    async def delete(strategy_id: str) -> bool
    async def get_holdings(strategy_id: str) -> List[StrategyHolding]
    async def get_portfolio(strategy_id: str) -> StrategyPortfolio
```

## 业务服务层详解

服务层封装业务逻辑，协调多个仓储和外部服务。

### 1. agent_service.py - 智能体服务

**功能**：
- 智能体注册和发现
- Agent Card 管理
- 智能体能力查询

**关键方法**：
```python
async def get_all_agents() -> List[AgentCard]
async def get_agent_by_name(name: str) -> AgentCard
async def register_agent(card: AgentCard) -> bool
```

### 2. agent_stream_service.py - 流式服务

**功能**：
- 处理流式用户输入
- 协调 AgentOrchestrator
- 管理 SSE 连接

**关键方法**：
```python
async def process_user_input_stream(
    user_input: UserInput
) -> AsyncGenerator[BaseResponse, None]
```

### 3. conversation_service.py - 对话服务

**功能**：
- 对话生命周期管理
- 消息持久化
- 对话检索

**关键方法**：
```python
async def create_conversation(user_id: str) -> Conversation
async def get_conversation(conversation_id: str) -> Conversation
async def add_message(conversation_id: str, message: Message)
async def get_messages(conversation_id: str) -> List[Message]
```

### 4. strategy_service.py - 策略服务

**功能**：
- 策略 CRUD
- 策略状态管理
- 策略执行控制

**关键方法**：
```python
async def create_strategy(request: CreateStrategyRequest) -> Strategy
async def start_strategy(strategy_id: str) -> bool
async def stop_strategy(strategy_id: str) -> bool
async def get_strategy_performance(strategy_id: str) -> PerformanceMetrics
```

### 5. strategy_persistence.py - 策略持久化

**功能**：
- 策略状态持久化
- 持仓记录保存
- 交易指令记录

**关键方法**：
```python
async def save_strategy_state(strategy_id: str, state: dict)
async def save_holdings(strategy_id: str, holdings: List[Holding])
async def save_instruction(instruction: Instruction)
```

### 6. strategy_autoresume.py - 策略自动恢复

**功能**：
- 服务器重启后恢复运行中的策略
- 从持久化状态恢复
- 处理异常中断

**关键方法**：
```python
async def auto_resume_strategies() -> List[Strategy]
async def resume_strategy(strategy_id: str) -> bool
```

### 7. task_service.py - 任务服务

**功能**：
- 任务调度
- 定时任务管理
- 任务状态追踪

**关键方法**：
```python
async def create_task(task: Task) -> Task
async def get_task(task_id: str) -> Task
async def cancel_task(task_id: str) -> bool
async def schedule_task(task: Task, schedule: str) -> bool
```

### 8. i18n_service.py - 国际化服务

**功能**：
- 加载语言文件
- 提供翻译服务
- 语言切换

**关键方法**：
```python
def get_supported_languages() -> List[str]
def get_translations(language: str) -> dict
def translate(key: str, language: str, **kwargs) -> str
```

### 9. user_profile_service.py - 用户服务

**功能**：
- 用户配置管理
- 偏好设置

**关键方法**：
```python
async def get_user_profile(user_id: str) -> UserProfile
async def update_user_profile(user_id: str, profile: UserProfile)
```

### 10. assets/asset_service.py - 资产服务

**功能**：
- 资产数据获取
- 价格查询
- 市场数据

**关键方法**：
```python
async def get_asset_price(symbol: str) -> AssetPrice
async def search_assets(query: str) -> List[Asset]
async def get_historical_data(symbol: str, period: str) -> List[OHLCV]
```

## 配置管理

### settings.py - 应用设置

**功能**：从环境变量加载配置

**配置项**：
```python
class Settings(BaseSettings):
    # 应用信息
    APP_NAME: str = "ValueCell Server"
    APP_VERSION: str = "0.1.0"
    APP_ENVIRONMENT: str = "production"
    
    # API 配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # 数据库
    DATABASE_URL: str = "sqlite+aiosqlite:///./valuecell.db"
    
    # AI 模型
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
```

## 错误处理和日志

### 错误响应格式
```json
{
  "code": 404,
  "msg": "Resource not found",
  "details": {
    "resource": "agent",
    "id": "unknown_agent"
  }
}
```

### 日志级别
- **DEBUG**：详细的调试信息（仅在开发模式）
- **INFO**：关键操作和状态变化
- **WARNING**：可恢复的错误
- **ERROR**：需要关注的错误
- **EXCEPTION**：未预期的异常（带堆栈）

## 性能优化

### 1. 异步操作
- 所有数据库操作使用 async/await
- HTTP 请求使用 httpx AsyncClient
- 流式响应避免阻塞

### 2. 连接池
- 数据库连接池
- HTTP 客户端连接复用

### 3. 缓存
- Agent Card 缓存
- 翻译文件缓存
- 用户配置缓存

### 4. 批量操作
- 批量数据库插入
- 批量价格查询

## 安全考虑

### 1. 输入验证
- Pydantic 自动验证所有输入
- 参数类型检查
- 范围验证

### 2. SQL 注入防护
- 使用 SQLAlchemy ORM
- 参数化查询

### 3. CORS 配置
- 可配置的允许源
- 凭据支持控制

### 4. 敏感信息
- 环境变量存储 API 密钥
- 日志中不记录密钥
- 数据库密码加密存储（推荐）

## 测试

### 单元测试示例
```python
@pytest.mark.asyncio
async def test_create_strategy():
    service = StrategyService()
    request = CreateStrategyRequest(
        name="Test Strategy",
        agent_name="grid_agent",
        config={"param": "value"}
    )
    strategy = await service.create_strategy(request)
    assert strategy.name == "Test Strategy"
```

### API 测试
```python
def test_get_agents(client: TestClient):
    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    assert "data" in response.json()
```

## 扩展指南

### 添加新路由
1. 在 `routers/` 创建新文件
2. 定义路由函数
3. 在 `app.py` 中注册

### 添加新模型
1. 在 `db/models/` 创建模型类
2. 继承 `Base`
3. 运行数据库迁移（如果使用 Alembic）

### 添加新服务
1. 在 `services/` 创建服务类
2. 实现业务逻辑
3. 在路由中调用

---

**相关文档**：
- [核心层](./02_core.md)
- [智能体层](./03_agents.md)
- [数据库层](./07_database.md)

