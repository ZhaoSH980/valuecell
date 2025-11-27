# 核心层 (Core)

## 概述

核心层是 ValueCell 的业务逻辑核心，负责协调智能体、管理任务执行、处理事件响应和维护对话状态。这一层实现了多智能体系统的编排逻辑，支持 Human-in-the-Loop (HITL) 和流式响应。

## 目录结构

```
core/
├── agent/              # 智能体基础设施
│   ├── card.py        # Agent Card 解析和查找
│   ├── client.py      # 智能体客户端
│   ├── connect.py     # 远程智能体连接
│   ├── decorator.py   # 智能体装饰器
│   ├── listener.py    # 事件监听器
│   └── responses.py   # 响应工具函数
├── conversation/       # 对话管理
│   ├── conversation_store.py  # 对话存储
│   ├── item_store.py          # 消息项存储
│   ├── manager.py             # 对话管理器
│   ├── models.py              # 对话模型
│   └── service.py             # 对话服务
├── coordinate/         # 协调器
│   ├── orchestrator.py # 主协调器
│   └── services.py     # 服务组合
├── event/              # 事件系统
│   ├── buffer.py       # 响应缓冲器
│   ├── factory.py      # 响应工厂
│   ├── router.py       # 响应路由器
│   └── service.py      # 事件响应服务
├── plan/               # 计划器
│   ├── models.py       # 计划模型
│   ├── planner.py      # 执行计划器
│   ├── prompts.py      # 计划器提示词
│   └── service.py      # 计划服务
├── super_agent/        # 超级智能体
│   ├── core.py         # 超级智能体核心
│   ├── prompts.py      # 提示词
│   └── service.py      # 超级智能体服务
├── task/               # 任务管理
│   ├── executor.py     # 任务执行器
│   ├── locator.py      # 智能体定位器
│   ├── manager.py      # 任务管理器
│   ├── models.py       # 任务模型
│   ├── service.py      # 任务服务
│   └── temporal.py     # 定时任务
├── constants.py        # 核心常量
└── types.py            # 核心类型定义
```

## 核心组件详解

### 1. 协调器 (Orchestrator)

**位置**：`core/coordinate/orchestrator.py`

#### AgentOrchestrator 类

**职责**：
- 协调整个用户输入处理流程
- 管理执行上下文（HITL 状态）
- 编排 SuperAgent、Planner、TaskExecutor
- 处理流式响应

**核心方法**：

##### process_user_input()
```python
async def process_user_input(
    user_input: UserInput
) -> AsyncGenerator[BaseResponse, None]:
    """
    处理用户输入的主入口，支持流式响应和 HITL。
    
    流程：
    1. 创建或恢复对话
    2. SuperAgent 分析意图
    3. 根据意图决定是否需要计划
    4. ExecutionPlanner 生成执行计划（可能暂停等待用户输入）
    5. TaskExecutor 执行任务
    6. 流式返回所有响应
    """
```

**工作流程**：
```
用户输入
    ↓
创建对话 → 保存用户消息
    ↓
SuperAgent 分析 → 决策（需要计划 / 直接响应 / 多智能体协作）
    ↓
[如果需要计划]
ExecutionPlanner → 生成计划（可能触发 UserInputRequest）
    ↓
TaskExecutor → 执行任务列表
    ↓
流式返回响应
    ↓
标记对话完成
```

##### resume_with_user_input()
```python
async def resume_with_user_input(
    conversation_id: str,
    user_input: UserInput
) -> AsyncGenerator[BaseResponse, None]:
    """
    恢复被暂停的执行（HITL 流程）。
    
    从 ExecutionContext 恢复状态，提供用户输入后继续执行。
    """
```

#### ExecutionContext 类

**职责**：管理暂停的执行状态

**字段**：
- `stage`：暂停的阶段（如 "planning"）
- `conversation_id`：对话 ID
- `thread_id`：线程 ID
- `user_id`：用户 ID
- `created_at`：创建时间
- `metadata`：额外元数据

**方法**：
- `is_expired()`：检查是否超时
- `validate_user()`：验证用户身份
- `add_metadata()`：添加元数据

### 2. 超级智能体 (SuperAgent)

**位置**：`core/super_agent/core.py`

#### SuperAgent 类

**职责**：
- 分析用户意图
- 决定处理策略（直接响应/单智能体/多智能体/需要计划）
- 选择合适的智能体

**决策类型** (`SuperAgentDecision`)：
- `NEEDS_PLANNING`：需要多步骤计划
- `DIRECT_RESPONSE`：直接回答（不需要智能体）
- `SINGLE_AGENT`：单个智能体处理
- `MULTI_AGENT`：多个智能体协作

**核心方法**：
```python
async def analyze_and_decide(
    user_message: str,
    available_agents: List[AgentCard],
    conversation_history: Optional[List] = None
) -> SuperAgentOutcome:
    """
    分析用户输入并决定处理策略。
    
    返回：
    - decision: 决策类型
    - selected_agents: 选中的智能体列表
    - reasoning: 推理过程
    - direct_response: 直接回复（如果适用）
    """
```

**提示词结构**（`prompts.py`）：
- 系统提示：定义 SuperAgent 角色
- 智能体能力描述：可用智能体列表
- 决策指南：如何选择处理策略
- 输出格式：结构化 JSON 响应

### 3. 计划器 (Planner)

**位置**：`core/plan/planner.py`

#### ExecutionPlanner 类

**职责**：
- 将用户需求分解为任务序列
- 支持 HITL（通过 UserInputRequest）
- 任务依赖管理

**核心方法**：
```python
async def create_plan(
    user_input: UserInput,
    available_agents: List[AgentCard],
    on_user_input_request: Optional[Callable] = None
) -> ExecutionPlan:
    """
    创建执行计划。
    
    参数：
    - user_input: 用户输入
    - available_agents: 可用智能体列表
    - on_user_input_request: HITL 回调函数
    
    返回：
    - ExecutionPlan: 包含任务列表的执行计划
    
    HITL 流程：
    当需要用户输入时，调用 on_user_input_request(UserInputRequest)
    外部代码通过 UserInputRequest.provide_response() 提供答案
    计划器等待后继续生成计划
    """
```

#### UserInputRequest 类

**职责**：表示对用户输入的请求（HITL）

**字段**：
- `prompt`：提示文本
- `response`：用户响应
- `event`：asyncio.Event（同步等待）

**方法**：
- `wait_for_response()`：异步等待用户响应
- `provide_response(response)`：提供响应并唤醒等待者

#### ExecutionPlan 模型

```python
@dataclass
class ExecutionPlan:
    tasks: List[Task]           # 任务列表
    reasoning: str              # 计划推理
    plan_id: str                # 计划 ID
    created_at: datetime        # 创建时间
```

### 4. 任务执行器 (Task Executor)

**位置**：`core/task/executor.py`

#### TaskExecutor 类

**职责**：
- 执行任务列表
- 管理任务依赖
- 处理定时任务
- 路由响应到合适的处理器

**核心方法**：

##### execute_task()
```python
async def execute_task(
    task: Task,
    conversation_id: str,
    thread_id: str
) -> AsyncGenerator[BaseResponse, None]:
    """
    执行单个任务。
    
    流程：
    1. 定位智能体（本地或远程）
    2. 准备上下文（依赖、语言、时区、用户配置）
    3. 调用智能体
    4. 处理流式响应
    5. 更新任务状态
    6. 处理定时任务（如果适用）
    """
```

##### execute_plan()
```python
async def execute_plan(
    plan: ExecutionPlan,
    conversation_id: str,
    thread_id: str
) -> AsyncGenerator[BaseResponse, None]:
    """
    执行整个计划（任务序列）。
    
    按顺序执行所有任务，处理依赖关系。
    """
```

#### ScheduledTaskResultAccumulator 类

**职责**：收集定时任务的输出

**功能**：
- 积累消息内容
- 过滤掉中间响应（reasoning、tool_call）
- 最终生成单个结果响应

### 5. 任务管理 (Task Management)

**位置**：`core/task/`

#### Task 模型 (`models.py`)

```python
@dataclass
class Task:
    task_id: str                # 任务 ID
    agent_name: str             # 执行智能体
    description: str            # 任务描述
    input_data: Dict            # 输入数据
    dependencies: List[str]     # 依赖的任务 ID
    status: TaskStatus          # 状态
    result: Optional[Any]       # 执行结果
    error: Optional[str]        # 错误信息
    schedule: Optional[str]     # 定时调度表达式
    metadata: Dict              # 元数据
```

#### TaskStatus 枚举

```python
class TaskStatus(str, Enum):
    PENDING = "pending"         # 等待执行
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消
    SCHEDULED = "scheduled"     # 已调度
```

#### TaskManager 类 (`manager.py`)

**职责**：管理任务生命周期

**方法**：
- `create_task()`：创建任务
- `get_task()`：获取任务
- `update_task_status()`：更新状态
- `cancel_task()`：取消任务
- `get_pending_tasks()`：获取待执行任务

#### Temporal Task Support (`temporal.py`)

**功能**：支持定时任务

```python
def calculate_next_execution_delay(schedule: str) -> float:
    """
    计算下次执行的延迟（秒）。
    
    支持的格式：
    - "*/5 * * * *" (cron 表达式)
    - "every 5 minutes"
    - "daily at 09:00"
    """
```

### 6. 事件和响应系统 (Event System)

**位置**：`core/event/`

#### ResponseFactory 类 (`factory.py`)

**职责**：创建标准化的响应对象

**方法**：
```python
def create_message(content: str, ...) -> BaseResponse
def create_reasoning(content: str, ...) -> BaseResponse
def create_tool_call(tool: str, args: dict, ...) -> BaseResponse
def create_error(error: str, ...) -> BaseResponse
def create_task_status(task: Task, ...) -> BaseResponse
def create_notification(content: str, ...) -> BaseResponse
```

#### ResponseBuffer 类 (`buffer.py`)

**职责**：
- 缓冲流式响应
- 合并多个消息段落
- 生成稳定的 item_id

**缓冲策略**：
- **立即发送**：错误、通知、工具调用
- **缓冲段落**：连续的消息内容合并为一个段落
- **任务结束刷新**：任务完成时强制刷新缓冲

**方法**：
```python
def annotate(response: BaseResponse) -> BaseResponse:
    """为响应添加 item_id 和其他元数据"""

def ingest(response: BaseResponse) -> List[SaveItem]:
    """
    接收响应，返回需要持久化的项。
    
    缓冲策略决定是立即返回还是先缓存。
    """

def flush_task(conversation_id, thread_id, task_id) -> List[SaveItem]:
    """强制刷新指定任务的缓冲"""
```

#### ResponseRouter 类 (`router.py`)

**职责**：路由响应并执行副作用

**副作用类型**：
- `FAIL_TASK`：任务失败时更新状态
- `COMPLETE_TASK`：任务完成时更新状态
- `SCHEDULE_TASK`：调度定时任务

**方法**：
```python
async def handle_status_update(
    factory: ResponseFactory,
    task: Task,
    thread_id: str,
    event
) -> RouteResult:
    """
    处理任务状态更新事件。
    
    返回：
    - responses: 需要发送的响应列表
    - side_effects: 需要执行的副作用列表
    """
```

#### EventResponseService 类 (`service.py`)

**职责**：统一的响应处理入口

**方法**：
```python
async def emit(response: BaseResponse) -> BaseResponse:
    """
    处理单个响应：
    1. 通过 buffer 注解
    2. 通过 buffer 摄入（可能缓冲）
    3. 持久化到 ConversationService
    """

async def emit_many(responses: List[BaseResponse]) -> List[BaseResponse]:
    """批量处理响应"""

async def flush_task_response(conversation_id, thread_id, task_id):
    """强制刷新任务响应"""

async def route_task_status(task: Task, thread_id: str, event) -> RouteResult:
    """路由任务状态更新"""
```

### 7. 对话管理 (Conversation)

**位置**：`core/conversation/`

#### Conversation 模型 (`models.py`)

```python
@dataclass
class Conversation:
    conversation_id: str        # 对话 ID
    user_id: str                # 用户 ID
    status: ConversationStatus  # 状态
    created_at: datetime        # 创建时间
    updated_at: datetime        # 更新时间
    metadata: Dict              # 元数据
```

#### ConversationStatus 枚举

```python
class ConversationStatus(str, Enum):
    ACTIVE = "active"           # 活跃
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败
    WAITING_USER = "waiting_user"  # 等待用户输入
```

#### ConversationItem 模型

```python
@dataclass
class ConversationItem:
    item_id: str                # 消息 ID
    conversation_id: str        # 对话 ID
    thread_id: Optional[str]    # 线程 ID（子任务）
    task_id: Optional[str]      # 任务 ID
    role: str                   # user/assistant/system
    event: str                  # 事件类型
    payload: Dict               # 载荷数据
    agent_name: Optional[str]   # 智能体名称
    metadata: Dict              # 元数据
    created_at: datetime        # 创建时间
```

#### ConversationStore 接口 (`conversation_store.py`)

**职责**：对话的持久化存储

**方法**：
```python
async def create(conversation: Conversation) -> Conversation
async def get(conversation_id: str) -> Optional[Conversation]
async def update(conversation: Conversation) -> Conversation
async def list(user_id: str) -> List[Conversation]
```

**实现**：
- `InMemoryConversationStore`：内存存储（测试用）
- 可扩展：数据库存储

#### ItemStore 接口 (`item_store.py`)

**职责**：对话消息的持久化存储

**方法**：
```python
async def add_item(item: ConversationItem) -> ConversationItem
async def get_items(conversation_id: str) -> List[ConversationItem]
async def get_item(item_id: str) -> Optional[ConversationItem]
```

**实现**：
- `InMemoryItemStore`：内存存储
- `SQLiteItemStore`：SQLite 存储

#### ConversationManager 类 (`manager.py`)

**职责**：高层对话管理

**方法**：
```python
async def create_conversation(user_id: str) -> Conversation
async def get_conversation(conversation_id: str) -> Conversation
async def update_conversation_status(conversation_id: str, status: ConversationStatus)
async def add_item(conversation_id: str, item: ConversationItem)
async def get_history(conversation_id: str) -> List[ConversationItem]
```

#### ConversationService 类 (`service.py`)

**职责**：对话服务的门面类

**封装**：
- ConversationStore
- ItemStore
- ConversationManager

**便捷方法**：
```python
async def create_conversation_with_message(user_id: str, message: str) -> Conversation
async def add_user_message(conversation_id: str, content: str)
async def add_assistant_message(conversation_id: str, content: str)
async def get_conversation_with_items(conversation_id: str) -> Tuple[Conversation, List[ConversationItem]]
```

### 8. 智能体基础设施 (Agent Infrastructure)

**位置**：`core/agent/`

#### AgentCard 解析 (`card.py`)

**功能**：
- 从 JSON 配置加载 Agent Card
- 填充缺失字段
- 验证配置

**方法**：
```python
def parse_local_agent_card_dict(agent_card_dict: dict) -> Optional[AgentCard]:
    """解析字典为 AgentCard"""

def find_local_agent_card_by_agent_name(agent_name: str) -> Optional[AgentCard]:
    """根据名称查找本地 Agent Card"""
```

#### 远程智能体连接 (`connect.py`)

**RemoteConnections 类**

**职责**：管理到远程智能体的连接

**方法**：
```python
def get_or_create_client(base_url: str) -> A2AClient:
    """获取或创建到远程智能体的客户端（连接池）"""

async def send_request(base_url: str, request: dict) -> AsyncGenerator:
    """发送请求到远程智能体并流式接收响应"""
```

#### 智能体装饰器 (`decorator.py`)

**功能**：将普通函数包装为符合智能体接口的对象

```python
def create_wrapped_agent(
    name: str,
    func: Callable,
    description: str = ""
) -> BaseAgent:
    """
    将异步生成器函数包装为 BaseAgent。
    
    用法：
    async def my_agent(user_input: UserInput) -> AsyncGenerator[BaseResponse, None]:
        yield factory.create_message("Hello")
    
    agent = create_wrapped_agent("my_agent", my_agent)
    """
```

#### 响应工具 (`responses.py`)

**便捷函数**：

```python
def streaming(content: str, **kwargs) -> BaseResponse:
    """创建流式消息响应"""

def notification(content: str, **kwargs) -> BaseResponse:
    """创建通知响应"""
```

**事件谓词** (`EventPredicates`)：

```python
class EventPredicates:
    @staticmethod
    def is_message(event: str) -> bool
    
    @staticmethod
    def is_reasoning(event: str) -> bool
    
    @staticmethod
    def is_tool_call(event: str) -> bool
    
    @staticmethod
    def is_error(event: str) -> bool
```

### 9. 核心类型定义 (Types)

**位置**：`core/types.py`

#### BaseAgent 接口

```python
class BaseAgent(Protocol):
    """智能体基础接口"""
    
    async def process(
        self,
        user_input: UserInput
    ) -> AsyncGenerator[BaseResponse, None]:
        """处理用户输入并流式返回响应"""
```

#### UserInput 模型

```python
@dataclass
class UserInput:
    user_id: str                # 用户 ID
    message: str                # 消息内容
    conversation_id: Optional[str]  # 对话 ID
    metadata: Optional[UserInputMetadata]  # 元数据
```

#### UserInputMetadata 模型

```python
@dataclass
class UserInputMetadata:
    language: Optional[str]     # 语言
    timezone: Optional[str]     # 时区
    context: Optional[Dict]     # 额外上下文
```

#### BaseResponse 模型

```python
@dataclass
class BaseResponse:
    event: str                  # 事件类型
    data: ResponseData          # 响应数据
    metadata: Dict              # 元数据
    conversation_id: str        # 对话 ID
    thread_id: Optional[str]    # 线程 ID
    task_id: Optional[str]      # 任务 ID
    item_id: Optional[str]      # 消息 ID（由 buffer 生成）
```

#### StreamResponseEvent 枚举

```python
class StreamResponseEvent(str, Enum):
    MESSAGE = "message"         # 消息
    REASONING = "reasoning"     # 推理过程
    TOOL_CALL = "tool_call"     # 工具调用
    ERROR = "error"             # 错误
    TASK_STATUS = "task_status" # 任务状态
    NOTIFICATION = "notification"  # 通知
```

## 核心常量

**位置**：`core/constants.py`

```python
# 特殊任务 ID
ORIGINAL_USER_INPUT = "original_user_input"
PLANNING_TASK = "planning_task"

# 元数据键
LANGUAGE = "language"
TIMEZONE = "timezone"
USER_PROFILE = "user_profile"
CURRENT_CONTEXT = "current_context"
DEPENDENCIES = "dependencies"
METADATA = "metadata"
```

## 工作流程示例

### 1. 简单查询流程

```
用户："查询 AAPL 股票价格"
    ↓
AgentOrchestrator.process_user_input()
    ↓
SuperAgent.analyze_and_decide()
    → decision: SINGLE_AGENT
    → selected_agent: "market_data_agent"
    ↓
TaskExecutor.execute_task()
    → 调用 market_data_agent
    → 流式返回价格信息
    ↓
EventResponseService.emit()
    → Buffer 缓冲
    → 持久化到 ConversationService
    ↓
返回给用户
```

### 2. 复杂计划流程

```
用户："分析 AAPL 并创建交易策略"
    ↓
SuperAgent 决定：NEEDS_PLANNING
    ↓
ExecutionPlanner.create_plan()
    → 可能触发 UserInputRequest（"请问投资金额？"）
    → 等待用户输入
    → 继续生成计划
    ↓
Plan:
    Task 1: research_agent 分析 AAPL
    Task 2: strategy_agent 创建策略（依赖 Task 1）
    ↓
TaskExecutor.execute_plan()
    → 执行 Task 1
    → 等待完成
    → 执行 Task 2（使用 Task 1 结果）
    ↓
流式返回所有响应
```

### 3. HITL 恢复流程

```
[计划器暂停]
ExecutionContext 保存：
    - stage: "planning"
    - conversation_id: "abc123"
    - thread_id: "thread456"
    ↓
UserInputRequest 发送给前端
    ↓
用户提供输入
    ↓
AgentOrchestrator.resume_with_user_input()
    → 恢复 ExecutionContext
    → 提供用户输入给 UserInputRequest
    → 计划器继续
    → 执行任务
```

## 测试策略

### 单元测试
- 每个模块有独立的 `tests/` 目录
- 使用 pytest + pytest-asyncio
- Mock 外部依赖

### 集成测试
- 测试完整的工作流程
- 使用内存存储（InMemoryStore）

### 测试覆盖
- 正常流程
- 错误处理
- HITL 场景
- 并发场景

## 性能考虑

### 1. 异步执行
- 所有 I/O 操作异步
- 任务并发执行（如果无依赖）

### 2. 流式响应
- 减少首字节时间
- 提升用户体验
- 支持长时间运行的任务

### 3. 缓冲策略
- 合并小消息段落
- 减少数据库写入
- 稳定的 item_id 生成

### 4. 上下文管理
- ExecutionContext 有 TTL
- 定期清理过期上下文

## 扩展点

### 1. 自定义存储
- 实现 ConversationStore 接口
- 实现 ItemStore 接口
- 注入到 ConversationService

### 2. 自定义计划器
- 继承 ExecutionPlanner
- 覆盖计划生成逻辑
- 注入到 AgentOrchestrator

### 3. 自定义路由器
- 扩展 ResponseRouter
- 添加新的副作用类型
- 自定义路由逻辑

### 4. 自定义智能体定位
- 实现 AgentLocator 接口
- 支持新的智能体协议
- 注入到 TaskExecutor

## 最佳实践

### 1. 错误处理
- 捕获特定异常
- 记录详细日志
- 返回友好错误消息

### 2. 日志记录
- 记录关键决策点
- 使用结构化日志
- 避免记录敏感信息

### 3. 资源管理
- 使用 async with 管理资源
- 及时关闭连接
- 清理临时数据

### 4. 类型安全
- 全面使用类型提示
- Pydantic 验证数据
- 避免 Any 类型

---

**相关文档**：
- [服务器层](./01_server.md)
- [智能体层](./03_agents.md)
- [事件系统详解](./08_event_system.md)（如果需要）

