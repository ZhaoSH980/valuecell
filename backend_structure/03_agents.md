# 智能体层 (Agents)

## 概述

智能体层包含 ValueCell 平台的各类智能体实现，每个智能体专注于特定的金融任务。智能体系统采用统一的接口设计，支持本地和远程调用，并通过 Agent Card 系统实现自描述和发现。

## 目录结构

```
agents/
├── common/             # 通用交易框架
│   └── trading/       # 交易智能体基础框架
│       ├── base_agent.py           # 策略智能体基类
│       ├── constants.py            # 常量定义
│       ├── models.py               # 数据模型
│       ├── utils.py                # 工具函数
│       ├── _internal/              # 内部实现
│       │   ├── runtime.py          # 策略运行时
│       │   ├── stream_controller.py # 流控制器
│       │   ├── state.py            # 状态管理
│       │   └── context.py          # 上下文管理
│       ├── data/                   # 数据获取
│       │   ├── fetcher.py          # 数据获取器
│       │   ├── aggregator.py       # 数据聚合器
│       │   └── validator.py        # 数据验证
│       ├── decision/               # 决策系统
│       │   ├── interfaces.py       # 决策接口
│       │   ├── llm_composer/       # LLM 决策组合器
│       │   │   ├── composer.py
│       │   │   └── prompts.py
│       │   └── grid_composer/      # 网格决策组合器
│       │       ├── grid_composer.py
│       │       └── price_ladder.py
│       ├── execution/              # 执行引擎
│       │   ├── engine.py           # 执行引擎
│       │   ├── order_manager.py    # 订单管理
│       │   ├── simulator.py        # 模拟交易
│       │   └── okx_executor.py     # OKX 执行器
│       ├── features/               # 特征工程
│       │   ├── interfaces.py       # 特征接口
│       │   ├── pipeline.py         # 特征管道
│       │   ├── technical.py        # 技术指标
│       │   └── market_data.py      # 市场数据特征
│       ├── history/                # 历史记录
│       │   ├── tracker.py          # 交易记录追踪
│       │   └── analyzer.py         # 分析器
│       └── portfolio/              # 投资组合管理
│           ├── manager.py          # 组合管理器
│           ├── position.py         # 持仓管理
│           └── metrics.py          # 绩效指标
├── grid_agent/         # 网格交易智能体
│   ├── __init__.py
│   └── grid_agent.py
├── news_agent/         # 新闻监控智能体
│   ├── __init__.py
│   ├── core.py         # 核心实现
│   ├── prompts.py      # 提示词
│   └── tools.py        # 工具函数
├── research_agent/     # 研究分析智能体
│   ├── __init__.py
│   ├── core.py         # 核心实现
│   ├── prompts.py      # 提示词
│   ├── knowledge.py    # 知识库
│   ├── sources.py      # 数据源
│   ├── schemas.py      # 数据模型
│   └── vdb.py          # 向量数据库
├── prompt_strategy_agent/  # 提示策略智能体
│   ├── __init__.py
│   ├── core.py         # 核心实现
│   └── templates/      # 策略模板
│       ├── default.txt
│       ├── aggressive.txt
│       └── insane.txt
├── sources/            # 数据源集成
│   ├── __init__.py
│   └── rootdata.py     # RootData API
└── utils/              # 智能体工具
    ├── __init__.py
    └── context.py      # 上下文构建
```

## 智能体接口

### BaseAgent 协议

**位置**：`core/types.py`

```python
class BaseAgent(Protocol):
    """智能体基础协议"""
    
    async def stream(
        self,
        query: str,
        conversation_id: str,
        task_id: str,
        dependencies: Optional[Dict] = None,
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        流式处理查询。
        
        参数：
        - query: 用户查询
        - conversation_id: 对话 ID
        - task_id: 任务 ID
        - dependencies: 依赖任务的结果
        
        返回：
        - 流式响应生成器
        """
```

### Agent Card 系统

每个智能体通过 JSON 配置文件定义其元数据：

**位置**：`configs/agent_cards/{agent_name}.json`

```json
{
  "name": "research_agent",
  "display_name": "研究分析智能体",
  "description": "分析公司财报、SEC 文件和市场数据",
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
    "tags": ["research", "SEC", "financial-analysis"]
  }
}
```

## 交易智能体框架

### 1. BaseStrategyAgent - 策略智能体基类

**位置**：`agents/common/trading/base_agent.py`

#### 功能
- 策略全生命周期管理
- 特征计算管道
- 决策生成
- 交易执行
- 持仓管理
- 状态持久化

#### 抽象方法

##### _build_features_pipeline()
```python
async def _build_features_pipeline(
    self, request: UserRequest
) -> BaseFeaturesPipeline | None:
    """
    构建特征管道。
    
    返回 None 使用默认管道，或返回自定义 FeaturesPipeline。
    """
```

##### _create_decision_composer()
```python
async def _create_decision_composer(
    self, request: UserRequest
) -> BaseComposer | None:
    """
    创建决策组合器。
    
    返回 None 使用默认 LLM 组合器，或返回自定义组合器。
    """
```

#### 生命周期钩子

```python
async def _on_start(self, runtime: StrategyRuntime, request: UserRequest):
    """策略启动后回调"""

async def _on_cycle_result(self, result: DecisionCycleResult, runtime: StrategyRuntime, request: UserRequest):
    """每个决策周期后回调"""

async def _on_stop(self, runtime: StrategyRuntime, request: UserRequest):
    """策略停止前回调"""
```

#### 核心流程

```
stream(request)
    ↓
创建 StrategyRuntime
    ↓
调用 _on_start()
    ↓
循环：
    ├─ FeaturesPipeline 计算特征
    ├─ DecisionComposer 生成决策
    ├─ ExecutionEngine 执行交易
    ├─ PortfolioManager 更新持仓
    ├─ 持久化状态
    ├─ 调用 _on_cycle_result()
    └─ 检查停止条件
    ↓
调用 _on_stop()
    ↓
最终化持久化
```

### 2. StrategyRuntime - 策略运行时

**位置**：`agents/common/trading/_internal/runtime.py`

#### 职责
- 协调各个组件
- 管理状态
- 控制决策周期
- 处理错误和恢复

#### 关键组件
```python
class StrategyRuntime:
    portfolio_manager: PortfolioManager    # 组合管理
    features_pipeline: BaseFeaturesPipeline  # 特征管道
    decision_composer: BaseComposer        # 决策器
    execution_engine: ExecutionEngine      # 执行引擎
    history_tracker: HistoryTracker        # 历史追踪
    stream_controller: StreamController    # 流控制器
```

#### 核心方法
```python
async def run_decision_cycle() -> DecisionCycleResult:
    """
    执行一个完整的决策周期。
    
    返回：
    - DecisionCycleResult: 包含特征、决策、执行结果
    """
```

### 3. FeaturesPipeline - 特征计算管道

**位置**：`agents/common/trading/features/`

#### BaseFeaturesPipeline 接口

```python
class BaseFeaturesPipeline(ABC):
    @abstractmethod
    async def compute_features(self, context: dict) -> FeaturesOutput:
        """
        计算市场特征。
        
        返回：
        - current_price: 当前价格
        - market_data: 市场数据（OHLCV）
        - technical_indicators: 技术指标
        - metadata: 其他元数据
        """
```

#### DefaultFeaturesPipeline 实现

**功能**：
- 获取实时价格
- 获取历史 K 线数据
- 计算技术指标（MA、RSI、MACD、Bollinger Bands）
- 数据验证和清洗

**技术指标**：
```python
indicators = {
    "ma_short": 简单移动平均（短期）,
    "ma_long": 简单移动平均（长期）,
    "rsi": 相对强弱指标,
    "macd": MACD 指标,
    "macd_signal": MACD 信号线,
    "bb_upper": 布林带上轨,
    "bb_lower": 布林带下轨,
    "volume_avg": 平均成交量
}
```

### 4. DecisionComposer - 决策组合器

**位置**：`agents/common/trading/decision/`

#### BaseComposer 接口

```python
class BaseComposer(ABC):
    @abstractmethod
    async def compose_decision(
        self,
        features: FeaturesOutput,
        portfolio: Portfolio,
        context: dict
    ) -> ComposedDecision:
        """
        生成交易决策。
        
        返回：
        - actions: 交易动作列表（buy/sell/hold）
        - reasoning: 决策理由
        - confidence: 信心水平
        """
```

#### LLMComposer - LLM 决策器

**功能**：
- 使用 LLM 分析市场特征
- 结合持仓状态
- 生成交易决策和推理
- 可配置策略风格（保守/平衡/激进）

**提示词结构**：
```
系统角色：专业交易策略师
当前市场数据：价格、指标
持仓信息：现金、持仓、盈亏
策略约束：风险管理规则
输出格式：JSON（actions + reasoning）
```

#### GridComposer - 网格交易决策器

**功能**：
- 基于价格网格的规则交易
- 价格上涨时卖出（获利）
- 价格下跌时买入（摊平）
- 支持现货和永续合约

**配置参数**：
```python
step_pct: float = 0.001    # 网格步长（0.1%）
max_steps: int = 3         # 最大步数
base_fraction: float = 0.08  # 基础仓位比例
use_llm_params: bool = True  # 是否使用 LLM 优化参数
```

**网格策略**：
```
现货模式：
- 价格上涨 > step_pct → 卖出部分仓位
- 价格下跌 > step_pct → 买入加仓

永续模式：
- 价格上涨 > step_pct → 增加空头（做空）
- 价格下跌 > step_pct → 增加多头（做多）
```

### 5. ExecutionEngine - 执行引擎

**位置**：`agents/common/trading/execution/`

#### 功能
- 订单生成和提交
- 风险检查
- 滑点控制
- 订单状态追踪

#### 执行模式

##### 模拟模式 (Simulator)
```python
class SimulatedExecutor:
    """
    模拟交易执行器。
    
    - 模拟订单成交
    - 模拟滑点
    - 无需真实资金
    """
```

##### 实盘模式 (OKX)
```python
class OKXExecutor:
    """
    OKX 交易所执行器。
    
    - 支持现货和永续合约
    - 支持 paper trading
    - 订单管理和追踪
    """
```

#### 执行流程

```
ComposedDecision
    ↓
生成订单列表
    ↓
风险检查
    - 检查余额
    - 检查持仓限制
    - 检查价格偏离
    ↓
提交订单
    ↓
等待成交
    ↓
返回执行结果
```

### 6. PortfolioManager - 组合管理器

**位置**：`agents/common/trading/portfolio/`

#### Portfolio 模型

```python
@dataclass
class Portfolio:
    cash: Decimal              # 现金余额
    positions: Dict[str, Position]  # 持仓字典
    total_value: Decimal       # 总价值
    unrealized_pnl: Decimal    # 未实现盈亏
    realized_pnl: Decimal      # 已实现盈亏
```

#### Position 模型

```python
@dataclass
class Position:
    symbol: str                # 交易对
    quantity: Decimal          # 数量
    avg_price: Decimal         # 平均价格
    current_price: Decimal     # 当前价格
    market_value: Decimal      # 市值
    unrealized_pnl: Decimal    # 未实现盈亏
    side: str                  # long/short
```

#### 功能

```python
class PortfolioManager:
    def update_position(self, symbol: str, quantity: Decimal, price: Decimal):
        """更新持仓"""
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """计算绩效指标"""
    
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
    
    def get_available_cash(self) -> Decimal:
        """获取可用现金"""
```

## 具体智能体实现

### 1. GridStrategyAgent - 网格交易智能体

**位置**：`agents/grid_agent/grid_agent.py`

#### 特点
- 基于 BaseStrategyAgent
- 使用 DefaultFeaturesPipeline
- 使用 GridComposer
- 适合震荡市场

#### 配置示例

```json
{
  "symbol": "BTC-USDT",
  "initial_capital": 10000,
  "step_pct": 0.001,
  "max_steps": 3,
  "cycle_interval": 60,
  "exchange": "okx",
  "network": "paper"
}
```

#### 使用场景
- 震荡行情套利
- 高频小额交易
- 低风险稳定收益

### 2. PromptStrategyAgent - 提示策略智能体

**位置**：`agents/prompt_strategy_agent/core.py`

#### 特点
- 完全由 LLM 驱动
- 可自定义策略模板
- 灵活的决策逻辑

#### 策略模板

##### default.txt - 平衡策略
```
目标：稳健增长
风险偏好：中等
持仓比例：30-70%
止损：-5%
```

##### aggressive.txt - 激进策略
```
目标：最大化收益
风险偏好：高
持仓比例：50-100%
止损：-10%
```

##### insane.txt - 极端策略
```
目标：超高收益
风险偏好：极高
持仓比例：100%（可杠杆）
止损：-20%
```

#### 使用场景
- 趋势跟踪
- 基本面交易
- 情绪驱动交易

### 3. ResearchAgent - 研究分析智能体

**位置**：`agents/research_agent/core.py`

#### 功能
- SEC 文件分析（10-K、10-Q、8-K）
- A 股公告分析
- 网络搜索
- 向量数据库检索
- 加密货币项目研究

#### 工具列表

##### fetch_periodic_sec_filings
```python
def fetch_periodic_sec_filings(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 5
) -> str:
    """
    获取 SEC 定期报告。
    
    支持的表单类型：
    - 10-K: 年度报告
    - 10-Q: 季度报告
    """
```

##### fetch_event_sec_filings
```python
def fetch_event_sec_filings(
    ticker: str,
    limit: int = 10
) -> str:
    """
    获取 SEC 事件报告（8-K）。
    
    包括：
    - 重大事件披露
    - 管理层变动
    - 财务事件
    """
```

##### fetch_ashare_filings
```python
def fetch_ashare_filings(
    stock_code: str,
    limit: int = 10
) -> str:
    """
    获取 A 股公告。
    
    来源：AKShare
    """
```

##### web_search
```python
def web_search(
    query: str,
    max_results: int = 5
) -> str:
    """
    网络搜索。
    
    使用 Crawl4AI 进行深度抓取。
    """
```

#### 知识库

**VectorDB 集成**（`vdb.py`）：
- 使用 LanceDB
- 文档向量化
- 语义搜索
- 相关性排序

#### 使用场景
- 基本面分析
- 公司研究
- 行业分析
- 事件影响评估

### 4. NewsAgent - 新闻监控智能体

**位置**：`agents/news_agent/core.py`

#### 功能
- 突发新闻获取
- 财经新闻分析
- 新闻情绪分析
- 网络搜索

#### 工具列表

##### get_breaking_news
```python
def get_breaking_news(
    keywords: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    获取突发新闻。
    
    来源：新闻 API
    """
```

##### get_financial_news
```python
def get_financial_news(
    symbol: Optional[str] = None,
    category: str = "general",
    limit: int = 20
) -> str:
    """
    获取财经新闻。
    
    分类：
    - general: 综合
    - stock: 股票
    - crypto: 加密货币
    - forex: 外汇
    """
```

#### 使用场景
- 实时新闻监控
- 事件驱动交易
- 市场情绪分析

## 数据源集成

### RootData API

**位置**：`agents/sources/rootdata.py`

#### 功能
- 加密货币项目数据
- VC 投资信息
- 加密货币名人数据

#### API 方法

```python
def search_crypto_projects(query: str) -> List[Project]:
    """搜索加密货币项目"""

def search_crypto_vcs(query: str) -> List[VC]:
    """搜索加密货币 VC"""

def search_crypto_people(query: str) -> List[Person]:
    """搜索加密货币领域人物"""
```

## 智能体工具

### 上下文构建

**位置**：`agents/utils/context.py`

```python
def build_ctx_from_dep(dependencies: Optional[Dict]) -> str:
    """
    从依赖任务结果构建上下文字符串。
    
    用于将前置任务的输出传递给当前任务。
    """
```

## 智能体通信模型

### StreamResponse 格式

```python
{
    "type": "message_chunk",
    "content": "分析结果...",
    "metadata": {
        "agent": "research_agent",
        "task_id": "task123"
    }
}
```

### 事件类型

- `message_chunk`：消息内容片段
- `reasoning`：推理过程
- `tool_call_started`：工具调用开始
- `tool_call_completed`：工具调用完成
- `error`：错误
- `done`：完成

## 配置系统

### Agent 配置文件

**位置**：`configs/agents/{agent_name}.yaml`

```yaml
name: research_agent
model:
  provider: openai
  name: gpt-4
  temperature: 0.7
  max_tokens: 4000
tools:
  - fetch_periodic_sec_filings
  - fetch_event_sec_filings
  - web_search
knowledge:
  enabled: true
  vector_db: lancedb
  index_path: ./data/knowledge/research
```

## 策略持久化

### 状态保存

```python
@dataclass
class StrategyState:
    strategy_id: str
    portfolio: Portfolio
    cycle_count: int
    last_cycle_time: datetime
    performance_metrics: PerformanceMetrics
    metadata: Dict
```

### 自动恢复

服务器重启后：
1. 从数据库加载策略状态
2. 恢复 PortfolioManager
3. 恢复 StrategyRuntime
4. 继续执行决策周期

## 性能监控

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    total_return: Decimal       # 总收益率
    annualized_return: Decimal  # 年化收益率
    sharpe_ratio: Decimal       # 夏普比率
    max_drawdown: Decimal       # 最大回撤
    win_rate: Decimal           # 胜率
    total_trades: int           # 总交易次数
    profit_factor: Decimal      # 盈亏比
```

### 实时监控

```python
class HistoryTracker:
    """追踪交易历史和绩效"""
    
    def record_trade(self, trade: Trade):
        """记录交易"""
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """计算绩效指标"""
    
    def get_trade_history(self) -> List[Trade]:
        """获取交易历史"""
```

## 风险管理

### 风险检查

```python
class RiskManager:
    def check_position_limit(self, symbol: str, quantity: Decimal) -> bool:
        """检查持仓限制"""
    
    def check_order_size(self, order: Order) -> bool:
        """检查订单大小"""
    
    def check_drawdown_limit(self, portfolio: Portfolio) -> bool:
        """检查回撤限制"""
```

### 风险参数

```python
max_position_pct: float = 0.3    # 单个持仓最大占比 30%
max_drawdown_pct: float = 0.15   # 最大回撤 15%
stop_loss_pct: float = 0.05      # 止损 5%
max_leverage: float = 1.0        # 最大杠杆
```

## 测试

### 回测支持

```python
class Backtester:
    """策略回测"""
    
    async def run_backtest(
        self,
        agent: BaseStrategyAgent,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal
    ) -> BacktestResult:
        """运行回测"""
```

### 单元测试

```python
@pytest.mark.asyncio
async def test_grid_agent():
    agent = GridStrategyAgent()
    request = UserRequest(
        symbol="BTC-USDT",
        initial_capital=10000
    )
    results = []
    async for response in agent.stream(request):
        results.append(response)
    assert len(results) > 0
```

## 扩展指南

### 创建新智能体

#### 1. 非交易智能体

```python
from valuecell.core.types import BaseAgent

class MyAgent(BaseAgent):
    async def stream(self, query, conversation_id, task_id, dependencies=None):
        # 实现逻辑
        yield streaming.message_chunk("响应内容")
        yield streaming.done()
```

#### 2. 交易智能体

```python
from valuecell.agents.common.trading.base_agent import BaseStrategyAgent

class MyStrategyAgent(BaseStrategyAgent):
    async def _build_features_pipeline(self, request):
        return MyFeaturesPipeline()
    
    async def _create_decision_composer(self, request):
        return MyComposer()
```

### 添加新工具

```python
def my_tool(param: str) -> str:
    """
    工具描述（LLM 会看到）。
    
    参数：
    - param: 参数描述
    
    返回：
    - 结果描述
    """
    # 实现
    return result

# 注册到智能体
agent = Agent(
    tools=[my_tool]
)
```

### 添加 Agent Card

在 `configs/agent_cards/` 创建 JSON 文件：

```json
{
  "name": "my_agent",
  "display_name": "我的智能体",
  "description": "智能体描述",
  "enabled": true,
  "capabilities": {
    "streaming": true
  }
}
```

## 最佳实践

### 1. 异步操作
- 所有 I/O 使用 async/await
- 避免阻塞操作

### 2. 错误处理
- 捕获特定异常
- 记录详细日志
- 优雅降级

### 3. 流式响应
- 及时 yield 响应
- 避免长时间无响应
- 最后发送 done 事件

### 4. 资源管理
- 及时关闭连接
- 清理临时数据
- 使用连接池

### 5. 日志记录
- 记录关键决策
- 记录工具调用
- 避免敏感信息

---

**相关文档**：
- [核心层](./02_core.md)
- [适配器层](./04_adapters.md)
- [交易框架详解](./08_trading_framework.md)（如果需要）

