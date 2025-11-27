# 适配器层 (Adapters)

## 概述

适配器层提供统一的接口来访问不同的外部服务，包括金融数据源、AI 模型和数据库。适配器模式使得系统可以轻松切换不同的实现，而不影响上层业务逻辑。

## 目录结构

```
adapters/
├── assets/             # 资产数据适配器
│   ├── __init__.py
│   ├── base.py        # 基础适配器接口
│   ├── manager.py     # 适配器管理器
│   ├── types.py       # 数据类型定义
│   ├── akshare_adapter.py    # AKShare 适配器
│   ├── yfinance_adapter.py   # Yahoo Finance 适配器
│   └── i18n_integration.py   # 国际化集成
├── db/                # 数据库适配器
│   └── __init__.py
└── models/            # 模型适配器
    ├── __init__.py
    └── factory.py     # 模型工厂
```

## 资产数据适配器

### 1. 适配器接口 (BaseDataAdapter)

**位置**：`adapters/assets/base.py`

#### 核心接口

```python
class BaseDataAdapter(ABC):
    """资产数据适配器基类"""
    
    def __init__(self, source: DataSource):
        self.source = source
    
    @abstractmethod
    def get_capabilities(self) -> List[AdapterCapability]:
        """
        返回适配器支持的能力。
        
        返回：
        - List[AdapterCapability]: 能力列表
        """
    
    @abstractmethod
    def search_assets(self, query: AssetSearchQuery) -> List[AssetSearchResult]:
        """
        搜索资产。
        
        参数：
        - query: 搜索查询
        
        返回：
        - List[AssetSearchResult]: 搜索结果
        """
    
    @abstractmethod
    def get_asset_info(self, symbol: str, exchange: Exchange) -> Optional[Asset]:
        """
        获取资产详细信息。
        
        参数：
        - symbol: 资产代码
        - exchange: 交易所
        
        返回：
        - Asset: 资产信息
        """
    
    @abstractmethod
    def get_current_price(self, symbol: str, exchange: Exchange) -> Optional[AssetPrice]:
        """
        获取当前价格。
        
        参数：
        - symbol: 资产代码
        - exchange: 交易所
        
        返回：
        - AssetPrice: 价格信息
        """
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        exchange: Exchange,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCV]:
        """
        获取历史数据。
        
        参数：
        - symbol: 资产代码
        - exchange: 交易所
        - start_date: 开始日期
        - end_date: 结束日期
        - interval: 时间间隔（1m, 5m, 1h, 1d, 1w）
        
        返回：
        - List[OHLCV]: K 线数据列表
        """
    
    @abstractmethod
    def get_batch_prices(
        self,
        symbols: List[str],
        exchange: Exchange
    ) -> Dict[str, AssetPrice]:
        """
        批量获取价格。
        
        参数：
        - symbols: 资产代码列表
        - exchange: 交易所
        
        返回：
        - Dict[symbol, AssetPrice]: 价格字典
        """
```

### 2. 数据类型定义 (types.py)

**位置**：`adapters/assets/types.py`

#### DataSource 枚举

```python
class DataSource(str, Enum):
    YFINANCE = "yfinance"      # Yahoo Finance
    AKSHARE = "akshare"        # AKShare（中国市场）
    BINANCE = "binance"        # 币安
    OKX = "okx"                # OKX
    CUSTOM = "custom"          # 自定义
```

#### Exchange 枚举

```python
class Exchange(str, Enum):
    NYSE = "nyse"              # 纽约证券交易所
    NASDAQ = "nasdaq"          # 纳斯达克
    SSE = "sse"                # 上海证券交易所
    SZSE = "szse"              # 深圳证券交易所
    HKEX = "hkex"              # 香港交易所
    BINANCE = "binance"        # 币安
    OKX = "okx"                # OKX
    UNKNOWN = "unknown"        # 未知
```

#### AssetType 枚举

```python
class AssetType(str, Enum):
    STOCK = "stock"            # 股票
    CRYPTO = "crypto"          # 加密货币
    FOREX = "forex"            # 外汇
    COMMODITY = "commodity"    # 商品
    INDEX = "index"            # 指数
    BOND = "bond"              # 债券
```

#### Asset 模型

```python
@dataclass
class Asset:
    symbol: str                # 资产代码
    name: str                  # 资产名称
    asset_type: AssetType      # 资产类型
    exchange: Exchange         # 交易所
    currency: str              # 计价货币
    description: Optional[str] # 描述
    sector: Optional[str]      # 行业
    market_cap: Optional[Decimal]  # 市值
    metadata: Dict[str, Any]   # 额外元数据
```

#### AssetPrice 模型

```python
@dataclass
class AssetPrice:
    symbol: str                # 资产代码
    price: Decimal             # 当前价格
    bid: Optional[Decimal]     # 买价
    ask: Optional[Decimal]     # 卖价
    volume: Optional[Decimal]  # 成交量
    timestamp: datetime        # 时间戳
    change: Optional[Decimal]  # 涨跌额
    change_pct: Optional[Decimal]  # 涨跌幅
    open: Optional[Decimal]    # 开盘价
    high: Optional[Decimal]    # 最高价
    low: Optional[Decimal]     # 最低价
    prev_close: Optional[Decimal]  # 前收盘价
```

#### OHLCV 模型

```python
@dataclass
class OHLCV:
    timestamp: datetime        # 时间戳
    open: Decimal              # 开盘价
    high: Decimal              # 最高价
    low: Decimal               # 最低价
    close: Decimal             # 收盘价
    volume: Decimal            # 成交量
    symbol: str                # 资产代码
```

#### AdapterCapability 模型

```python
@dataclass
class AdapterCapability:
    asset_type: AssetType      # 支持的资产类型
    exchanges: List[Exchange]  # 支持的交易所
    features: List[str]        # 支持的功能
    # 功能示例：
    # - "real_time_price"
    # - "historical_data"
    # - "company_info"
    # - "fundamental_data"
```

### 3. YFinance 适配器

**位置**：`adapters/assets/yfinance_adapter.py`

#### 特点
- 免费、无需 API 密钥
- 支持全球主要市场
- 实时价格（延迟 15 分钟）
- 历史数据
- 公司基本面信息

#### 支持的市场
- 美股（NYSE、NASDAQ）
- 加密货币（通过 -USD 后缀）
- 外汇对
- 商品期货
- 全球指数

#### 实现示例

```python
class YFinanceAdapter(BaseDataAdapter):
    def __init__(self):
        super().__init__(DataSource.YFINANCE)
    
    def get_current_price(self, symbol: str, exchange: Exchange) -> Optional[AssetPrice]:
        """获取当前价格"""
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return AssetPrice(
            symbol=symbol,
            price=Decimal(str(info.get('currentPrice', 0))),
            bid=Decimal(str(info.get('bid', 0))),
            ask=Decimal(str(info.get('ask', 0))),
            volume=Decimal(str(info.get('volume', 0))),
            timestamp=datetime.now(),
            change=Decimal(str(info.get('regularMarketChange', 0))),
            change_pct=Decimal(str(info.get('regularMarketChangePercent', 0))),
            open=Decimal(str(info.get('regularMarketOpen', 0))),
            high=Decimal(str(info.get('regularMarketDayHigh', 0))),
            low=Decimal(str(info.get('regularMarketDayLow', 0))),
            prev_close=Decimal(str(info.get('previousClose', 0)))
        )
    
    def get_historical_data(
        self,
        symbol: str,
        exchange: Exchange,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCV]:
        """获取历史数据"""
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        result = []
        for index, row in df.iterrows():
            result.append(OHLCV(
                timestamp=index.to_pydatetime(),
                open=Decimal(str(row['Open'])),
                high=Decimal(str(row['High'])),
                low=Decimal(str(row['Low'])),
                close=Decimal(str(row['Close'])),
                volume=Decimal(str(row['Volume'])),
                symbol=symbol
            ))
        return result
```

#### 符号格式
```
股票：AAPL, TSLA, MSFT
加密货币：BTC-USD, ETH-USD
外汇：EURUSD=X
商品：GC=F (黄金期货)
指数：^GSPC (标普500)
```

### 4. AKShare 适配器

**位置**：`adapters/assets/akshare_adapter.py`

#### 特点
- 免费、无需 API 密钥
- 专注中国市场
- 实时行情
- 历史数据
- A股特有数据（财报、公告）

#### 支持的市场
- A股（上交所、深交所）
- 港股（部分）
- 期货
- 基金
- 债券

#### 实现示例

```python
class AKShareAdapter(BaseDataAdapter):
    def __init__(self):
        super().__init__(DataSource.AKSHARE)
        # 优化：延迟导入，避免启动时加载
        self._akshare = None
    
    @property
    def ak(self):
        """延迟加载 akshare"""
        if self._akshare is None:
            import akshare as ak
            self._akshare = ak
        return self._akshare
    
    def get_current_price(self, symbol: str, exchange: Exchange) -> Optional[AssetPrice]:
        """获取当前价格"""
        try:
            # 转换为 AKShare 格式（如 sh600000）
            ak_symbol = self._convert_symbol(symbol, exchange)
            df = self.ak.stock_zh_a_spot_em()
            
            # 查找对应股票
            row = df[df['代码'] == ak_symbol].iloc[0]
            
            return AssetPrice(
                symbol=symbol,
                price=Decimal(str(row['最新价'])),
                volume=Decimal(str(row['成交量'])),
                timestamp=datetime.now(),
                change=Decimal(str(row['涨跌额'])),
                change_pct=Decimal(str(row['涨跌幅'])),
                open=Decimal(str(row['今开'])),
                high=Decimal(str(row['最高'])),
                low=Decimal(str(row['最低'])),
                prev_close=Decimal(str(row['昨收']))
            )
        except Exception as e:
            logger.error(f"AKShare get_current_price error: {e}")
            return None
    
    def _convert_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        转换符号格式。
        
        600000 (SSE) -> sh600000
        000001 (SZSE) -> sz000001
        """
        if exchange == Exchange.SSE:
            return f"sh{symbol}"
        elif exchange == Exchange.SZSE:
            return f"sz{symbol}"
        return symbol
```

#### 符号格式
```
上交所：600000, 600519
深交所：000001, 000002
创业板：300001
科创板：688001
港股：00700
```

### 5. 适配器管理器 (AdapterManager)

**位置**：`adapters/assets/manager.py`

#### 职责
- 注册和管理多个适配器
- 路由请求到合适的适配器
- 缓存和优化
- 故障转移

#### 核心功能

```python
class AdapterManager:
    def __init__(self):
        self.adapters: Dict[DataSource, BaseDataAdapter] = {}
        self.exchange_routing: Dict[str, List[BaseDataAdapter]] = {}
        self._ticker_cache: Dict[str, BaseDataAdapter] = {}
        self.lock = threading.RLock()
    
    def register_adapter(self, adapter: BaseDataAdapter):
        """注册适配器并重建路由表"""
        with self.lock:
            self.adapters[adapter.source] = adapter
            self._rebuild_routing_table()
    
    def _rebuild_routing_table(self):
        """
        根据适配器能力构建路由表。
        
        Exchange -> List[Adapter]
        """
        self.exchange_routing.clear()
        
        for adapter in self.adapters.values():
            capabilities = adapter.get_capabilities()
            for cap in capabilities:
                for exchange in cap.exchanges:
                    exchange_key = exchange.value
                    if exchange_key not in self.exchange_routing:
                        self.exchange_routing[exchange_key] = []
                    self.exchange_routing[exchange_key].append(adapter)
    
    def get_current_price(
        self,
        symbol: str,
        exchange: Exchange
    ) -> Optional[AssetPrice]:
        """
        获取价格（带智能路由）。
        
        优先级：
        1. 检查缓存（找到之前成功的适配器）
        2. 根据交易所查找支持的适配器
        3. 尝试所有适配器（故障转移）
        """
        # 1. 检查缓存
        cache_key = f"{symbol}:{exchange.value}"
        with self._cache_lock:
            cached_adapter = self._ticker_cache.get(cache_key)
        
        if cached_adapter:
            try:
                price = cached_adapter.get_current_price(symbol, exchange)
                if price:
                    return price
            except Exception as e:
                logger.warning(f"Cached adapter failed: {e}")
        
        # 2. 根据交易所查找
        exchange_key = exchange.value
        adapters = self.exchange_routing.get(exchange_key, [])
        
        for adapter in adapters:
            try:
                price = adapter.get_current_price(symbol, exchange)
                if price:
                    # 更新缓存
                    with self._cache_lock:
                        self._ticker_cache[cache_key] = adapter
                    return price
            except Exception as e:
                logger.warning(f"Adapter {adapter.source} failed: {e}")
                continue
        
        # 3. 尝试所有适配器（故障转移）
        for adapter in self.adapters.values():
            if adapter in adapters:
                continue  # 已经尝试过
            try:
                price = adapter.get_current_price(symbol, exchange)
                if price:
                    with self._cache_lock:
                        self._ticker_cache[cache_key] = adapter
                    return price
            except Exception:
                continue
        
        return None
    
    def get_batch_prices(
        self,
        symbols: List[str],
        exchange: Exchange
    ) -> Dict[str, AssetPrice]:
        """
        批量获取价格（并行）。
        
        使用线程池并发获取。
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.get_current_price,
                    symbol,
                    exchange
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    price = future.result()
                    if price:
                        results[symbol] = price
                except Exception as e:
                    logger.error(f"Failed to get price for {symbol}: {e}")
        
        return results
    
    def configure_yfinance(self, **kwargs):
        """快捷配置 Yahoo Finance"""
        adapter = YFinanceAdapter()
        self.register_adapter(adapter)
    
    def configure_akshare(self, **kwargs):
        """快捷配置 AKShare"""
        adapter = AKShareAdapter()
        self.register_adapter(adapter)
```

#### 单例模式

```python
_manager_instance: Optional[AdapterManager] = None
_manager_lock = threading.Lock()

def get_adapter_manager() -> AdapterManager:
    """获取全局适配器管理器单例"""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = AdapterManager()
    return _manager_instance
```

### 6. 国际化集成

**位置**：`adapters/assets/i18n_integration.py`

#### 功能
- 资产名称翻译
- 交易所名称翻译
- 行业分类翻译

```python
def translate_asset_name(symbol: str, language: str) -> str:
    """翻译资产名称"""
    translations = {
        "AAPL": {"zh-Hans": "苹果", "zh-Hant": "蘋果"},
        "TSLA": {"zh-Hans": "特斯拉", "zh-Hant": "特斯拉"}
    }
    return translations.get(symbol, {}).get(language, symbol)

def translate_exchange_name(exchange: Exchange, language: str) -> str:
    """翻译交易所名称"""
    translations = {
        Exchange.NYSE: {"zh-Hans": "纽约证券交易所", "zh-Hant": "紐約證券交易所"},
        Exchange.SSE: {"zh-Hans": "上海证券交易所", "zh-Hant": "上海證券交易所"}
    }
    return translations.get(exchange, {}).get(language, exchange.value)
```

## 模型适配器

### 模型工厂 (factory.py)

**位置**：`adapters/models/factory.py`

#### 功能
- 根据配置创建 AI 模型实例
- 支持多个提供商
- 统一的模型接口

#### 支持的提供商

```python
class ModelProvider(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    SILICONFLOW = "siliconflow"
    OPENAI_COMPATIBLE = "openai-compatible"
```

#### 核心方法

```python
def create_model_for_agent(agent_name: str) -> Any:
    """
    根据智能体名称创建模型。
    
    加载顺序：
    1. 智能体特定配置（configs/agents/{agent_name}.yaml）
    2. 提供商配置（configs/providers/{provider}.yaml）
    3. 环境变量
    
    返回：
    - Agno Model 实例
    """
    config_manager = get_config_manager()
    agent_config = config_manager.get_agent_config(agent_name)
    
    model_config = agent_config.get("model", {})
    provider = model_config.get("provider", "openai")
    model_name = model_config.get("name", "gpt-4")
    
    if provider == ModelProvider.OPENAI:
        from agno.models import OpenAIChat
        return OpenAIChat(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    elif provider == ModelProvider.GOOGLE:
        from agno.models import Gemini
        return Gemini(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    elif provider == ModelProvider.DEEPSEEK:
        from agno.models import OpenAIChat
        return OpenAIChat(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
    
    # ... 其他提供商
```

#### 配置示例

**智能体配置**（`configs/agents/research_agent.yaml`）：

```yaml
name: research_agent
model:
  provider: openai
  name: gpt-4
  temperature: 0.7
  max_tokens: 4000
```

**提供商配置**（`configs/providers/openai.yaml`）：

```yaml
provider: openai
api_key_env: OPENAI_API_KEY
base_url_env: OPENAI_BASE_URL
default_model: gpt-4
models:
  - gpt-4
  - gpt-4-turbo
  - gpt-3.5-turbo
```

## 使用示例

### 获取股票价格

```python
from valuecell.adapters.assets import get_adapter_manager
from valuecell.adapters.assets.types import Exchange

manager = get_adapter_manager()

# 单个价格
price = manager.get_current_price("AAPL", Exchange.NASDAQ)
print(f"AAPL: ${price.price}")

# 批量价格
symbols = ["AAPL", "MSFT", "GOOGL"]
prices = manager.get_batch_prices(symbols, Exchange.NASDAQ)
for symbol, price in prices.items():
    print(f"{symbol}: ${price.price}")
```

### 获取历史数据

```python
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

data = manager.get_historical_data(
    symbol="AAPL",
    exchange=Exchange.NASDAQ,
    start_date=start_date,
    end_date=end_date,
    interval="1d"
)

for ohlcv in data:
    print(f"{ohlcv.timestamp}: Open={ohlcv.open}, Close={ohlcv.close}")
```

### 搜索资产

```python
from valuecell.adapters.assets.types import AssetSearchQuery, AssetType

query = AssetSearchQuery(
    keyword="apple",
    asset_type=AssetType.STOCK
)

results = manager.search_assets(query)
for result in results:
    print(f"{result.symbol}: {result.name}")
```

## 扩展指南

### 添加新数据源

1. 继承 `BaseDataAdapter`
2. 实现所有抽象方法
3. 定义能力（`get_capabilities`）
4. 注册到管理器

```python
class MyAdapter(BaseDataAdapter):
    def __init__(self):
        super().__init__(DataSource.CUSTOM)
    
    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                asset_type=AssetType.STOCK,
                exchanges=[Exchange.NYSE],
                features=["real_time_price"]
            )
        ]
    
    def get_current_price(self, symbol, exchange):
        # 实现
        pass

# 注册
manager = get_adapter_manager()
manager.register_adapter(MyAdapter())
```

### 添加新模型提供商

在 `factory.py` 中添加新的分支：

```python
elif provider == "my_provider":
    from my_sdk import MyModel
    return MyModel(
        model=model_name,
        api_key=os.getenv("MY_API_KEY")
    )
```

## 性能优化

### 1. 缓存策略
- Ticker → Adapter 映射缓存
- 减少适配器查找开销

### 2. 并行请求
- 使用线程池批量获取
- 最多 10 个并发

### 3. 延迟加载
- AKShare 延迟导入
- 避免启动时加载大型库

### 4. 故障转移
- 多个适配器自动切换
- 提高可用性

## 错误处理

### 1. 适配器级别
- 捕获特定异常
- 返回 None 而非抛出异常
- 记录警告日志

### 2. 管理器级别
- 尝试多个适配器
- 故障转移
- 最终返回 None

### 3. 应用级别
- 检查返回值
- 提供降级方案
- 用户友好的错误消息

## 测试

### 单元测试

```python
def test_yfinance_adapter():
    adapter = YFinanceAdapter()
    price = adapter.get_current_price("AAPL", Exchange.NASDAQ)
    assert price is not None
    assert price.symbol == "AAPL"
    assert price.price > 0
```

### 集成测试

```python
def test_adapter_manager_routing():
    manager = AdapterManager()
    manager.configure_yfinance()
    manager.configure_akshare()
    
    # 测试美股路由
    price = manager.get_current_price("AAPL", Exchange.NASDAQ)
    assert price is not None
    
    # 测试 A 股路由
    price = manager.get_current_price("600000", Exchange.SSE)
    assert price is not None
```

---

**相关文档**：
- [智能体层](./03_agents.md)
- [配置层](./05_config.md)
- [服务器层](./01_server.md)

