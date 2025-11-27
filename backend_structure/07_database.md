# 数据库层 (Database)

## 概述

数据库层负责数据持久化，使用 SQLAlchemy 2.0+ 异步 ORM，支持 SQLite 和其他关系型数据库。采用仓储模式封装数据访问逻辑，提供清晰的数据操作接口。

## 目录结构

```
server/db/
├── __init__.py
├── connection.py       # 数据库连接管理
├── init_db.py          # 数据库初始化
├── models/             # SQLAlchemy 模型
│   ├── __init__.py
│   ├── base.py         # 基础模型类
│   ├── agent.py        # 智能体模型
│   ├── asset.py        # 资产模型
│   ├── strategy.py     # 策略模型
│   ├── strategy_detail.py      # 策略详情
│   ├── strategy_holding.py     # 持仓记录
│   ├── strategy_portfolio.py  # 投资组合
│   ├── strategy_instruction.py # 交易指令
│   ├── strategy_compose_cycle.py # 组合周期
│   ├── strategy_prompt.py      # 策略提示
│   ├── user_profile.py         # 用户配置
│   └── watchlist.py            # 观察列表
└── repositories/       # 仓储层
    ├── __init__.py
    ├── agent_repository.py
    ├── strategy_repository.py
    ├── user_profile_repository.py
    └── watchlist_repository.py
```

## 数据库连接

### connection.py - 连接管理

**位置**：`server/db/connection.py`

#### 核心组件

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, StaticPool
from typing import AsyncGenerator

# 全局引擎和会话工厂
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

def get_database_url() -> str:
    """
    获取数据库 URL。
    
    优先级：
    1. 环境变量 DATABASE_URL
    2. 默认值 sqlite+aiosqlite:///./valuecell.db
    
    返回：
    - str: 数据库 URL
    """
    from valuecell.utils.env import get_env
    return get_env(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./valuecell.db"
    )

def create_engine() -> AsyncEngine:
    """
    创建数据库引擎。
    
    配置：
    - SQLite: 使用 StaticPool（单文件数据库）
    - 其他: 使用默认连接池
    - echo: 根据环境变量 SQL_ECHO 控制
    
    返回：
    - AsyncEngine: 异步引擎
    """
    url = get_database_url()
    is_sqlite = "sqlite" in url
    
    from valuecell.utils.env import get_env_bool
    echo = get_env_bool("SQL_ECHO", False)
    
    if is_sqlite:
        # SQLite 特殊配置
        engine = create_async_engine(
            url,
            echo=echo,
            poolclass=StaticPool,  # SQLite 单连接池
            connect_args={"check_same_thread": False}
        )
    else:
        # MySQL / PostgreSQL 等
        engine = create_async_engine(
            url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True  # 连接健康检查
        )
    
    return engine

def get_engine() -> AsyncEngine:
    """
    获取全局引擎单例。
    
    返回：
    - AsyncEngine: 异步引擎
    """
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine

def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    获取会话工厂。
    
    返回：
    - async_sessionmaker: 会话工厂
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False  # 避免访问已提交对象时重新查询
        )
    return _session_factory

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话（依赖注入用）。
    
    用法（在 FastAPI 路由中）：
    @app.get("/items")
    async def get_items(session: AsyncSession = Depends(get_session)):
        ...
    
    返回：
    - AsyncSession: 异步会话
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def close_engine():
    """关闭数据库引擎（应用关闭时调用）"""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
```

### init_db.py - 数据库初始化

**位置**：`server/db/init_db.py`

#### 功能
- 创建所有表
- 初始化基础数据
- 数据库迁移（预留）

```python
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from .connection import get_engine
from .models import Base

async def create_tables(engine: AsyncEngine):
    """
    创建所有表。
    
    参数：
    - engine: 异步引擎
    """
    async with engine.begin() as conn:
        # 创建所有表
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

async def init_default_data(engine: AsyncEngine):
    """
    初始化默认数据。
    
    参数：
    - engine: 异步引擎
    """
    # TODO: 插入默认数据（如果需要）
    pass

def init_database(force: bool = False) -> bool:
    """
    同步初始化数据库（用于应用启动）。
    
    参数：
    - force: 是否强制重建（删除现有表）
    
    返回：
    - bool: 是否成功
    """
    import asyncio
    
    try:
        engine = get_engine()
        
        if force:
            # 警告：会删除所有数据！
            async def drop_and_create():
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.drop_all)
                    await conn.run_sync(Base.metadata.create_all)
            
            asyncio.run(drop_and_create())
            logger.warning("Database forcefully recreated (all data lost)")
        else:
            # 仅创建不存在的表
            asyncio.run(create_tables(engine))
        
        # 初始化默认数据
        asyncio.run(init_default_data(engine))
        
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
```

## 数据模型

### base.py - 基础模型

**位置**：`server/db/models/base.py`

#### Base 类

```python
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, String, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

class Base(DeclarativeBase):
    """所有模型的基类"""
    pass

class TimestampMixin:
    """时间戳混入类"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间"
    )

class SoftDeleteMixin:
    """软删除混入类"""
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否删除"
    )
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="删除时间"
    )
```

### 模型示例

#### strategy.py - 策略模型

```python
from decimal import Decimal
from typing import Optional
from sqlalchemy import String, Text, JSON, Numeric, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin

import enum

class StrategyStatus(str, enum.Enum):
    """策略状态枚举"""
    DRAFT = "draft"           # 草稿
    ACTIVE = "active"         # 活跃
    PAUSED = "paused"         # 暂停
    STOPPED = "stopped"       # 停止
    ERROR = "error"           # 错误

class Strategy(Base, TimestampMixin):
    """策略模型"""
    __tablename__ = "strategies"
    
    # 主键
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # 基本信息
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # 配置
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    
    # 状态
    status: Mapped[StrategyStatus] = mapped_column(
        SQLEnum(StrategyStatus),
        default=StrategyStatus.DRAFT,
        nullable=False,
        index=True
    )
    
    # 时间
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # 财务
    initial_capital: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0")
    )
    
    # 关系
    holdings = relationship("StrategyHolding", back_populates="strategy")
    portfolio = relationship("StrategyPortfolio", back_populates="strategy", uselist=False)
    
    def __repr__(self):
        return f"<Strategy(id={self.id}, name={self.name}, status={self.status})>"
```

#### strategy_holding.py - 持仓模型

```python
from decimal import Decimal
from sqlalchemy import String, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin

class StrategyHolding(Base, TimestampMixin):
    """策略持仓模型"""
    __tablename__ = "strategy_holdings"
    
    # 主键
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # 外键
    strategy_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("strategies.id"),
        nullable=False,
        index=True
    )
    
    # 资产信息
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # 持仓数据
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    avg_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # 盈亏
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    
    # 关系
    strategy = relationship("Strategy", back_populates="holdings")
    
    def __repr__(self):
        return f"<StrategyHolding(symbol={self.symbol}, quantity={self.quantity})>"
```

#### user_profile.py - 用户配置模型

```python
from sqlalchemy import String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin

class UserProfile(Base, TimestampMixin):
    """用户配置模型"""
    __tablename__ = "user_profiles"
    
    # 主键（用户 ID）
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # 偏好设置
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    language: Mapped[str] = mapped_column(String(10), default="en-US")
    
    # 其他配置
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, language={self.language})>"
```

#### watchlist.py - 观察列表模型

```python
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin

class Watchlist(Base, TimestampMixin):
    """观察列表模型"""
    __tablename__ = "watchlist"
    
    # 主键
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # 用户
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # 资产信息
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # 约束：同一用户不能添加相同的资产两次
    __table_args__ = (
        UniqueConstraint('user_id', 'symbol', 'exchange', name='uq_user_symbol_exchange'),
    )
    
    def __repr__(self):
        return f"<Watchlist(symbol={self.symbol}, exchange={self.exchange})>"
```

## 仓储层

### 仓储模式

仓储模式封装数据访问逻辑，提供清晰的业务接口。

#### 基础仓储类

```python
from typing import Generic, TypeVar, List, Optional, Type
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """基础仓储类"""
    
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """根据 ID 获取"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self) -> List[T]:
        """获取所有"""
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())
    
    async def create(self, instance: T) -> T:
        """创建"""
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance
    
    async def update(self, instance: T) -> T:
        """更新"""
        await self.session.merge(instance)
        await self.session.flush()
        return instance
    
    async def delete(self, id: str) -> bool:
        """删除"""
        await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        await self.session.flush()
        return True
```

#### strategy_repository.py - 策略仓储

```python
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.strategy import Strategy, StrategyStatus
from ..models.strategy_holding import StrategyHolding
from ..models.strategy_portfolio import StrategyPortfolio

class StrategyRepository:
    """策略仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """获取策略（包含关联数据）"""
        result = await self.session.execute(
            select(Strategy)
            .where(Strategy.id == strategy_id)
            .options(
                selectinload(Strategy.holdings),
                selectinload(Strategy.portfolio)
            )
        )
        return result.scalar_one_or_none()
    
    async def get_by_user(
        self,
        user_id: str,
        status: Optional[StrategyStatus] = None
    ) -> List[Strategy]:
        """获取用户的策略"""
        query = select(Strategy).where(Strategy.user_id == user_id)
        
        if status:
            query = query.where(Strategy.status == status)
        
        query = query.order_by(Strategy.created_at.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def create(self, strategy: Strategy) -> Strategy:
        """创建策略"""
        self.session.add(strategy)
        await self.session.flush()
        await self.session.refresh(strategy)
        return strategy
    
    async def update_status(
        self,
        strategy_id: str,
        status: StrategyStatus
    ) -> bool:
        """更新状态"""
        strategy = await self.get_by_id(strategy_id)
        if strategy:
            strategy.status = status
            await self.session.flush()
            return True
        return False
    
    async def get_holdings(
        self,
        strategy_id: str
    ) -> List[StrategyHolding]:
        """获取持仓"""
        result = await self.session.execute(
            select(StrategyHolding)
            .where(StrategyHolding.strategy_id == strategy_id)
        )
        return list(result.scalars().all())
    
    async def save_holdings(
        self,
        holdings: List[StrategyHolding]
    ):
        """批量保存持仓"""
        self.session.add_all(holdings)
        await self.session.flush()

def get_strategy_repository(
    session: AsyncSession
) -> StrategyRepository:
    """获取策略仓储（依赖注入）"""
    return StrategyRepository(session)
```

## 查询优化

### 1. 预加载关联数据

```python
# 使用 selectinload 预加载
result = await session.execute(
    select(Strategy)
    .options(selectinload(Strategy.holdings))
    .where(Strategy.id == strategy_id)
)
```

### 2. 批量插入

```python
# 使用 add_all 批量插入
holdings = [StrategyHolding(...) for _ in range(100)]
session.add_all(holdings)
await session.flush()
```

### 3. 索引优化

```python
# 在模型中定义索引
user_id: Mapped[str] = mapped_column(
    String(64),
    nullable=False,
    index=True  # 添加索引
)
```

### 4. 分页查询

```python
async def get_paginated(
    self,
    page: int = 1,
    page_size: int = 20
) -> List[Strategy]:
    """分页查询"""
    offset = (page - 1) * page_size
    result = await self.session.execute(
        select(Strategy)
        .limit(page_size)
        .offset(offset)
    )
    return list(result.scalars().all())
```

## 事务管理

### 手动事务

```python
async def complex_operation():
    """复杂操作（需要事务）"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            # 操作 1
            strategy = Strategy(...)
            session.add(strategy)
            
            # 操作 2
            holdings = [StrategyHolding(...)]
            session.add_all(holdings)
            
            # 提交
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 嵌套事务（Savepoint）

```python
async def nested_operation():
    """嵌套事务"""
    factory = get_session_factory()
    async with factory() as session:
        # 外层事务
        strategy = Strategy(...)
        session.add(strategy)
        
        # 内层事务（Savepoint）
        async with session.begin_nested():
            holdings = [StrategyHolding(...)]
            session.add_all(holdings)
            # 如果这里失败，只回滚 holdings，不影响 strategy
        
        await session.commit()
```

## 数据库迁移

### 使用 Alembic（推荐）

```bash
# 安装
pip install alembic

# 初始化
alembic init migrations

# 生成迁移
alembic revision --autogenerate -m "Add new field"

# 应用迁移
alembic upgrade head

# 回滚
alembic downgrade -1
```

### 配置 Alembic

**alembic.ini**:
```ini
sqlalchemy.url = sqlite+aiosqlite:///./valuecell.db
```

**env.py**:
```python
from valuecell.server.db.models import Base

target_metadata = Base.metadata
```

## 测试

### 使用内存数据库

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest.fixture
async def test_session():
    """测试数据库会话"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 创建会话
    async with AsyncSession(engine) as session:
        yield session
    
    # 清理
    await engine.dispose()

@pytest.mark.asyncio
async def test_create_strategy(test_session):
    """测试创建策略"""
    repo = StrategyRepository(test_session)
    
    strategy = Strategy(
        id="test123",
        user_id="user1",
        name="Test Strategy",
        agent_name="grid_agent"
    )
    
    created = await repo.create(strategy)
    assert created.id == "test123"
    
    # 验证
    fetched = await repo.get_by_id("test123")
    assert fetched is not None
    assert fetched.name == "Test Strategy"
```

## 性能优化建议

### 1. 连接池
- 根据负载调整 `pool_size` 和 `max_overflow`
- 启用 `pool_pre_ping` 检测失效连接

### 2. 查询优化
- 使用索引加速查询
- 避免 N+1 查询（使用 selectinload）
- 只查询需要的字段

### 3. 批量操作
- 使用 `add_all` 批量插入
- 使用 `bulk_insert_mappings` 更快插入

### 4. 缓存
- 对不常变化的数据进行缓存
- 使用 Redis 缓存查询结果

---

**相关文档**：
- [服务器层](./01_server.md)
- [工具层](./06_utils.md)
- [配置层](./05_config.md)

