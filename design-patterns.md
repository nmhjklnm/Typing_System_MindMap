# Python 设计模式

> **说明**：本文档整理常用的设计模式及其在 Python 中的实现。设计模式是软件开发中反复出现问题的通用解决方案。

---

## **设计模式分类**

设计模式通常分为三大类：

1. **创建型模式（Creational）**：关注对象的创建机制
2. **结构型模式（Structural）**：关注类和对象的组合
3. **行为型模式（Behavioral）**：关注对象之间的通信和职责分配

---

## **1. 创建型模式**

### [x] **工厂模式（Factory Pattern）**

**概念**：用一个工厂类来创建对象，调用者不需要知道具体类名。根据配置或参数动态选择实现。

**为什么需要**：
- 解耦对象的创建和使用
- 根据配置动态选择实现
- 便于扩展新的实现类型

**核心要素**：
- 统一的接口/协议（返回类型）
- 工厂方法（接收参数，返回实例）
- 多个具体实现类

**应用场景**：
- 数据库连接：根据配置选择 MySQL/PostgreSQL/SQLite
- 队列系统：根据环境选择内存/Redis/SQLite 队列
- 日志系统：根据级别选择不同的日志处理器
- 支付系统：根据用户选择不同的支付方式

**Python 特性结合**：
- 使用 `Literal` 类型限制参数值
- 使用 `Protocol` 定义返回类型
- 使用 `TypeAlias` 提供类型别名

**example**：
```python
from typing import Literal, Protocol

# 1. 定义协议（统一接口）
class TaskQueue(Protocol):
    async def add_task(self, task: Task) -> None: ...
    async def get_pending_tasks(self) -> list[Task]: ...

# 2. 类型别名：限制队列类型
QueueType = Literal["sqlite", "redis", "memory"]

# 3. 工厂类
class QueueFactory:
    @staticmethod
    def create(queue_type: QueueType, **kwargs) -> TaskQueue:
        """根据类型创建队列"""
        if queue_type == "sqlite":
            return SQLiteTaskQueue(**kwargs)
        elif queue_type == "redis":
            return RedisTaskQueue(**kwargs)
        elif queue_type == "memory":
            return InMemoryTaskQueue()
        else:
            raise ValueError(f"Unsupported: {queue_type}")

# 4. 使用
# 开发环境
dev_queue = QueueFactory.create("memory")
# 生产环境
prod_queue = QueueFactory.create("sqlite", db_path="prod.db")
```

---

### [x] **单例模式（Singleton Pattern）**

**概念**：确保一个类只有一个实例，并提供全局访问点。

**为什么需要**：
- 全局资源管理：避免重复创建
- 状态共享：多处访问同一实例
- 节省资源：如数据库连接池、配置管理器

**核心要素**：
- 私有化实例变量（类变量存储唯一实例）
- 控制实例创建（重写 `__new__` 或使用装饰器）
- 全局访问点

**应用场景**：
- 数据库连接池
- 配置管理器（只加载一次配置文件）
- 日志记录器
- 线程池、进程池

**实现方式**：
1. 重写 `__new__` 方法
2. 使用装饰器
3. 使用元类

**example**：
```python
# 方式 1：重写 __new__
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.connection = "db_connection"
        self._initialized = True

# 方式 2：装饰器实现
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.settings = {"debug": True}

# 使用
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True
```

---

### [x] **建造者模式（Builder Pattern）**

**概念**：将复杂对象的构建过程分步骤进行，使用者可以按需构建。通过链式调用逐步设置参数，最后调用 `build()` 生成对象。

**为什么需要**：
- 构造函数参数太多，不易理解
- 对象构建过程复杂，需要多个步骤
- 同样的构建过程可以创建不同的表示

**核心要素**：
- 建造者类（Builder）：提供构建方法
- 链式调用（返回 self）
- 最终构建方法（build()）

**应用场景**：
- SQL 查询构建器（逐步添加 SELECT、WHERE、ORDER BY）
- HTTP 请求构建器（设置 URL、headers、body）
- 配置对象构建（逐步设置各种选项）
- UI 组件构建（设置样式、属性、事件）

**Python 特性结合**：
- 链式调用（方法返回 `self`）
- 类型提示（返回 `"QueryBuilder"` 确保链式调用类型正确）
- 最终验证（build 方法中检查必需参数）

**example**：
```python
class QueryBuilder:
    def __init__(self):
        self._table: Optional[str] = None
        self._fields: list[str] = []
        self._where: Optional[str] = None
        self._limit: Optional[int] = None
    
    def table(self, name: str) -> "QueryBuilder":
        """设置表名（返回 self 支持链式调用）"""
        self._table = name
        return self
    
    def select(self, *fields: str) -> "QueryBuilder":
        self._fields.extend(fields)
        return self
    
    def where(self, condition: str) -> "QueryBuilder":
        self._where = condition
        return self
    
    def limit(self, count: int) -> "QueryBuilder":
        self._limit = count
        return self
    
    def build(self) -> str:
        """构建 SQL 查询"""
        if not self._table:
            raise ValueError("必须指定表名")
        
        sql = f"SELECT {', '.join(self._fields or ['*'])} FROM {self._table}"
        if self._where:
            sql += f" WHERE {self._where}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql

# 使用：链式调用
query = (
    QueryBuilder()
    .table("users")
    .select("id", "name", "email")
    .where("age > 18")
    .limit(10)
    .build()
)
print(query)  # SELECT id, name, email FROM users WHERE age > 18 LIMIT 10
```

---

## **2. 结构型模式**

### [x] **Mixin 模式**

**概念**：通过多重继承，将特定功能"混入"到类中，实现代码复用。Mixin 类通常不单独实例化，只提供某些方法或属性。

**为什么需要**：
- 多个类需要相同的功能，但不适合用继承
- 避免重复代码
- 功能模块化，可灵活组合

**核心要素**：
- Mixin 类：提供特定功能（如时间戳、序列化）
- 主类：通过多重继承混入功能
- 命名约定：Mixin 类名通常以 Mixin 结尾

**应用场景**：
- 添加时间戳（CreatedAt、UpdatedAt）
- 添加序列化功能（to_dict、from_dict）
- 添加日志功能
- 添加缓存功能

**Python 特性结合**：
- 多重继承（`class Model(TimestampMixin, SerializeMixin)`）
- 方法解析顺序（MRO）
- 类型提示（Protocol 确保 Mixin 正确使用）

**example**：
```python
from datetime import datetime

class TimestampMixin:
    """时间戳 Mixin"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        """更新修改时间"""
        self.updated_at = datetime.now()

class SerializeMixin:
    """序列化 Mixin"""
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}

# 使用 Mixin
class User(TimestampMixin, SerializeMixin):
    def __init__(self, name: str, email: str):
        super().__init__()  # 调用 Mixin 的 __init__
        self.name = name
        self.email = email

user = User("Alice", "alice@example.com")
print(user.created_at)  # 自动有时间戳
print(user.to_dict())   # 自动有序列化
```

---

### [x] **协议模式（Protocol Pattern）**

**概念**：定义一组方法签名（接口），任何实现了这些方法的类都满足该协议。基于"结构化类型"（鸭子类型），不需要显式继承。

**为什么需要**：
- Python 的"鸭子类型"：如果它走路像鸭子、叫声像鸭子，那它就是鸭子
- 不需要继承，更灵活
- 静态类型检查：mypy 可以验证是否符合协议
- 可以给第三方类"追加"协议支持

**核心要素**：
- 使用 `typing.Protocol` 定义协议
- 定义方法签名（方法体用 `...`）
- 实现类无需继承协议类
- mypy 自动检查实现是否符合

**应用场景**：
- 定义接口规范（如任务队列、数据库连接）
- 替代抽象基类（ABC）
- 类型约束（函数参数必须实现某协议）
- 第三方库集成

**与 ABC 的区别**：
- Protocol：结构化类型，基于方法签名
- ABC：名义类型，必须显式继承

**example**：
```python
from typing import Protocol, Optional

# 1. 定义协议
class TaskQueue(Protocol):
    """任务队列协议（接口规范）"""
    async def add_task(self, task: Task) -> None: ...
    async def get_task(self, task_id: str) -> Optional[Task]: ...

# 2. 实现类（无需继承）
class MemoryQueue:
    async def add_task(self, task: Task) -> None:
        # 实现...
        pass
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        # 实现...
        return None

# 3. 类型检查
async def process_queue(queue: TaskQueue):
    """接受任何实现了 TaskQueue 协议的对象"""
    await queue.add_task(task)  # ✅ mypy 知道有 add_task 方法
    
# MemoryQueue 无需继承 TaskQueue，mypy 也能识别
memory_queue = MemoryQueue()
await process_queue(memory_queue)  # ✅ 类型检查通过
```

---

### [x] **适配器模式（Adapter Pattern）**

**概念**：将一个类的接口转换成客户期望的另一个接口。让原本不兼容的类可以一起工作。

**为什么需要**：
- 集成第三方库，但接口不匹配
- 复用旧代码，但接口已过时
- 统一不同来源的数据格式

**核心要素**：
- 目标接口（期望的接口）
- 被适配者（需要适配的类）
- 适配器（实现目标接口，内部调用被适配者）

**应用场景**：
- 统一不同日志库的接口
- 统一不同数据库的接口
- 统一不同支付接口
- 数据格式转换（JSON ↔ XML）

**实现方式**：
1. 对象适配器（组合）：推荐
2. 类适配器（多重继承）

**example**：
```python
from typing import Protocol

# 1. 目标接口：我们期望的统一接口
class Logger(Protocol):
    def log(self, level: str, message: str) -> None: ...

# 2. 被适配者 A（接口不同）
class ThirdPartyLoggerA:
    def write_log(self, msg: str, severity: int) -> None:
        print(f"[LoggerA] Severity {severity}: {msg}")

# 3. 适配器 A：将 ThirdPartyLoggerA 适配为 Logger
class LoggerAdapterA:
    def __init__(self, logger: ThirdPartyLoggerA):
        self.logger = logger
    
    def log(self, level: str, message: str) -> None:
        # 转换：level 字符串 -> severity 数字
        severity_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        severity = severity_map.get(level.upper(), 1)
        self.logger.write_log(message, severity)

# 4. 使用统一接口
def use_logger(logger: Logger):
    logger.log("INFO", "应用启动")
    logger.log("ERROR", "发生错误")

# 适配第三方库
adapter = LoggerAdapterA(ThirdPartyLoggerA())
use_logger(adapter)  # ✅ 统一接口
```

---

### [x] **外观模式（Facade Pattern）**

**概念**：为复杂的子系统提供一个简单的统一接口。隐藏系统的复杂性，提供一个高层接口让子系统更易使用。

**为什么需要**：
- 简化复杂 API：对外提供简单接口
- 隐藏底层细节：用户不需要了解内部实现
- 降低耦合：子系统变化不影响外部使用
- 提高可读性：一行代码完成复杂操作

**核心要素**：
- 复杂的子系统（多个类协同工作）
- 外观类（封装子系统，提供简单接口）
- 客户端只与外观交互

**应用场景**：
- 视频处理：转换需要协调解码器、编码器、音频处理等
- 订单处理：创建订单需要库存检查、支付、发货等多步骤
- 邮件发送：组装邮件需要 SMTP、MIME、附件处理等
- SDK 封装：将复杂的第三方 API 封装成简单接口

**与适配器的区别**：
- **适配器**：接口转换，一对一（让不兼容的接口协同工作）
- **外观**：简化接口，一对多（为复杂子系统提供简单入口）

**Python 特性结合**：
- 使用类方法封装复杂流程
- 可以结合工厂模式创建外观
- 返回类型使用简单类型

**example**：
```python
# 复杂的子系统
class VideoFile:
    """视频文件类"""
    def __init__(self, filename: str):
        self.filename = filename
        self.codec_type = self._detect_codec()
    
    def _detect_codec(self) -> str:
        # 复杂的编解码器检测逻辑
        return "h264"

class AudioMixer:
    """音频混合器"""
    def fix(self, video: VideoFile) -> str:
        # 复杂的音频处理逻辑
        return "fixed_audio"

class BitrateReader:
    """比特率读取器"""
    def read(self, video: VideoFile) -> int:
        # 复杂的比特率计算
        return 1920

class CodecFactory:
    """编解码器工厂"""
    @staticmethod
    def extract(video: VideoFile) -> str:
        # 根据视频类型提取编解码器
        return video.codec_type

# 外观类：简化接口
class VideoConverter:
    """视频转换外观
    
    隐藏了视频转换的所有复杂细节：
    - 编解码器检测和提取
    - 音频处理
    - 比特率调整
    - 格式转换
    
    对外只提供一个简单的 convert 方法
    """
    
    def convert(self, filename: str, target_format: str) -> str:
        """转换视频格式（一行代码搞定）
        
        Args:
            filename: 源文件名
            target_format: 目标格式（"mp4", "avi", "mkv" 等）
        
        Returns:
            转换后的文件名
        """
        # 内部协调多个复杂子系统
        print(f"开始转换 {filename} -> {target_format}")
        
        # 1. 加载视频
        video = VideoFile(filename)
        
        # 2. 提取编解码器
        codec = CodecFactory.extract(video)
        print(f"检测到编解码器: {codec}")
        
        # 3. 处理音频
        audio = AudioMixer().fix(video)
        print(f"音频处理完成: {audio}")
        
        # 4. 读取比特率
        bitrate = BitrateReader().read(video)
        print(f"比特率: {bitrate}")
        
        # 5. 执行转换（实际项目中会调用 ffmpeg 等）
        output = filename.rsplit(".", 1)[0] + f".{target_format}"
        print(f"转换完成: {output}")
        
        return output

# 使用：非常简单！
converter = VideoConverter()

# ✅ 一行代码完成复杂的视频转换
converter.convert("movie.avi", "mp4")
converter.convert("video.mkv", "avi")

# 如果没有外观，用户需要：
# video = VideoFile("movie.avi")
# codec = CodecFactory.extract(video)
# audio = AudioMixer().fix(video)
# bitrate = BitrateReader().read(video)
# ... 自己协调所有步骤
```

**实际应用示例：订单处理外观**
```python
class OrderFacade:
    """订单处理外观"""
    
    def __init__(self):
        # 内部依赖多个子系统
        self.inventory = InventoryService()
        self.payment = PaymentService()
        self.shipping = ShippingService()
        self.notification = NotificationService()
    
    def place_order(self, user_id: str, items: list[str]) -> str:
        """下单（一行代码完成）
        
        内部协调：
        1. 检查库存
        2. 处理支付
        3. 创建发货单
        4. 发送通知
        """
        # 1. 检查库存
        if not self.inventory.check_available(items):
            raise ValueError("库存不足")
        
        # 2. 处理支付
        order_id = self.payment.charge(user_id, items)
        
        # 3. 减少库存
        self.inventory.decrease(items)
        
        # 4. 创建发货单
        self.shipping.create_shipment(order_id, items)
        
        # 5. 发送通知
        self.notification.send_order_confirmation(user_id, order_id)
        
        return order_id

# 使用：外部调用非常简单
facade = OrderFacade()
order_id = facade.place_order("user-123", ["item-1", "item-2"])
```

---

### [x] **依赖注入（Dependency Injection）** ⭐⭐⭐

**概念**：不在类内部创建依赖对象，而是从外部传入（注入）。将依赖关系的创建和使用分离。

**为什么需要**：
- **解耦**：类不依赖具体实现，只依赖接口
- **可测试**：可以注入 mock 对象进行单元测试
- **灵活**：运行时可以切换不同的实现
- **符合依赖倒置原则**：高层模块不依赖低层模块，都依赖抽象

**核心要素**：
- 依赖接口（Protocol）：定义依赖的规范
- 依赖实现：具体的实现类
- 注入方式：通过构造函数、setter 或属性注入

**注入方式**：
1. **构造函数注入**（推荐）：在 `__init__` 中传入依赖
2. **属性注入**：直接设置属性
3. **方法注入**：通过方法参数传入

**应用场景**：
- 服务层依赖数据访问层
- 控制器依赖业务服务
- 测试时注入 mock 对象
- 配置不同环境的实现

**设计原则本质**：
依赖注入本质上是**依赖倒置原则（DIP）**的实现方式，是一种**设计原则**，但因其重要性和普遍性，通常作为结构型模式讨论。

**Python 特性结合**：
- 使用 Protocol 定义依赖接口
- 类型提示让 IDE 可以检查依赖
- 可以结合工厂模式创建依赖

**你的 queue 项目已经在用了！**

**example**：
```python
from typing import Protocol

# 1. 定义依赖接口
class TaskQueue(Protocol):
    """任务队列接口（依赖抽象）"""
    async def add_task(self, task: Task) -> None: ...
    async def get_pending_tasks(self) -> list[Task]: ...

# 2. 具体实现
class SQLiteTaskQueue:
    async def add_task(self, task: Task) -> None:
        # SQLite 实现
        pass
    
    async def get_pending_tasks(self) -> list[Task]:
        # SQLite 实现
        return []

# 3. 依赖注入（构造函数注入）
class TaskWorker:
    """任务工作器
    
    ✅ 好的设计：依赖从外部注入
    - 不关心 queue 是 SQLite 还是 Redis
    - 可以轻松切换实现
    - 可以注入 mock 对象测试
    """
    
    def __init__(self, queue: TaskQueue):  # 🔥 依赖注入！
        self.queue = queue  # 从外部传入，不在内部创建
    
    async def process(self):
        """处理任务"""
        tasks = await self.queue.get_pending_tasks()
        for task in tasks:
            # 处理任务...
            pass

# 4. 使用：在外部创建依赖并注入
# 生产环境
prod_queue = SQLiteTaskQueue(db_path="prod.db")
prod_worker = TaskWorker(prod_queue)  # 注入 SQLite 队列

# 测试环境
class MockQueue:
    """测试用的 mock 队列"""
    async def add_task(self, task: Task) -> None:
        pass
    
    async def get_pending_tasks(self) -> list[Task]:
        return [Task(id="test-1", name="测试任务")]

test_queue = MockQueue()
test_worker = TaskWorker(test_queue)  # 注入 mock 队列
```

**对比：不使用依赖注入（坏的设计）**
```python
class TaskWorker:
    """❌ 坏的设计：内部创建依赖"""
    
    def __init__(self):
        self.queue = SQLiteTaskQueue()  # ❌ 硬编码依赖
        # 问题：
        # 1. 无法切换到 Redis
        # 2. 无法测试（无法注入 mock）
        # 3. 与 SQLiteTaskQueue 强耦合
```

**测试时的优势**
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_process_tasks():
    """测试任务处理（使用 mock 依赖）"""
    
    # 创建 mock 队列
    mock_queue = AsyncMock()
    mock_queue.get_pending_tasks.return_value = [
        Task(id="test-1", name="测试任务")
    ]
    
    # 注入 mock 依赖
    worker = TaskWorker(mock_queue)
    await worker.process()
    
    # 验证
    assert mock_queue.get_pending_tasks.called
```

---

### [x] **仓储模式（Repository Pattern）** ⭐⭐

**概念**：将数据访问逻辑封装在仓储类中，为业务层提供类似集合（Collection）的接口来操作数据。业务层不直接操作数据库。

**为什么需要**：
- **抽象数据访问**：业务层不关心数据存储在哪里（MySQL、MongoDB、内存）
- **集中管理查询**：所有数据访问逻辑集中在一处
- **便于测试**：可以使用内存仓储替代数据库
- **易于切换存储**：从 SQLite 迁移到 PostgreSQL 只需改仓储实现

**核心要素**：
- 仓储接口（Protocol）：定义数据访问方法
- 具体仓储：实现具体的存储逻辑
- 实体对象：业务实体（如 User、Task）
- 业务层只依赖仓储接口

**典型方法**：
- `find_by_id(id)`: 根据 ID 查询
- `find_by_xxx(value)`: 根据属性查询
- `find_all()`: 查询所有
- `save(entity)`: 保存实体
- `delete(id)`: 删除实体

**应用场景**：
- 用户数据访问：UserRepository
- 订单数据访问：OrderRepository
- 任务数据访问：TaskRepository（你的 queue 项目就是这个！）
- 商品数据访问：ProductRepository

**Python 特性结合**：
- 使用 Protocol 定义仓储接口
- 使用 async/await 处理异步数据访问
- 返回领域对象（dataclass、pydantic 模型）

**你的 queue 项目本质上就是仓储模式！**

**example**：
```python
from typing import Protocol, Optional
from dataclasses import dataclass

# 1. 实体对象
@dataclass
class User:
    id: str
    email: str
    name: str

# 2. 仓储接口（Protocol）
class UserRepository(Protocol):
    """用户仓储接口
    
    提供类似集合的操作接口：
    - 查询：find_by_id, find_by_email, find_all
    - 修改：save, delete
    """
    
    async def find_by_id(self, user_id: str) -> Optional[User]: ...
    async def find_by_email(self, email: str) -> Optional[User]: ...
    async def find_all(self, limit: int = 100) -> list[User]: ...
    async def save(self, user: User) -> None: ...
    async def delete(self, user_id: str) -> None: ...

# 3. 具体实现：SQLite 仓储
class SQLiteUserRepository:
    """SQLite 用户仓储"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def find_by_id(self, user_id: str) -> Optional[User]:
        # SELECT * FROM users WHERE id = ?
        return None
    
    async def save(self, user: User) -> None:
        # INSERT OR REPLACE INTO users ...
        pass
    
    # ... 其他方法

# 4. 业务层：只依赖仓储接口
class UserService:
    """用户服务（业务层）"""
    
    def __init__(self, repository: UserRepository):
        self.users = repository  # 依赖注入仓储
    
    async def register(self, email: str, name: str) -> User:
        """注册用户"""
        # 1. 检查邮箱是否存在
        existing = await self.users.find_by_email(email)
        if existing:
            raise ValueError("邮箱已存在")
        
        # 2. 创建用户
        user = User(id=generate_id(), email=email, name=name)
        await self.users.save(user)
        
        return user

# 5. 使用：切换存储非常简单
# 生产环境
sqlite_repo = SQLiteUserRepository("prod.db")
prod_service = UserService(sqlite_repo)

# 测试环境
class InMemoryUserRepository:
    """内存用户仓储（测试用）"""
    def __init__(self):
        self._users: dict[str, User] = {}
    
    async def find_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)
    
    async def save(self, user: User) -> None:
        self._users[user.id] = user

test_repo = InMemoryUserRepository()
test_service = UserService(test_repo)
```

**你的 queue 项目示例**
```python
# TaskQueue 就是一个任务仓储！
class TaskQueue(Protocol):  # 仓储接口
    async def add_task(self, task: Task) -> None: ...  # save
    async def get_task(self, task_id: str) -> Optional[Task]: ...  # find_by_id
    async def get_pending_tasks(self) -> list[Task]: ...  # find_all (with filter)

# 不同的实现（存储后端）
class SQLiteTaskQueue: ...  # SQLite 仓储
class RedisTaskQueue: ...   # Redis 仓储
class InMemoryTaskQueue: ...  # 内存仓储

# 业务层（Worker）只依赖接口
class TaskWorker:
    def __init__(self, queue: TaskQueue):  # 依赖仓储接口
        self.queue = queue
```

---

## **3. 行为型模式**

### [x] **策略模式（Strategy Pattern）**

**概念**：定义一系列算法，把它们封装起来，使它们可以互相替换。算法的变化独立于使用算法的客户。

**为什么需要**：
- 避免大量 if-elif-else
- 算法可以独立变化
- 便于添加新算法
- 运行时动态切换算法

**核心要素**：
- 策略协议（定义算法接口）
- 具体策略（实现不同算法）
- 上下文类（使用策略）

**应用场景**：
- 支付方式：支付宝、微信、银行卡
- 排序算法：快速排序、归并排序、冒泡排序
- 压缩算法：ZIP、RAR、GZIP
- 数据处理：转大写、转小写、反转
- 任务重试策略：固定间隔、指数退避、随机延迟

**Python 特性结合**：
- 使用 Protocol 定义策略接口
- 策略可以是类、函数、lambda
- 运行时切换策略

**example**：
```python
from typing import Protocol

# 1. 策略协议
class DataProcessor(Protocol):
    def process(self, data: str) -> str: ...

# 2. 具体策略
class UpperCaseProcessor:
    def process(self, data: str) -> str:
        return data.upper()

class LowerCaseProcessor:
    def process(self, data: str) -> str:
        return data.lower()

class ReverseProcessor:
    def process(self, data: str) -> str:
        return data[::-1]

# 3. 上下文类：使用策略
class TextEditor:
    def __init__(self, processor: DataProcessor):
        self.processor = processor
    
    def set_processor(self, processor: DataProcessor):
        """运行时切换策略"""
        self.processor = processor
    
    def transform(self, text: str) -> str:
        return self.processor.process(text)

# 使用
text = "Hello World"
editor = TextEditor(UpperCaseProcessor())
print(editor.transform(text))  # HELLO WORLD

editor.set_processor(LowerCaseProcessor())
print(editor.transform(text))  # hello world
```

---

### [x] **观察者模式（Observer Pattern）**

**概念**：定义对象间的一对多依赖，当一个对象状态改变时，所有依赖它的对象都会收到通知并自动更新。

**为什么需要**：
- 解耦：被观察者不需要知道观察者的细节
- 事件驱动：状态改变自动通知
- 动态订阅：可以随时添加/移除观察者
- 一对多关系

**核心要素**：
- 被观察者（Subject）：维护观察者列表，状态改变时通知
- 观察者（Observer）：接收通知并做出反应
- 订阅/取消订阅机制

**应用场景**：
- 任务状态变化监听（完成时发邮件、记日志）
- UI 事件系统（按钮点击、数据更新）
- 消息队列订阅
- 文件系统监听

**Python 特性结合**：
- 使用 Protocol 定义观察者接口
- 使用 `@property.setter` 触发通知
- 可以用装饰器简化订阅

**example**：
```python
from typing import Protocol
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"

# 1. 观察者协议
class TaskObserver(Protocol):
    def on_status_change(
        self, task_id: str, 
        old_status: TaskStatus, 
        new_status: TaskStatus
    ) -> None: ...

# 2. 被观察者
class Task:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._status = TaskStatus.PENDING
        self._observers: list[TaskObserver] = []
    
    def attach(self, observer: TaskObserver):
        """添加观察者"""
        self._observers.append(observer)
    
    def notify(self, old_status: TaskStatus, new_status: TaskStatus):
        """通知所有观察者"""
        for observer in self._observers:
            observer.on_status_change(self.task_id, old_status, new_status)
    
    @property
    def status(self) -> TaskStatus:
        return self._status
    
    @status.setter
    def status(self, new_status: TaskStatus):
        """状态改变时自动通知"""
        if new_status != self._status:
            old_status = self._status
            self._status = new_status
            self.notify(old_status, new_status)

# 3. 具体观察者
class LoggerObserver:
    def on_status_change(self, task_id: str, old: TaskStatus, new: TaskStatus):
        print(f"[日志] {task_id}: {old.value} -> {new.value}")

class EmailObserver:
    def on_status_change(self, task_id: str, old: TaskStatus, new: TaskStatus):
        if new == TaskStatus.COMPLETED:
            print(f"[邮件] 任务 {task_id} 已完成")

# 使用
task = Task("task-001")
task.attach(LoggerObserver())
task.attach(EmailObserver())

task.status = TaskStatus.RUNNING   # 自动通知
task.status = TaskStatus.COMPLETED  # 自动通知
```

---

### [x] **装饰器模式（Decorator Pattern）**

**概念**：动态地给对象添加新功能，而不改变其结构。通过包装的方式扩展功能，符合"开放-封闭原则"。

**为什么需要**：
- 遵循开放-封闭原则：对扩展开放，对修改封闭
- 比继承更灵活（可以动态添加/移除功能）
- 可以组合多个装饰器
- 不修改原有代码

**核心要素**：
- 原始对象
- 装饰器函数/类（接收对象，返回增强的对象）
- 可叠加使用

**应用场景**：
- 函数计时、日志记录
- 权限检查、登录验证
- 缓存、重试
- 性能监控、错误处理

**Python 特性结合**：
- 函数装饰器（`@decorator`）
- 类装饰器
- `functools.wraps` 保留元信息
- 可叠加装饰器

**Python 特有优势**：
- 原生支持装饰器语法
- 装饰器可以是函数、类、方法

**example**：
```python
from functools import wraps
import time

# 1. 函数装饰器
def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.3f}秒")
        return result
    return wrapper

def logger(func):
    """日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}，参数: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} 返回: {result}")
        return result
    return wrapper

# 2. 叠加使用（从下往上应用）
@timer
@logger
def calculate(x: int, y: int) -> int:
    time.sleep(1)
    return x + y

# 3. 类装饰器
class RetryDecorator:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"第 {attempt + 1} 次尝试失败: {e}")
                    if attempt == self.max_retries - 1:
                        raise
        return wrapper

@RetryDecorator(max_retries=3)
def unstable_function():
    # 不稳定的函数
    pass
```

---

## **4. 设计模式总结对比**

| 模式 | 类别 | 核心目的 | 主要优势 | 典型场景 |
|------|------|----------|----------|----------|
| **工厂模式** | 创建型 | 解耦对象创建 | 配置驱动、易扩展 | 多种数据库/队列选择 |
| **单例模式** | 创建型 | 全局唯一实例 | 节省资源、状态共享 | 配置管理器、连接池 |
| **建造者模式** | 创建型 | 分步构建对象 | 链式调用、清晰易读 | SQL 查询、HTTP 请求 |
| **Mixin** | 结构型 | 功能混入 | 代码复用、灵活组合 | 时间戳、序列化 |
| **协议模式** | 结构型 | 定义接口规范 | 鸭子类型、类型检查 | 任务队列、数据处理 |
| **适配器模式** | 结构型 | 接口转换 | 统一接口、兼容旧代码 | 第三方库集成 |
| **外观模式** | 结构型 | 简化复杂接口 | 隐藏细节、易用性 | 视频转换、订单处理 |
| **依赖注入** ⭐ | 结构型 | 解耦依赖关系 | 可测试、可替换 | 服务层、测试 mock |
| **仓储模式** | 结构型 | 抽象数据访问 | 隔离存储、易切换 | 用户/订单/任务数据访问 |
| **策略模式** | 行为型 | 算法可替换 | 避免 if-else、易扩展 | 支付方式、排序算法 |
| **观察者模式** | 行为型 | 状态变化通知 | 解耦、事件驱动 | 任务监听、UI 事件 |
| **装饰器模式** | 行为型 | 动态添加功能 | 不修改原代码、可叠加 | 日志、计时、缓存 |

> **说明**：⭐ 标记的模式在现代开发中极其重要，强烈推荐掌握

---

## **5. 在实际项目中的应用**

### **queue 项目中已使用的模式**

#### ✅ **工厂模式** - `QueueFactory`
```python
# factory.py
queue = QueueFactory.create("sqlite", db_path="prod.db")
```

#### ✅ **协议模式** - `TaskQueue Protocol`
```python
# protocol.py
class TaskQueue(Protocol):
    async def add_task(self, task: Task) -> None: ...
```

#### ✅ **Mixin 模式** - `TimestampMixin`
```python
# models.py
class Task(TimestampMixin):
    # 自动获得 created_at、updated_at
    pass
```

#### ✅ **依赖注入** - `TaskWorker`
```python
# worker.py
class TaskWorker:
    def __init__(self, queue: TaskQueue):  # 🔥 依赖注入
        self.queue = queue  # 从外部传入，不在内部创建
```

#### ✅ **仓储模式** - `TaskQueue` 本质上是任务仓储
```python
# TaskQueue 提供类似集合的接口操作任务
# - add_task() → save()
# - get_task() → find_by_id()
# - get_pending_tasks() → find_all(filter)
```

### **可以添加的模式**

#### 建议 1：**观察者模式** - 任务状态监听
```python
# 监听任务完成，发送通知
class TaskCompletionObserver:
    async def on_task_completed(self, task_id: str, result: str):
        # 发送邮件、webhook 等
        pass
```

#### 建议 2：**策略模式** - 任务重试策略
```python
# 不同的重试策略
class RetryStrategy(Protocol):
    def get_delay(self, attempt: int) -> float: ...

class FixedRetryStrategy:
    def get_delay(self, attempt: int) -> float:
        return 5.0  # 固定 5 秒

class ExponentialRetryStrategy:
    def get_delay(self, attempt: int) -> float:
        return 2 ** attempt  # 指数退避
```

#### 建议 3：**装饰器模式** - 任务计时
```python
@timer
@retry(max_attempts=3)
async def process_task(task: Task) -> str:
    # 自动计时、自动重试
    pass
```

---

## **6. 设计模式的选择原则**

### **何时使用？**

**创建型模式**：
1. **工厂模式**：需要根据配置/参数创建不同类型的对象
2. **单例模式**：需要全局唯一的资源（谨慎使用，可能导致测试困难）
3. **建造者模式**：构造函数参数超过 5 个，或构建过程复杂

**结构型模式**（最多！）：
4. **Mixin**：多个类需要相同的辅助功能（时间戳、序列化等）
5. **协议模式**：需要定义接口，但不想强制继承（推荐！）
6. **适配器模式**：需要集成接口不兼容的第三方库
7. **外观模式**：需要简化复杂子系统的使用
8. **依赖注入** ⭐：类有外部依赖时，始终使用（可测试性的关键）
9. **仓储模式** ⭐：需要数据访问层时，隔离业务逻辑和存储细节

**行为型模式**：
10. **策略模式**：有多个算法/实现，且需要运行时切换
11. **观察者模式**：一个对象状态改变需要通知多个其他对象
12. **装饰器模式**：需要动态添加功能（日志、缓存、权限检查等）

### **不要过度设计**

- **YAGNI 原则**：You Aren't Gonna Need It（你不会需要它）
- **先简单实现**，有需求再重构
- **设计模式是工具**，不是目的
- **代码可读性优先**

---

## **7. Python 设计模式的特点**

### **Python 的优势**

1. **鸭子类型**：Protocol 让接口更灵活
2. **一等函数**：策略可以是函数，不一定要类
3. **装饰器语法**：原生支持装饰器模式
4. **多重继承**：Mixin 模式更自然
5. **动态类型**：工厂模式更简洁

### **与静态语言的区别**

| 特性 | Python | Java/C++ |
|------|--------|----------|
| 接口定义 | Protocol（结构化类型） | Interface/Abstract Class（名义类型） |
| 装饰器 | 语法糖 `@decorator` | 需要手动实现包装类 |
| 单例 | 装饰器/元类 | 私有构造函数 + 静态方法 |
| 策略 | 可以用函数 | 通常需要类 |

---

## **8. 推荐阅读**

- **《设计模式：可复用面向对象软件的基础》**（GoF，四人帮）
- **《Head First 设计模式》**（图解版，易懂）
- **Python 官方文档**：typing 模块、collections.abc 模块
- **pydantic 源码**：大量使用了工厂、协议、Mixin 模式

---

## **9. 快速参考**

### **创建对象时**
- 需要根据配置选择实现 → **工厂模式**
- 需要全局唯一实例 → **单例模式**
- 参数太多、构建复杂 → **建造者模式**

### **组合类与依赖时**（结构型）
- 需要混入辅助功能 → **Mixin**
- 需要定义接口规范 → **协议模式**
- 需要统一不同接口 → **适配器模式**
- 需要简化复杂子系统 → **外观模式**
- **类有外部依赖** → **依赖注入** ⭐（始终使用，可测试性关键）
- **需要数据访问层** → **仓储模式** ⭐（隔离存储细节）

### **处理行为时**
- 需要切换算法 → **策略模式**
- 需要状态通知 → **观察者模式**
- 需要动态添加功能 → **装饰器模式**

---

## **附录：12 个模式的本质总结**

### **核心价值**

这 12 个模式解决了软件开发中的核心问题：

1. **如何创建对象？**（创建型 3 个） → 工厂、单例、建造者
2. **如何组合功能？**（结构型 6 个） → Mixin、协议、适配器、外观、**依赖注入**、**仓储**
3. **如何处理变化？**（行为型 3 个） → 策略、观察者、装饰器

### **最重要的三个模式** ⭐⭐⭐

如果只学三个，就学这三个：

1. **依赖注入**：可测试性的基础，现代开发必备
2. **协议模式**：Python 的接口定义，优于抽象基类
3. **工厂模式**：配置驱动，解耦创建逻辑

### **其余 23 种 GoF 模式为什么不推荐？**

| 模式 | 不推荐原因 |
|------|-----------|
| **迭代器** | Python 原生支持（`__iter__`、`__next__`） |
| **命令模式** | 一等函数直接传递即可 |
| **模板方法** | 继承就能实现，太简单 |
| **享元模式** | 现代硬件很少需要共享对象节省内存 |
| **桥接模式** | Python 组合足够灵活 |
| **组合模式** | 用得少，场景特殊（树形结构） |
| **代理模式** | 与装饰器、外观重叠 |
| **责任链模式** | 用得少，中间件模式更常见 |
| **备忘录模式** | 99% 项目用不到 |
| **访问者模式** | 在 Python 鸭子类型下很别扭 |
| **中介者模式** | 消息队列/事件总线更常见 |
| **状态模式** | 策略模式的特例 |
| **原型模式** | `copy.deepcopy()` 解决 |
| **抽象工厂** | 工厂模式的变种，过度设计 |
| **解释器模式** | 99.9% 的人一辈子用不到 |

**结论**：这 12 个模式已经涵盖了 95% 的实际场景。其余模式要么被语言特性取代，要么使用频率极低。专注于这 12 个，够用了！

