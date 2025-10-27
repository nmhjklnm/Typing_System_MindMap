# FAQ

## 1. TypedDict vs pydantic，IDE 提示有区别吗？

**都有 IDE 提示**。TypedDict 是字典用 `user["name"]` 访问，pydantic 是对象用 `user.name` 访问。核心区别：TypedDict 零开销无运行时验证（标准库），pydantic 有运行时验证和类型转换（第三方库）。
**选择建议**：API/用户输入用 pydantic（需验证），处理外部字典/库开发用 TypedDict（轻量/兼容性）。

## 2. typing.md 和 collections.md 的关系？

**collections.md 覆盖抽象基类类型**。`Callable`、`Iterable`、`Iterator`、`Generator` 等类型在 Python 3.9+ 推荐从 `collections.abc` 导入，因此在 typing.md 中已删除这些条目，避免重复和混淆。两者各司其职：typing.md 专注类型构造器（如 TypeVar、Protocol），collections.md 覆盖抽象基类。

## 3. 类型系统的反对称性在 Python 中有体现吗？

**大部分成立**。反对称性（`A <: B` 且 `B <: A` → `A = B`）在类继承中天然成立（不同类不可能互为子类型），在结构相同的 Protocol 中也成立（等价类型）。
**特殊情况**：`Any` 和任何类型双向兼容但不等价（渐进类型设计）；`NewType` 故意创建单向子类型关系（`UserId <: int` 但 `int ⊄: UserId`）。

## 4. 为什么说 Python 中是 object，类型系统中是 Any？

**两个层面的 Top 类型**。`object` 是**运行时**所有类的基类（类层级顶层），但类型检查器限制只能使用基础方法。`Any` 是**类型检查**的 Top 类型（类型层级顶层），可以赋给任何类型、进行任何操作，但会跳过类型检查。
**区别**：`object` 保证类型安全（单向赋值），`Any` 牺牲类型安全（双向赋值+任意操作）。选择：需要接受任何值但保证安全用 `object`，需要绕过类型检查用 `Any`。

## 5. Never 为什么很少见到使用？

**理论重要，实践罕见**。`Never` 是 Bottom 类型（所有类型的子类型），但只在两种情况用：①永不返回的函数（`def raise_error() -> Never: raise Exception()`），但大多数人用旧的 `NoReturn`；②穷尽性检查（`assert_never()`），但很少有人写。
**根本原因**：Python 类型系统实用优先，`Never` 只在极端情况有意义。大多数函数正常返回，异常处理独立于类型系统，99% 的开发者一辈子用不到 `Never`。

## 6. 为什么类型变量用 `T` 这样的单字母命名？

**约定俗成的传统**。源自数学和早期编程语言（C++/Java），单字母（`T`=Type、`K`=Key、`V`=Value）简洁且形成视觉信号，一眼就能区分类型变量和普通变量。
**不是强制**：可以用 `ItemType`、`ResponseType` 等描述性名字。简单泛型用 `T`，多个类型变量或有约束时用描述性名字更清晰。

## 7. 受限 TypeVar 和 Union 有什么区别？

**核心区别：类型一致性**。`T = TypeVar('T', int, str)` 保证输入和输出是**同一个**具体类型（`process(42) → int`）；`Union[int, str]` 只表示可能性，返回类型是联合类型（`process(42) → Union[int, str]`），丢失了精确信息。
**应用**：泛型函数用受限 TypeVar 保持类型精确度；普通参数真的接受多种类型且无需区分用 Union。

## 8. 泛型的中括号语法是如何实现的？

**通过 `__class_getitem__` 魔术方法实现**。当你写 `Box[int]` 时，Python 调用 `Box.__class_getitem__(int)` 返回一个特化的类。继承 `Generic[T]` 后类自动获得这个能力。
**两个步骤**：`Box[int]` 是类型特化（调用 `__class_getitem__`，得到类），`Box[int](42)` 是实例化（调用 `__init__`，得到对象）。就像"填模板"和"用模板造物"两步。

## 9. 类体中 `name: type` 这种写法是什么？

**类属性注解（class attribute annotations）**。只声明属性类型不创建实际属性，需在 `__init__` 中赋值。与类变量不同（`name: str = "value"` 会创建共享的类变量）。
**作用**：为类型检查器和 IDE 提供属性类型信息，不产生运行时开销。适合泛型类（如 `items: list[T]`）清晰展示所有实例属性及类型。
**pydantic 特殊行为**：继承 `BaseModel` 时，类属性注解会触发元类魔法，自动创建字段描述符、生成 `__init__` 和验证逻辑，相当于既声明又赋值。

## 10. Protocol 是泛型吗？

**是的，Protocol 内部继承了 Generic**。可以直接写 `Protocol[T]` 而无需同时继承 `Generic[T]`。设计动机：接口定义也需要泛型能力来保持类型安全（如 `Repository[T]` 可以约束返回 `T` 类型）。
**区别**：`Protocol[T]` 定义接口规范（无需显式继承，鸭子类型），`Generic[T]` 创建具体实现（必须显式继承）。两者常配合：用 Protocol 定义接口，用 Generic 实现类，既灵活又类型安全。

## 11. 为什么 Python 从鸭子类型进化到了可验证的结构化类型系统？

**工程化需求驱动**。大型项目中鸭子类型维护困难（团队协作、重构、理解函数期望），IDE 无法精准提示，类型错误只能运行时发现成本高。PEP 484（2014）引入**渐进式类型系统**（可选的类型提示），既保留鸭子类型灵活性，又提供静态检查安全性。
**核心理念**：工程化安全性 + Python 灵活性两者兼得。可以只在需要的地方添加类型，不破坏现有代码。

## 12. 泛型可以使用多个类型参数吗？

**是的，可以任意数量**。泛型就像函数可以接受多个参数，`Generic[K, V]` 需要在使用时提供对应数量的具体类型如 `Box[int, str]`。标准库例子：`Dict[K, V]`（键值类型）、`Tuple[int, str, bool]`（多位置类型）。
**注意**：类型参数数量必须匹配定义（`Generic[K, V]` 就必须传两个类型，不能多也不能少）。

## 13. `class Box[T]` 新语法中的 T 和外部 `T = TypeVar('T')` 是同一个吗？

**不是，是两个独立的 T**。新语法 `class Box[T]:` 会自动创建一个**新的** TypeVar，只在类内部作用域有效；外部的 `T = TypeVar('T')` 是另一个独立的变量。当你在外部写 `Box[T]` 时，用的是外部那个 T，不是类定义时自动创建的。
**避免混淆**：不要混用新旧语法。Python 3.12+ 用 `class Box[T]:`（自动管理 TypeVar），Python 3.9-3.11 用 `class Box(Generic[T])`+外部 `T = TypeVar('T')`（手动管理）。混用会造成作用域混乱。

## 14. 为什么 `BoxOfT = Box[T]` 后不能写 `BoxOfT[int]`？
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Box(Generic[T]):  # Box 是泛型类（工厂）
    pass

# ❌ 错误：对参数化类型再参数化
BoxOfT = Box[T]  # Box[T] 是参数化类型（产品）
def f() -> BoxOfT[int]: ...  # 错误！相当于 (Box[T])[int]
```

**`Box[T]` 已经是参数化类型，不能再参数化**。泛型类像"工厂"（`Box`），参数化类型像"产品"（`Box[T]` 或 `Box[int]`）。产品不能再加工。

## 15. 嵌套泛型有什么实际使用场景？

**API 响应封装**：`ApiResponse[PaginatedData[User]]` 统一响应格式包含分页数据；**数据处理管道**：`Result[list[User], Error]` 表示操作成功返回用户列表或失败返回错误；**装饰器模式**：`CachedRepository[Product]` 包含 `Repository[Product]`，带缓存的仓储层。
**核心价值**：类型安全的层层传递，每一层保持类型信息，IDE 和类型检查器能追踪到底，解耦和组合不同泛型类。

## 16. `Callable[[T], U]` 是什么意思？

**描述函数类型**。`Callable` 用来注解可调用对象（函数）的类型，`[[T], U]` 是语法：两层方括号，内层 `[T]` 是参数类型列表（一个 T 类型参数），`U` 是返回类型。
**完整含义**：接收一个 T 类型参数，返回 U 类型值的函数。**示例**：`Callable[[int], str]` 表示 `def func(x: int) -> str`；`Callable[[], None]` 表示无参数无返回值；`Callable[[int, str], bool]` 表示接收两个参数。
**用途**：描述高阶函数（接收或返回函数的函数），如 `def map[T, U](value: T, func: Callable[[T], U]) -> U`。

## 17. 为什么需要 TypeAlias？

**显式标注类型别名**。简单赋值 `Vector = list[float]` 类型检查器无法区分是类型别名还是普通变量赋值。使用 `Vector: TypeAlias = list[float]`（Python 3.10+）明确告诉类型检查器这是类型别名，避免混淆。
**使用建议**：简单场景可以省略（类型检查器会推断），复杂泛型别名或团队规范要求时使用 `TypeAlias` 显式标注，提高代码可读性和类型检查准确性。

## 18. `List[Dog]` 和 `List[Animal]` 是什么关系？

**没有子类型关系（不变）**。即使 `Dog <: Animal`，列表可变导致写入不安全：如果允许 `List[Dog]` 当作 `List[Animal]` 使用，就能往里面添加 `Cat`，破坏类型安全。
**对比**：`Sequence[Dog] <: Sequence[Animal]`（只读容器协变），`tuple[Dog, ...] <: tuple[Animal, ...]`（元组不可变）。
**实践**：函数参数用 `Sequence` 而非 `List`，除非真的需要修改列表。

## 19. TypeVar 的 covariant 和 contravariant 默认是什么？

**默认都是 `False`（不变）**。三种合法组合：①不变 `TypeVar('T')`（默认），②协变 `TypeVar('T', covariant=True)`，③逆变 `TypeVar('T', contravariant=True)`。
**禁止**：`covariant=True, contravariant=True` 会报错 `TypeError: Bivariant type variables are not supported`，因为既协变又逆变在逻辑上矛盾。
**原因**：默认不变最安全，避免可变容器的类型漏洞。需要协变/逆变时必须显式声明。

## 20. 为什么需要手动开启协变/逆变？

**安全优先的设计**。默认不变避免类型漏洞：如果 `Box[Dog]` 自动协变为 `Box[Animal]`，就能往狗盒子里放猫（`box.set(Cat())`），破坏类型安全。
**显式声明意图**：协变（只读/生产者）和逆变（只写/消费者）有严格限制，类型检查器会验证——协变类型不能出现在参数位置，逆变类型不能出现在返回值位置。手动开启强制开发者思考类型安全。

## 21. 逆变有哪些工程应用场景？

**所有"消费者"角色都是逆变**：处理器（Handler）、验证器（Validator）、写入器（Writer）、观察者（Observer）、策略（Strategy）、比较器（Comparator）。
**本质规律**：能处理父类型的对象，自然能处理子类型的对象（因为子类拥有父类的所有特征）。`Handler[Animal] <: Handler[Dog]`——Animal 的处理器能处理 Dog，Dog 的处理器不能处理 Animal（可能缺少 Cat 的行为）。
**与协变对比**：协变是"生产者"（输出端，返回值），逆变是"消费者"（输入端，参数）。

## 22. 为什么参数位置是逆变的？

**替换原则**：需要"处理 Dog 的函数"时，"处理 Animal 的函数"可以替换（能处理更宽泛类型的函数更通用），但"处理 Chihuahua 的函数"不行（太狭窄）。所以 `Callable[[Animal], None] <: Callable[[Dog], None]`（逆变）。
**位置决定型变**：类型参数在返回值位置→协变（生产者），在参数位置→逆变（消费者），既在返回值又在参数→不变（读写都有）。不是"消费者产生逆变"，而是"参数位置要求逆变才能保证类型安全"。

## 23. 返回值位置的类型参数一定要协变吗？

**不是，可以不变**！协变是**优化/放松**，不是强制要求。不变总是安全的（默认），协变提供更多灵活性（`DataSource[Dog]` 可以当 `DataSource[Animal]` 用）。
**选择建议**：默认用不变（简单安全）；确定只有输出且需要灵活性时用协变；既有输入又有输出必须用不变。类型检查器会验证协变类型不出现在参数位置。

## 24. TypeGuard、isinstance、@runtime_checkable 完整对比表

| 场景 | isinstance | @runtime_checkable | TypeGuard | 推荐 |
|------|-----------|-------------------|-----------|------|
| **普通类** | ✅ `isinstance(x, Dog)` | - | ⚠️ 可以但没必要 | isinstance |
| **TypedDict** | ❌ 不支持 | ❌ 不支持 | ✅ 唯一选择 | TypeGuard |
| **Protocol** | ❌ 默认不支持 | ✅ 简单但不精确 | ✅ 精确自定义 | 看需求 |
| **Union** | ⚠️ 部分支持 | ❌ 不支持 | ✅ 完全支持 | TypeGuard |
| **Literal** | ❌ 不支持 | ❌ 不支持 | ✅ 唯一选择 | TypeGuard |
| **泛型容器** | ⚠️ 只能检查外层 | ❌ 不支持 | ✅ 可以深度检查 | TypeGuard |
| **复杂类型** | ❌ 不支持 | ❌ 不支持 | ✅ 唯一选择 | TypeGuard |

## 25. TypeIs 和 TypeGuard 有什么区别？

**本质：TypeGuard 是单向窄化，TypeIs 是双向窄化**。`TypeGuard` 只窄化 True 分支，False 分支类型不变；`TypeIs`（Python 3.13+）双向窄化——True 分支确认类型，False 分支排除类型。
**使用场景**：输入类型明确（如 `int | str`）用 `TypeIs` 获得 False 分支窄化；输入是 `Any`/`object` 或检查不精确用 `TypeGuard`。
**关键约束**：`TypeIs` 要求检查函数必须可靠（返回 True 一定是，返回 False 一定不是），`TypeGuard` 允许不完整检查。

## 26. 类型窄化只适用于 Union 类型吗？

**不是，但 Union 是最常见场景**。类型窄化适用于：①Union 类型（`int | str` → `int`），②Optional（`T | None` → `T`），③宽泛到具体（`Any`/`object` → `str`），④父类到子类（`Animal` → `Dog`），⑤字面量子集（`Literal["a", "b"]` → `Literal["a"]`）。
**为什么 Union 最常见**：Union 表达"多种可能性"，窄化的目的就是确定具体哪一种。实际代码中 Union 应用广泛（函数参数、可选值、错误处理），且 TypeIs 的双向窄化在 Union 中最有价值。

## 27. 除了 Mixin，Python 还有哪些常用设计模式？

**12 个核心模式符合经典三大分类**。创建型 3 个：工厂、单例、建造者；结构型 6 个⭐：Mixin、协议、适配器、外观、依赖注入、仓储；行为型 3 个：策略、观察者、装饰器。
**最重要的三个**：①依赖注入（可测试性基础，结构型），②协议模式（Python 接口定义，结构型），③工厂模式（配置驱动，创建型）。其余 GoF 23 种要么被 Python 语言特性取代（迭代器、命令模式），要么使用频率极低（享元、备忘录、访问者）。这 12 个已涵盖 95% 实际场景。


