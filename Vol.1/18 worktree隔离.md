# worktree隔离

worktree 保证每个任务应该在各自独立工作空间里执行。

## 名词解释

`worktree`：同一个仓库的另一个独立检出目录。

`隔离执行`：任务 A 在自己的目录里跑，任务 B 在自己的目录里跑，彼此默认不共享未提交改动。

`绑定`：把某个任务 ID 和某个 worktree 记录明确关联起来。

## 最小心智模型

```
任务板
  负责回答：做什么、谁在做、状态如何

worktree 注册表
  负责回答：在哪做、目录在哪、对应哪个任务
```

两者通过`task_id`关联起来

```
.tasks/task_12.json
  {
    "id": 12,
    "subject": "Refactor auth flow",
    "status": "in_progress",
    "worktree": "auth-refactor"
  }

.worktrees/index.json
  {
    "worktrees": [
      {
        "name": "auth-refactor",
        "path": ".worktrees/auth-refactor",
        "branch": "wt/auth-refactor",
        "task_id": 12,
        "status": "active"
      }
    ]
  }
```
