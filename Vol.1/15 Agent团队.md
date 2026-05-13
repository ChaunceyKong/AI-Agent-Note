# Agent团队

子 agent 适合一次性委派；团队系统解决的是“有人长期在线、能继续接活、能互相协作”。

Agent团队与subagent的区别：

- subagent：一次性执行单元，执行完一次任务后，就消失。
  ```
  创建 -> 执行 -> 返回摘要 -> 消失
  ```
  
- Agent团队：一批有身份、能长期存在、能反复协作的队友。

## 名词解释

`队友`：一个拥有名字、角色、消息入口和生命周期的持久 agent。

`名册`：团队成员列表。

`邮箱`：每个队友的收件箱。别人把消息发给它，它在自己的下一轮工作前先去收消息。

## 最小心智模型

每个队友都是 一个有自己循环、自己收件箱、自己上下文的人。

```
lead
  |
  +-- spawn alice (coder)
  +-- spawn bob (tester)
  |
  +-- send message --> alice inbox
  +-- send message --> bob inbox

alice
  |
  +-- 自己的 messages
  +-- 自己的 inbox
  +-- 自己的 agent loop

bob
  |
  +-- 自己的 messages
  +-- 自己的 inbox
  +-- 自己的 agent loop
```

## 关键数据结构

1、TeamMember

```
member = {
    "name": "alice",
    "role": "coder",
    "status": "working",
}
```

`name`：名字

`role`：角色

`status`：状态

2、TeamConfig

```
config = {
    "team_name": "default",
    "members": [member1, member2],
}
```

它通常可以放在：`.team/config.json`

这份名册让系统重启以后，仍然知道：
团队里曾经有谁，
每个人当前是什么角色

3、MessageEnvelope

把消息正文和元信息一起包起来的一条记录。

```
message = {
    "type": "message",
    "from": "lead",
    "content": "Please review auth module.",
    "timestamp": 1710000000.0,
}
```

## 最小实现

1、先有一份队伍名册

```
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.team_dir = team_dir
        self.config_path = team_dir / "config.json"
        self.config = self._load_config()
```

2、spawn 一个持久队友

```
def spawn(self, name: str, role: str, prompt: str):
    member = {"name": name, "role": role, "status": "working"}
    self.config["members"].append(member)
    self._save_config()

    thread = threading.Thread(
        target=self._teammate_loop,
        args=(name, role, prompt),
        daemon=True,
    )
    thread.start()
```

这里的关键不在于线程本身，而在于：

**队友一旦被创建，就不只是一次性工具调用，而是一个有持续生命周期的成员。**

3、给每个队友一个邮箱

```
.team/inbox/alice.jsonl
.team/inbox/bob.jsonl
```

收消息时：

- 读出全部
- 解析为消息列表
- 清空收件箱

4、队友每轮先看邮箱，再继续工作

```
def teammate_loop(name: str, role: str, prompt: str):
    messages = [{"role": "user", "content": prompt}]

    while True:
        inbox = bus.read_inbox(name)
        for item in inbox:
            messages.append({"role": "user", "content": json.dumps(item)})

        response = client.messages.create(...)
        ...
```




## 执行流程

```
用户目标 / lead 判断需要长期分工
  ->
spawn teammate
  ->
写入 .team/config.json
  ->
通过 inbox 分派消息、摘要、任务线索
  ->
teammate 先 drain inbox
  ->
进入自己的 agent loop 和工具调用
  ->
把结果回送给 lead，或继续等待下一轮工作
```
