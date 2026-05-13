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

团队里曾经有谁

每个人当前是什么角色



