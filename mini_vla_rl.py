import torch
import numpy as np
import time

# ==========================================
# 1. 模拟“虚拟厨房” (Environment) -> 保持不变
# ==========================================
class DummyKitchenEnv:
    def __init__(self):
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        dummy_vision = torch.rand(3, 64, 64) 
        dummy_language = "把鸡蛋打进锅里"
        return {"vision": dummy_vision, "language": dummy_language}
        
    def step(self, action_chunk):
        self.step_count += 1
        time.sleep(0.5) 
        reward = np.random.choice([0.0, 0.0, 0.0, 10.0, -5.0])
        done = (reward == 10.0) or (self.step_count >= 4)
        next_obs = {"vision": torch.rand(3, 64, 64), "language": "把鸡蛋打进锅里"}
        return next_obs, reward, done

# ==========================================
# 2. 模拟“机器人的大脑” (VLA Model) -> 保持不变
# ==========================================
class DummyVLAModel:
    def predict(self, observation):
        action_chunk = torch.rand(10, 6) 
        return action_chunk

# ==========================================
# 🌟 3. 新增：经验回放池 (Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=1000):
        # 这是一个大水池，capacity 是它的最大容量
        self.buffer = []
        self.capacity = capacity
        
    def push(self, obs, action, reward, next_obs, done):
        # 如果水池满了，就把最旧的水滴挤出去 (先进先出)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            
        # 把刚才整理好的 5 个核心数据打包成一个元组，扔进池子
        transition = (obs, action, reward, next_obs, done)
        self.buffer.append(transition)
        
        print(f"💧 [回放池] 成功收集 1 滴经验！当前池子总水量: {len(self.buffer)} 滴")

    def sample(self, batch_size):
        # 模拟“经理”拿着水桶来随机舀水（抽取一批数据用来训练大脑）
        # 实际代码中这里会用随机采样
        print(f"🪣 [经理] 从池子里随机舀出了 {batch_size} 滴经验去更新大脑！")
        return self.buffer[:batch_size]

# ==========================================
# 4. 强化学习核心循环 (带数据收集功能)
# ==========================================
if __name__ == "__main__":
    env = DummyKitchenEnv()
    model = DummyVLAModel()
    
    # 实例化我们的池子
    memory_pool = ReplayBuffer(capacity=100) 
    
    total_episodes = 2 
    
    for episode in range(total_episodes):
        print(f"\n{'='*15} 开始第 {episode + 1} 局 {'='*15}")
        obs = env.reset()
        
        while True:
            print("-" * 40)
            # 大脑思考
            actions = model.predict(obs)
            
            # 环境执行
            next_obs, reward, done = env.step(actions)
            print(f"💯 [环境反馈] 动作执行完毕，得分: {reward}")
            
            # 🌟 关键步骤：把这一步的所有信息扔进池子！
            memory_pool.push(obs, actions, reward, next_obs, done)
            
            # 判断是否结束
            if done:
                print("🏁 本局结束，触发 break。")
                break 
                
            # 准备下一步
            obs = next_obs

    # 等 2 局全部跑完后，看看池子里的情况，并让经理来舀水
    print("\n" + "="*40)
    print(f"🌊 试错结束！池子里总共攒了 {len(memory_pool.buffer)} 滴经验。")
    
    # 假设攒够了数据，经理决定舀 3 滴出来看看
    if len(memory_pool.buffer) >= 3:
        sample_data = memory_pool.sample(batch_size=3)
        print("🔍 经理检查了其中一滴经验的结构：")
        print(f"   - 动作矩阵的形状: {sample_data[0][1].shape} (Action Chunking)")
        print(f"   - 这滴经验的得分: {sample_data[0][2]} 分")