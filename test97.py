import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

def get_memory_usage():
    """获取当前GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                   These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding

class StandardResNet3DBlock(nn.Module):
    """标准的3D ResNet Block - 使用GroupNorm，仅推理模式，带时间投影"""
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32, time_embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        # 时间投影层
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels)
            )
    
    def forward(self, x, timestep_embedding):
        # 第一个卷积 + GroupNorm
        out = self.conv1(x)
        out = self.gn1(out)
        
        # 时间投影，广播到空间维度
        time_emb = self.time_proj(timestep_embedding)  # [B, C]
        # 扩展到匹配3D张量的形状 [B, C, D, H, W]
        time_emb = time_emb[:, :, None, None, None]  # [B, C, 1, 1, 1]
        out = out + time_emb
        out = F.relu(out)
        
        # 第二个卷积 + GroupNorm
        out = self.gn2(self.conv2(out))
        
        # shortcut连接
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class MemoryOptimizedResNet3DBlock(nn.Module):
    """显存优化的3D ResNet Block - 使用维度切分和GroupNorm，仅推理模式，带时间投影"""
    def __init__(self, in_channels, out_channels, stride=1, split_size=2, num_groups=32, time_embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.split_size = split_size
        
        # 时间投影层
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels)
            )
    
    def _split_conv3d_with_time(self, conv_layer, gn_layer, x, time_emb=None, dim=2):
        """对3D卷积进行维度切分处理，支持时间嵌入"""
        # 在指定维度上切分输入张量
        splits = torch.chunk(x, self.split_size, dim=dim)
        outputs = []
        
        for split in splits:
            # 对每个分片单独进行卷积操作
            out = conv_layer(split)
            out = gn_layer(out)
            
            # 如果有时间嵌入，添加到每个分片
            if time_emb is not None:
                out = out + time_emb
            
            outputs.append(out)
            
            # 立即删除中间结果以释放显存
            del out
        
        # 合并结果
        result = torch.cat(outputs, dim=dim)
        
        # 清理中间变量
        del outputs, splits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
    
    def forward(self, x, timestep_embedding):
        # 时间投影，广播到空间维度
        time_emb = self.time_proj(timestep_embedding)  # [B, C]
        time_emb = time_emb[:, :, None, None, None]  # [B, C, 1, 1, 1]
        
        # 第一个卷积层使用切分，加入时间嵌入
        out = self._split_conv3d_with_time(self.conv1, self.gn1, x, time_emb)
        out = F.relu(out)
        
        # 第二个卷积层使用切分，不加时间嵌入
        out = self._split_conv3d_with_time(self.conv2, self.gn2, out)
        
        # shortcut连接
        shortcut_out = self.shortcut(x)
        out += shortcut_out
        out = F.relu(out)
        
        return out

class GradientCheckpointingResNet3DBlock(nn.Module):
    """使用梯度检查点的3D ResNet Block - 使用GroupNorm，仅推理模式，带时间投影"""
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32, time_embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        
        # 时间投影层
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels)
            )
    
    def forward(self, x, timestep_embedding):
        # 时间投影，广播到空间维度
        time_emb = self.time_proj(timestep_embedding)  # [B, C]
        time_emb = time_emb[:, :, None, None, None]  # [B, C, 1, 1, 1]
        
        # 推理模式下不需要梯度检查点，直接前向传播
        out = self.conv1(x)
        out = self.gn1(out)
        out = out + time_emb  # 添加时间嵌入
        out = F.relu(out)
        
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def test_memory_usage():
    """测试不同实现的显存使用情况 - 仅推理模式，带时间嵌入"""
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数
    batch_size = 2
    in_channels = 64
    out_channels = 32
    D, H, W = 96, 120, 64  # 3D张量的深度、高度、宽度
    num_groups = 32  # GroupNorm参数
    time_embedding_dim = 512  # 时间嵌入维度
    
    # 创建测试输入 (五维张量: batch, channel, depth, height, width)
    input_tensor = torch.randn(batch_size, in_channels, D, H, W, device=device)
    
    # 创建时间步和时间嵌入
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    time_embed = timestep_embedding(timesteps, time_embedding_dim)
    
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"时间嵌入形状: {time_embed.shape}")
    print(f"输入张量大小: {input_tensor.numel() * 4 / 1024**3:.3f} GB")
    print(f"时间嵌入大小: {time_embed.numel() * 4 / 1024**6:.3f} MB")
    
    models = {
        'Standard': StandardResNet3DBlock(in_channels, out_channels, num_groups=num_groups, time_embedding_dim=time_embedding_dim).to(device),
        'Memory Optimized (split=2)': MemoryOptimizedResNet3DBlock(in_channels, out_channels, split_size=2, num_groups=num_groups, time_embedding_dim=time_embedding_dim).to(device),
        'Memory Optimized (split=4)': MemoryOptimizedResNet3DBlock(in_channels, out_channels, split_size=4, num_groups=num_groups, time_embedding_dim=time_embedding_dim).to(device),
        'Memory Optimized (split=8)': MemoryOptimizedResNet3DBlock(in_channels, out_channels, split_size=8, num_groups=num_groups, time_embedding_dim=time_embedding_dim).to(device)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"测试 {name}")
        print(f"{'='*50}")
        
        # 清空显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.eval()  # 设置为推理模式
        
        # 预热GPU（运行几次以确保所有CUDA内核已加载）
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor, time_embed)
        
        # 使用CUDA事件进行准确的时间测量
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # 同步GPU
            torch.cuda.synchronize()
            
            # 开始计时
            start_event.record()
            
            # 推理测试（带时间嵌入）
            with torch.no_grad():
                output = model(input_tensor, time_embed)
            
            # 结束计时
            end_event.record()
            torch.cuda.synchronize()
            
            inference_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
        else:
            # CPU上的时间测量
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor, time_embed)
            inference_time = time.time() - start_time
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        print(f"输出形状: {output.shape}")
        print(f"推理时间: {inference_time:.4f}s")
        print(f"峰值显存使用: {peak_memory:.3f} GB")
        
        results[name] = {
            'inference_time': inference_time,
            'peak_memory': peak_memory,
            'output_shape': output.shape
        }
        
        # 清理
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 打印对比结果
    print(f"\n{'='*80}")
    print("性能对比总结 - 推理模式 + 时间嵌入")
    print(f"{'='*80}")
    
    standard_memory = results['Standard']['peak_memory']
    
    print(f"{'方法':<35} {'推理显存(GB)':<15} {'显存节省':<15} {'推理时间(s)':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        memory = result['peak_memory']
        memory_saved = f"{(1 - memory/standard_memory)*100:.1f}%" if standard_memory > 0 else "N/A"
        
        print(f"{name:<35} {memory:<15.3f} {memory_saved:<15} {result['inference_time']:<15.4f}")

def test_different_split_sizes():
    """测试不同切分大小的效果 - 仅推理模式，带时间嵌入"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过切分大小测试")
        return
    
    print(f"\n{'='*80}")
    print("测试不同切分大小的效果 - 推理模式 + 时间嵌入")
    print(f"{'='*80}")
    
    batch_size = 2
    in_channels = 64
    out_channels = 128
    D, H, W = 97, 160, 64
    num_groups = 32
    time_embedding_dim = 512
    
    input_tensor = torch.randn(batch_size, in_channels, D, H, W, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    time_embed = timestep_embedding(timesteps, time_embedding_dim)
    
    split_sizes = [1, 2, 4, 8, 16]
    
    print(f"{'切分大小':<10} {'峰值显存(GB)':<15} {'推理时间(s)':<15} {'显存节省':<15}")
    print("-" * 60)
    
    baseline_memory = None
    
    for split_size in split_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = MemoryOptimizedResNet3DBlock(in_channels, out_channels, split_size=split_size, 
                                           num_groups=num_groups, time_embedding_dim=time_embedding_dim).to(device)
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor, time_embed)
        
        # 使用CUDA事件进行准确的时间测量
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.no_grad():
            output = model(input_tensor, time_embed)
        
        end_event.record()
        torch.cuda.synchronize()
        
        inference_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        if baseline_memory is None:
            baseline_memory = peak_memory
            memory_saved = "baseline"
        else:
            memory_saved = f"{(1 - peak_memory/baseline_memory)*100:.1f}%"
        
        print(f"{split_size:<10} {peak_memory:<15.3f} {inference_time:<15.4f} {memory_saved:<15}")
        
        del model, output
        torch.cuda.empty_cache()
if __name__ == "__main__":

    
    test_memory_usage()
    

    test_different_split_sizes()
    

    
    print(f"\n{'='*80}")
    print("所有测试完成！")
    print(f"{'='*80}")
