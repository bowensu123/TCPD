import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List, Tuple, Callable, Union, Optional
import math
import gc
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

class MemoryProfiler:
    """显存监控工具"""
    
    def __init__(self):
        self.memory_logs = []
        self.time_logs = []
        self.operation_logs = []
        self.start_time = time.time()
    
    def log_memory(self, operation: str = ""):
        """记录当前显存使用"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            cached_memory = torch.cuda.memory_reserved() / (1024 * 1024)
        else:
            current_memory = max_memory = cached_memory = 0
        
        self.memory_logs.append({
            'allocated': current_memory,
            'max_allocated': max_memory, 
            'cached': cached_memory
        })
        self.time_logs.append(time.time() - self.start_time)
        self.operation_logs.append(operation)
    
    def reset_peak(self):
        """重置峰值显存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_summary(self):
        """获取显存使用摘要"""
        if not self.memory_logs:
            return "无显存记录"
        
        allocated = [log['allocated'] for log in self.memory_logs]
        max_allocated = [log['max_allocated'] for log in self.memory_logs]
        cached = [log['cached'] for log in self.memory_logs]
        
        return {
            'peak_allocated': max(allocated),
            'final_allocated': allocated[-1],
            'peak_cached': max(cached),
            'operations': len(self.memory_logs)
        }
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """绘制显存使用图表"""
        if not self.memory_logs:
            print("无显存数据可绘制")
            return
        
        allocated = [log['allocated'] for log in self.memory_logs]
        cached = [log['cached'] for log in self.memory_logs]
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_logs, allocated, 'b-', label='已分配显存', linewidth=2)
        plt.plot(self.time_logs, cached, 'r--', label='缓存显存', alpha=0.7)
        
        # 添加操作标记
        for i, (t, op) in enumerate(zip(self.time_logs, self.operation_logs)):
            if op and i % 5 == 0:  # 每5个点标记一次，避免过密
                plt.annotate(op, (t, allocated[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('时间 (秒)')
        plt.ylabel('显存使用 (MB)')
        plt.title('显存使用变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"显存使用图已保存到: {save_path}")
        else:
            plt.show()


class MemoryOptimizedTiler:
    """
    五维tensor切分处理工具，专门针对GroupNorm优化显存使用
    假设tensor维度为: (batch, channel, depth, height, width)
    """
    
    def __init__(self, enable_gc: bool = True, clear_cache: bool = True, enable_profiling: bool = True):
        self.enable_gc = enable_gc
        self.clear_cache = clear_cache
        self.enable_profiling = enable_profiling
        self.profiler = MemoryProfiler() if enable_profiling else None
    
    def _memory_cleanup(self, *tensors_to_delete, operation: str = "cleanup"):
        """强制清理显存，避免内存碎片"""
        if self.profiler:
            self.profiler.log_memory(f"before_{operation}")
        
        # 删除指定tensor
        for tensor in tensors_to_delete:
            if tensor is not None:
                del tensor
        
        # 强制垃圾回收
        if self.enable_gc:
            gc.collect()
        
        # 清空CUDA缓存
        if self.clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.profiler:
            self.profiler.log_memory(f"after_{operation}")
    
    def _get_memory_usage(self) -> float:
        """获取当前GPU显存使用量(MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0
    
    def single_dim_tile_optimized(self, 
                                 tensor: torch.Tensor, 
                                 process_fn: Callable[[torch.Tensor], torch.Tensor],
                                 dim: int, 
                                 tile_size: int,
                                 overlap: int = 0) -> torch.Tensor:
        """
        优化的单维度切分，特别针对GroupNorm处理
        
        Args:
            tensor: 输入五维tensor (B, C, D, H, W)
            process_fn: 处理函数
            dim: 切分的维度 (0-4)
            tile_size: 每个tile的大小
            overlap: 重叠区域大小（用于处理边界效应）
        """
        if self.profiler:
            self.profiler.log_memory("tile_start")
            self.profiler.reset_peak()
        
        original_shape = tensor.shape
        dim_size = original_shape[dim]
        device = tensor.device
        dtype = tensor.dtype
        
        # 预分配输出tensor
        # 先通过一个小sample确定输出形状
        with record_function("sample_shape_detection"):
            sample_slice = [slice(None)] * 5
            sample_slice[dim] = slice(0, min(tile_size, dim_size))
            sample_input = tensor[tuple(sample_slice)]
            
            if self.profiler:
                self.profiler.log_memory("after_sample_slice")
            
            with torch.no_grad():
                sample_output = process_fn(sample_input)
            
            if self.profiler:
                self.profiler.log_memory("after_sample_process")
        
        output_shape = list(sample_output.shape)
        output_shape[dim] = original_shape[dim]  # 保持原始维度大小
        
        # 预分配输出tensor
        with record_function("output_allocation"):
            output = torch.empty(output_shape, dtype=sample_output.dtype, device=device)
            
            if self.profiler:
                self.profiler.log_memory("after_output_allocation")
        
        # 清理sample
        self._memory_cleanup(sample_input, sample_output, operation="sample_cleanup")
        
        # 计算tile数量
        num_tiles = math.ceil(dim_size / tile_size)
        
        for i in range(num_tiles):
            with record_function(f"tile_{i}"):
                start_idx = max(0, i * tile_size - overlap)
                end_idx = min((i + 1) * tile_size + overlap, dim_size)
                
                # 创建输入切片
                input_slice = [slice(None)] * 5
                input_slice[dim] = slice(start_idx, end_idx)
                
                # 提取tile (避免复制，使用view)
                tile_input = tensor[tuple(input_slice)]
                
                if self.profiler:
                    self.profiler.log_memory(f"tile_{i}_input_ready")
                
                # 处理tile
                with torch.no_grad():
                    tile_output = process_fn(tile_input)
                
                if self.profiler:
                    self.profiler.log_memory(f"tile_{i}_processed")
                
                # 处理重叠区域
                if overlap > 0 and i > 0:
                    # 移除前面的重叠
                    tile_slice = [slice(None)] * 5
                    tile_slice[dim] = slice(overlap, tile_output.shape[dim])
                    tile_output = tile_output[tuple(tile_slice)]
                
                if overlap > 0 and i < num_tiles - 1:
                    # 移除后面的重叠
                    tile_slice = [slice(None)] * 5
                    tile_slice[dim] = slice(0, tile_output.shape[dim] - overlap)
                    tile_output = tile_output[tuple(tile_slice)]
                
                # 计算输出位置
                out_start = i * tile_size
                out_end = min((i + 1) * tile_size, dim_size)
                
                # 直接写入预分配的输出tensor
                output_slice = [slice(None)] * 5
                output_slice[dim] = slice(out_start, out_end)
                output[tuple(output_slice)] = tile_output
                
                if self.profiler:
                    self.profiler.log_memory(f"tile_{i}_written")
                
                # 立即清理中间变量
                self._memory_cleanup(tile_input, tile_output, operation=f"tile_{i}_cleanup")
        
        if self.profiler:
            self.profiler.log_memory("tile_complete")
        
        return output
    
    def multi_dim_tile_optimized(self, 
                               tensor: torch.Tensor,
                               process_fn: Callable[[torch.Tensor], torch.Tensor],
                               dims: List[int],
                               tile_sizes: List[int]) -> torch.Tensor:
        """
        优化的多维度切分
        """
        assert len(dims) == len(tile_sizes), "dims和tile_sizes长度必须相等"
        assert len(dims) <= 2, "目前最多支持2个维度同时切分"
        
        if len(dims) == 1:
            return self.single_dim_tile_optimized(tensor, process_fn, dims[0], tile_sizes[0])
        
        # 两个维度的情况
        dim1, dim2 = dims
        size1, size2 = tile_sizes
        
        original_shape = tensor.shape
        device = tensor.device
        
        # 获取输出形状
        sample_slice = [slice(None)] * 5
        sample_slice[dim1] = slice(0, min(size1, original_shape[dim1]))
        sample_slice[dim2] = slice(0, min(size2, original_shape[dim2]))
        sample_input = tensor[tuple(sample_slice)]
        
        with torch.no_grad():
            sample_output = process_fn(sample_input)
        
        output_shape = list(sample_output.shape)
        output_shape[dim1] = original_shape[dim1]
        output_shape[dim2] = original_shape[dim2]
        
        # 预分配输出
        output = torch.empty(output_shape, dtype=sample_output.dtype, device=device)
        
        # 清理sample
        self._memory_cleanup(sample_input, sample_output)
        
        # 计算tile数量
        num_tiles1 = math.ceil(original_shape[dim1] / size1)
        num_tiles2 = math.ceil(original_shape[dim2] / size2)
        
        for i in range(num_tiles1):
            for j in range(num_tiles2):
                # 输入切片
                start1 = i * size1
                end1 = min((i + 1) * size1, original_shape[dim1])
                start2 = j * size2
                end2 = min((j + 1) * size2, original_shape[dim2])
                
                input_slice = [slice(None)] * 5
                input_slice[dim1] = slice(start1, end1)
                input_slice[dim2] = slice(start2, end2)
                
                # 处理tile
                tile_input = tensor[tuple(input_slice)]
                
                with torch.no_grad():
                    tile_output = process_fn(tile_input)
                
                # 写入输出
                output_slice = [slice(None)] * 5
                output_slice[dim1] = slice(start1, end1)
                output_slice[dim2] = slice(start2, end2)
                output[tuple(output_slice)] = tile_output
                
                # 立即清理
                self._memory_cleanup(tile_input, tile_output)
        
        return output
    
    def groupnorm_aware_tile(self,
                           tensor: torch.Tensor,
                           process_fn: Callable[[torch.Tensor], torch.Tensor],
                           preferred_dims: List[int] = [0, 2, 3, 4],  # 避免沿channel维度切分
                           max_memory_mb: float = 512) -> torch.Tensor:
        """
        针对GroupNorm优化的自适应切分
        
        Args:
            tensor: 输入tensor
            process_fn: 处理函数
            preferred_dims: 优先切分的维度（避免channel维度）
            max_memory_mb: 最大显存使用量
        """
        if self.profiler:
            self.profiler.log_memory("groupnorm_tile_start")
        
        # 估算tensor大小
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        if tensor_size_mb <= max_memory_mb:
            with torch.no_grad():
                with record_function("direct_processing"):
                    result = process_fn(tensor)
                    if self.profiler:
                        self.profiler.log_memory("direct_processing_complete")
                    return result
        
        # 选择切分策略
        shape = tensor.shape
        
        # 优先选择非channel维度中最大的维度
        best_dim = None
        best_size = 0
        
        for dim in preferred_dims:
            if dim < len(shape) and shape[dim] > best_size:
                best_dim = dim
                best_size = shape[dim]
        
        if best_dim is None:
            best_dim = max(range(len(shape)), key=lambda x: shape[x])
        
        # 计算tile大小
        reduction_factor = math.ceil(tensor_size_mb / max_memory_mb)
        tile_size = max(1, shape[best_dim] // reduction_factor)
        
        print(f"切分维度: {best_dim}, 原始大小: {shape[best_dim]}, tile大小: {tile_size}")
        print(f"预估显存减少: {tensor_size_mb:.1f}MB -> {tensor_size_mb/reduction_factor:.1f}MB")
        
        return self.single_dim_tile_optimized(tensor, process_fn, best_dim, tile_size)
    
    def progressive_tile(self,
                        tensor: torch.Tensor,
                        process_fn: Callable[[torch.Tensor], torch.Tensor],
                        target_memory_mb: float = 256,
                        memory_monitor: bool = True) -> torch.Tensor:
        """
        渐进式切分，动态调整tile大小
        """
        current_memory = self._get_memory_usage()
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        # 如果当前显存使用已经很高，使用更激进的切分
        if current_memory > target_memory_mb:
            target_memory_mb = target_memory_mb * 0.5
        
        print(f"当前显存使用: {current_memory:.1f}MB")
        print(f"目标显存限制: {target_memory_mb:.1f}MB")
        
        # 尝试不同的切分策略
        strategies = [
            (0, 1),      # batch维度，每次处理1个
            (2, 8),      # depth维度，每次处理8个
            (3, 32),     # height维度，每次处理32个
            (4, 32),     # width维度，每次处理32个
        ]
        
        for dim, initial_tile_size in strategies:
            if tensor.shape[dim] <= initial_tile_size:
                continue
                
            # 估算这种策略的显存使用
            estimated_mb = tensor_size_mb * (initial_tile_size / tensor.shape[dim])
            
            if estimated_mb <= target_memory_mb:
                print(f"选择策略: 维度{dim}, tile大小{initial_tile_size}")
                return self.single_dim_tile_optimized(tensor, process_fn, dim, initial_tile_size)
        
        # 如果所有策略都不行，使用最激进的策略
        print("使用最激进的切分策略")
        return self.single_dim_tile_optimized(tensor, process_fn, 0, 1)


# 针对GroupNorm优化的ResNet Sequential
class GroupNormResNetSequential(nn.Module):
    """使用GroupNorm的ResNet Sequential模块"""
    def __init__(self, in_channels: int, num_groups: int = 8):
        super().__init__()
        # 使用GroupNorm替代BatchNorm
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x  # Initialize h with the input x
        h = self.norm1(h)  # Apply GroupNorm
        h = F.relu(h)  # Apply non-linear activation
        
        # Pad h with zeros in spatial and temporal dimensions
        h = F.pad(h, (1, 1, 1, 1, 2, 0), mode="constant", value=0)
        h = self.conv1(h)
        
        return h


# 显存监控装饰器
def memory_monitor(func):
    """显存监控装饰器"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"显存变化: {start_memory:.1f}MB -> {end_memory:.1f}MB (Δ{end_memory-start_memory:+.1f}MB)")
            print(f"峰值显存: {peak_memory:.1f}MB")
            
        return result
    return wrapper


def detailed_profiling_test():
    """详细的profiling测试"""
    if not torch.cuda.is_available():
        print("需要CUDA环境进行显存测试")
        return
    
    device = torch.device('cuda')
    
    # 测试配置
    test_configs = [
        {"batch": 2, "channels": 16, "depth": 32, "height": 128, "width": 128, "name": "小规模"},
        {"batch": 4, "channels": 32, "depth": 64, "height": 256, "width": 256, "name": "中规模"},
        {"batch": 8, "channels": 64, "depth": 128, "height": 512, "width": 512, "name": "大规模"},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"形状: ({config['batch']}, {config['channels']}, {config['depth']}, {config['height']}, {config['width']})")
        
        # 创建数据
        tensor = torch.randn(config['batch'], config['channels'], config['depth'], 
                           config['height'], config['width'], device=device)
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        print(f"Tensor大小: {tensor_size_mb:.1f}MB")
        
        # 创建模型
        model = GroupNormResNetSequential(config['channels'], num_groups=8).to(device)
        model.eval()
        
        def process_fn(x):
            return model(x)
        
        # 测试1: 直接处理 (如果显存够用)
        if tensor_size_mb < 500:  # 只在小规模数据上测试直接处理
            print(f"\n--- 直接处理 ---")
            torch.cuda.reset_peak_memory_stats()
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            ) as prof:
                with record_function("direct_processing"):
                    with torch.no_grad():
                        direct_result = process_fn(tensor)
            
            print(f"峰值显存: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB")
            
            # 保存profile结果
            prof.export_chrome_trace(f"direct_processing_{config['name']}.json")
            print(f"直接处理profile已保存: direct_processing_{config['name']}.json")
            
            del direct_result
            torch.cuda.empty_cache()
        
        # 测试2: Tile处理
        print(f"\n--- Tile处理 ---")
        tiler = MemoryOptimizedTiler(enable_profiling=True)
        torch.cuda.reset_peak_memory_stats()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            with record_function("tile_processing"):
                with torch.no_grad():
                    tile_result = tiler.groupnorm_aware_tile(
                        tensor, process_fn, max_memory_mb=200
                    )
        
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB")
        
        # 保存profile结果
        prof.export_chrome_trace(f"tile_processing_{config['name']}.json")
        print(f"Tile处理profile已保存: tile_processing_{config['name']}.json")
        
        # 显示详细的显存日志
        if tiler.profiler:
            summary = tiler.profiler.get_summary()
            print(f"显存峰值: {summary['peak_allocated']:.1f}MB")
            print(f"最终显存: {summary['final_allocated']:.1f}MB")
            print(f"操作数量: {summary['operations']}")
            
            # 绘制显存使用图
            try:
                tiler.profiler.plot_memory_usage(f"memory_usage_{config['name']}.png")
            except Exception as e:
                print(f"绘图失败: {e}")
        
        # 清理
        del tensor, tile_result, tiler
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("所有测试完成!")
    print("生成的文件:")
    print("1. *.json - Chrome trace文件，可在chrome://tracing中查看")
    print("2. memory_usage_*.png - 显存使用图表")


# 使用示例
@memory_monitor
def optimized_example():
    """优化后的使用示例"""
    
    # 创建大tensor测试
    batch_size, channels, depth, height, width = 4, 32, 64, 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    print(f"创建tensor形状: {(batch_size, channels, depth, height, width)}")
    
    tensor = torch.randn(batch_size, channels, depth, height, width, device=device)
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    print(f"Tensor大小: {tensor_size_mb:.1f}MB")
    
    # 创建GroupNorm模型
    model = GroupNormResNetSequential(channels, num_groups=8).to(device)
    model.eval()
    
    # 定义处理函数
    def process_fn(x):
        return model(x)
    
    # 创建优化的tiler
    tiler = MemoryOptimizedTiler(enable_gc=True, clear_cache=True, enable_profiling=True)
    
    # 测试不同的切分策略
    print("\n=== GroupNorm感知的自适应切分 ===")
    with torch.no_grad():
        result1 = tiler.groupnorm_aware_tile(tensor, process_fn, max_memory_mb=200)
        print(f"结果形状: {result1.shape}")
    
    # 显示profiler结果
    if tiler.profiler:
        summary = tiler.profiler.get_summary()
        print(f"显存使用摘要: {summary}")
        
        # 绘制显存使用图
        try:
            tiler.profiler.plot_memory_usage("memory_usage_example.png")
        except Exception as e:
            print(f"绘图失败 (可能缺少matplotlib): {e}")
    
    # 清理结果
    tiler._memory_cleanup(result1, tensor, operation="final_cleanup")
    
    print(f"\n显存优化完成!使用设备: {device}")
    print(f"创建tensor形状: {(batch_size, channels, depth, height, width)}")
    
    tensor = torch.randn(batch_size, channels, depth, height, width, device=device)
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    print(f"Tensor大小: {tensor_size_mb:.1f}MB")
    
    # 创建GroupNorm模型
    model = GroupNormResNetSequential(channels, num_groups=8).to(device)
    model.eval()
    
    # 定义处理函数
    def process_fn(x):
        return model(x)
    
    # 创建优化的tiler
    tiler = MemoryOptimizedTiler(enable_gc=True, clear_cache=True)
    
    # 测试不同的切分策略
    print("\n=== GroupNorm感知的自适应切分 ===")
    with torch.no_grad():
        result1 = tiler.groupnorm_aware_tile(tensor, process_fn, max_memory_mb=200)
        print(f"结果形状: {result1.shape}")
    
    # 清理结果
    tiler._memory_cleanup(result1)
    
    print("\n=== 渐进式切分 ===")
    with torch.no_grad():
        result2 = tiler.progressive_tile(tensor, process_fn, target_memory_mb=150)
        print(f"结果形状: {result2.shape}")
    
    # 清理
    tiler._memory_cleanup(result2, tensor)
    
    print("\n显存优化完成!")


if __name__ == "__main__":
    optimized_example()
