import matplotlib.pyplot as plt

def plot_loss():
    # 设置中文字体（如果是 Linux 虚拟机可能不带中文字体，建议用英文标注更稳）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
    
    try:
        with open('loss_bn.txt', 'r') as f:
            losses = [float(line.strip()) for line in f]
    except FileNotFoundError:
        print("错误：找不到 loss_bn.txt，请先运行 train.py")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='#1f77b4', linewidth=2, label='CAE + Batch Norm')
    
    # 修饰图表
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Convergence Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存高清图
    plt.savefig('loss_curve_result.png', dpi=300, bbox_inches='tight')
    print("论文插图已生成：loss_curve_result.png")

if __name__ == "__main__":
    plot_loss()