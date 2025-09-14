import matplotlib.pyplot as plt


class RiskMapVisualizer:
    def __init__(self, title="RiskMap", cmap='hot', interpolation='nearest', plt_show=False):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = None
        self.title = title
        self.cmap = cmap
        self.interpolation = interpolation
        plt.ion()  # 开启交互模式
        if plt_show:
            plt.show()

    def update(self, vis_data):
        """
        vis_data: dict，包含 'risk_avg' 和 'extent'
        """
        risk_avg = vis_data['risk_avg']
        extent = vis_data['extent']

        if self.im is None:
            # 第一次绘制
            self.im = self.ax.imshow(risk_avg, origin='lower', extent=extent,
                                     cmap=self.cmap, interpolation=self.interpolation, aspect='auto')
            self.ax.set_title(self.title)
            self.ax.set_xlabel("local x (m)")
            self.ax.set_ylabel("local y (m)")
            plt.colorbar(self.im, ax=self.ax)
            plt.tight_layout()
        else:
            # 更新已有图像
            self.im.set_data(risk_avg)
            self.im.set_extent(extent)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
