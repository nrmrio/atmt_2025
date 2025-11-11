
class ConstantWarmup:
    def __init__(self, optimizer, base_lr, warmup_steps):
        self.optim = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, int(warmup_steps))
        self.step_num = 0
    def step(self):
        self.step_num += 1
        scale = min(1.0, self.step_num / self.warmup_steps)
        lr = self.base_lr * scale
        for g in self.optim.param_groups:
            g['lr'] = lr

class LinearWarmupDecay:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps=None):
        self.optim = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = total_steps
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            scale = self.step_num / self.warmup_steps
        elif self.total_steps:
            remain = max(1, self.total_steps - self.warmup_steps)
            after = min(self.step_num - self.warmup_steps, remain)
            scale = max(0.0, 1.0 - after / remain)
        else:
            scale = 1.0
        lr = self.base_lr * scale
        for g in self.optim.param_groups:
            g['lr'] = lr
