import torch
from torch.optim.optimizer import Optimizer, required


class ActiveSGD(Optimizer):

    def __init__(self, params, stepSize, lr=required, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, lrLow=0.9, lrHigh=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, stepSize=stepSize, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        lrLow=0.9, lrHigh=0.1)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ActiveSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ActiveSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lrLow = group['lrLow']
            lrHigh = group['lrHigh']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    # d_p = d_p.add(p, alpha=weight_decay)
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                
                param_state = self.state[p]
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0

                    param_state['gai'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    param_state['cumm'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()

                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                # Accumulate gradients for the epoch
                param_state['cumm']+=(p.grad)

                param_state['step'] += 1

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                # p.add_(d_p, alpha=-group['lr'])
                # parameter update
                p -= group['lr']*param_state['gai']*d_p


                # SetLR if i>0
                if param_state['step']/group['stepSize'] > 1 and param_state['step']%group['stepSize']==0:
                    tmp2 = param_state['gradOld'].clone()##could be eliminated
                    tmp3 = param_state['cumm'].clone()##could be eliminated
                    tmp5 = param_state['gai'].clone()##may be the one that needs cloning

                    param_state['gai'] = torch.where(tmp2*tmp3<=0, tmp5*group['lrLow'], tmp5+group['lrHigh'])
                    gai = param_state['gai']
                    # print(f'gradOld {tmp2}, cumm {tmp3}, gai {gai}')


                # Resetting the accumulated gradients after each epoch
                if param_state['step']%group['stepSize']==0:
                    cumm = param_state['cumm']
                    param_state['gradOld'] = cumm.clone()
                    param_state['cumm'] = torch.zeros_like(p, memory_format=torch.preserve_format)


        return loss