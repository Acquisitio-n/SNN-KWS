# train.py  【混合精度 + 梯度累积 + 防二次 backward】
import torch, argparse, tqdm, os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler      # ← 新增
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional
from dataset import GSC12
from model import KWS_SNN

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--T", type=int, default=8)
parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

def main():
    train_set = GSC12(args.data, "training")
    val_set   = GSC12(args.data, "validation")
    train_ld = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_set, batch_size=args.bs, shuffle=False,
                          num_workers=4, pin_memory=True)

    net = KWS_SNN(n_class=12, T=args.T).to(args.device)
    print("Params:", sum(p.numel() for p in net.parameters()))

    opt      = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)
    loss_fn   = torch.nn.CrossEntropyLoss()
    writer    = SummaryWriter()
    scaler    = GradScaler()                       # ← 混合精度
    best_acc  = 0.0

    for epoch in range(args.epoch):
        # ---------- train ----------
        net.train()
        correct, total, loss_sum = 0, 0, 0.0
        for batch_idx, (wav, lbl) in enumerate(tqdm.tqdm(train_ld, desc=f"Train {epoch}")):
            wav, lbl = wav.to(args.device, non_blocking=True), lbl.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast():                      # ← FP16 前向
                out = net(wav)                    # T 步一次性展开
                loss = loss_fn(out, lbl)

            scaler.scale(loss).backward()         # ← FP16 梯度
            correct  += (out.argmax(1) == lbl).sum().item()
            total    += lbl.size(0)
            loss_sum += loss.item()

            # 梯度累积：每 accum 步才更新权重
            if (batch_idx + 1) % args.accum == 0 or (batch_idx + 1) == len(train_ld):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            functional.reset_net(net)             # ← 清膜电位 & 图

        tr_acc = correct / total
        writer.add_scalar("train/acc",  tr_acc, epoch)
        writer.add_scalar("train/loss", loss_sum / len(train_ld), epoch)

        # ---------- val ----------
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for wav, lbl in val_ld:
                wav, lbl = wav.to(args.device, non_blocking=True), lbl.to(args.device, non_blocking=True)
                with autocast():                # 推理也 FP16
                    out = net(wav)
                correct += (out.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
                functional.reset_net(net)
        val_acc = correct / total
        writer.add_scalar("val/acc", val_acc, epoch)
        scheduler.step()

        print(f"Epoch {epoch:02d}  train={tr_acc:.3f}  val={val_acc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  accum={args.accum}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), "best.pt")

    print("Best val acc:", best_acc)
    writer.close()

if __name__ == "__main__":
    main()