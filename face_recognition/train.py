"""Train ResNet50-IR with ArcFace on a faces_emore RecordIO dataset."""
import argparse
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from face_recognition.arcface import ArcMarginProduct
from face_recognition.config import discover_recordio_files, read_num_classes
from face_recognition.model import ResNet50IR
from face_recognition.recordio import RecordIODataset

def checkpoint_name(kind):
    if kind not in {"best", "last"}: raise ValueError("checkpoint kind must be best or last")
    return f"{kind}.pt"

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path(".")); parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet50_arcface")); parser.add_argument("--device", default="cuda"); parser.add_argument("--max-samples", type=int); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args(); rec, idx = discover_recordio_files(args.data_root); device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((112,112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    dataset = RecordIODataset(rec, idx, transform, args.max_samples)
    classes = read_num_classes(rec)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    encoder, head = ResNet50IR().to(device), ArcMarginProduct(512, classes).to(device); optimizer = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4); loss_fn = nn.CrossEntropyLoss(); args.output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        encoder.train(); head.train(); total = 0.0; started = time.perf_counter()
        for batch, (images, target) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(head(encoder(images), target), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += loss.item()
            if batch % args.log_interval == 0:
                rate = batch * args.batch_size / (time.perf_counter() - started)
                print(f"epoch={epoch + 1}/{args.epochs} batch={batch}/{len(loader)} loss={total / batch:.4f} images_per_sec={rate:.1f}", flush=True)
        state = {"epoch": epoch + 1, "encoder": encoder.state_dict(), "arcface": head.state_dict(), "optimizer": optimizer.state_dict()}; torch.save(state, args.output_dir / checkpoint_name("last")); print(f"epoch={epoch + 1} loss={total / len(loader):.4f}")
if __name__ == "__main__": main()
