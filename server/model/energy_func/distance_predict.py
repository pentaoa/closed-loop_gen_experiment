import os
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class FeatureDistanceDataset(Dataset):
    def __init__(self, eeg_feature_dir, clip_feature_path, subject, train=True):
        """
        初始化数据集
        
        Args:
            eeg_feature_dir: EEG feature directory
            clip_feature_path: CLIP feature file path
            subject: subject ID (e.g. 'sub-01')
            train: whether this is training set
        """
        self.train = train
        mode = "train" if train else "test"
        
        # Load EEG features
        eeg_feature_path = os.path.join(eeg_feature_dir, subject, f"eeg_features_ATMS_{mode}.pt")
        eeg_data = torch.load(eeg_feature_path)
        self.eeg_features = eeg_data['features']
        self.eeg_labels = eeg_data['labels']
        
        # Load CLIP image features
        clip_data = torch.load(clip_feature_path)
        self.clip_features = clip_data['img_features']
        
        print(f"EEG features shape: {self.eeg_features.shape}")
        print(f"EEG labels shape: {self.eeg_labels.shape}")
        print(f"CLIP features shape: {self.clip_features.shape}")
        
        # Build positive/negative pairs
        self.samples = self._create_samples()
        
    def _create_samples(self):
        """Create paired samples with correct matching."""
        samples = []
        label_to_indices = {}
        for i, label in enumerate(self.eeg_labels):
            l = label.item()
            label_to_indices.setdefault(l, []).append(i)
        unique_labels = sorted(label_to_indices.keys())
        
        for i, lab1 in enumerate(tqdm(unique_labels, desc="Building pairs")):
            idxs1 = label_to_indices[lab1]
            img1 = self.clip_features[lab1]
            for lab2 in unique_labels:
                if lab1 == lab2:
                    continue
                idxs2 = label_to_indices[lab2]
                img2 = self.clip_features[lab2]
                # cosine distance between the two CLIP features
                dist = 1.0 - F.cosine_similarity(img1.unsqueeze(0), img2.unsqueeze(0), dim=1).item()
                for i1 in idxs1:
                    for i2 in idxs2:
                        samples.append({
                            'eeg_feature1': self.eeg_features[i1],
                            'img_feature1': img1,
                            'eeg_feature2': self.eeg_features[i2],
                            'img_feature2': img2,
                            'img_distance': dist,
                            'label1': lab1,
                            'label2': lab2
                        })
        print(f"Built {len(samples)} samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['eeg_feature1'],
            s['img_feature1'],
            s['eeg_feature2'],
            s['img_distance'],
            s['img_feature2']  # only for evaluation
        )

class FeatureDistancePredictor(nn.Module):
    def __init__(self, eeg_dim=512, img_dim=1024, hidden_dim=512):
        """
        Initialize distance predictor network.
        
        Args:
            eeg_dim: dimension of EEG features
            img_dim: dimension of CLIP image features
            hidden_dim: hidden layer size
        """
        super().__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(eeg_dim * 2 + img_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, eeg1, img1, eeg2):
        combined = torch.cat([eeg1, img1, eeg2], dim=1)
        d = self.fusion_net(combined)
        # map to [0, 2]
        return (torch.sigmoid(d) * 2).squeeze(1)

def train_model(model, train_loader, val_loader, device, args, subject):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val = float('inf')
    train_losses, val_losses = [], []
    save_dir = os.path.join(args.output_dir, subject, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        running = 0.
        for bidx, (e1, i1, e2, tgt, _) in enumerate(train_loader, 1):
            e1, i1, e2, tgt = e1.to(device), i1.to(device), e2.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(e1, i1, e2)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            if bidx % 50 == 0:
                print(f"[{subject}] Epoch {epoch+1}/{args.epochs} Batch {bidx}/{len(train_loader)} Loss {loss.item():.4f}")
        avg_tr = running / len(train_loader)
        train_losses.append(avg_tr)
        
        model.eval()
        vl, preds, targs = 0., [], []
        with torch.no_grad():
            for e1, i1, e2, tgt, _ in val_loader:
                e1, i1, e2, tgt = e1.to(device), i1.to(device), e2.to(device), tgt.to(device)
                out = model(e1, i1, e2)
                vl += criterion(out, tgt).item()
                preds.extend(out.cpu().numpy())
                targs.extend(tgt.cpu().numpy())
        avg_val = vl / len(val_loader)
        val_losses.append(avg_val)
        
        mse = mean_squared_error(targs, preds)
        mae = mean_absolute_error(targs, preds)
        r2  = r2_score(targs, preds)
        print(f"[{subject}] Epoch {epoch+1}/{args.epochs} Train {avg_tr:.4f} Val {avg_val:.4f} MSE {mse:.4f} MAE {mae:.4f} R2 {r2:.4f}")
        
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"[{subject}] Saved best model")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
    
    # plot losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{subject} Training/Validation Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    
    return model, save_dir

def evaluate_model(model, test_loader, device, save_dir, subject):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for e1, i1, e2, tgt, _ in test_loader:
            e1, i1, e2 = e1.to(device), i1.to(device), e2.to(device)
            out = model(e1, i1, e2)
            preds.extend(out.cpu().numpy())
            targs.extend(tgt.numpy())
    preds = np.array(preds); targs = np.array(targs)
    mse = mean_squared_error(targs, preds)
    mae = mean_absolute_error(targs, preds)
    r2  = r2_score(targs, preds)
    print(f"[{subject}] Test  MSE {mse:.4f} MAE {mae:.4f} R2 {r2:.4f}")
    
    # scatter
    plt.figure(figsize=(8,8))
    plt.scatter(targs, preds, alpha=0.3)
    z = np.polyfit(targs, preds, 1); p = np.poly1d(z)
    xs = np.sort(targs)
    plt.plot(xs, p(xs), 'r--')
    maxv = max(targs.max(), preds.max())
    plt.plot([0,maxv],[0,maxv],'g-')
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"{subject} Predictions (MSE={mse:.4f})")
    plt.savefig(os.path.join(save_dir, "test_scatter.png"))
    
    # histograms
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(targs, bins=30, alpha=0.5, label='True')
    plt.hist(preds, bins=30, alpha=0.5, label='Pred')
    plt.legend(); plt.title("Distance Distribution")
    plt.subplot(1,2,2)
    errs = preds - targs
    plt.hist(errs, bins=30)
    plt.title(f"Error dist (μ={errs.mean():.4f} σ={errs.std():.4f})")
    plt.savefig(os.path.join(save_dir, "test_hist.png"))
    
    # save metrics
    with open(os.path.join(save_dir, "test_metrics.txt"), "w") as f:
        f.write(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\n")
    return

def main():
    parser = argparse.ArgumentParser(description='Feature Distance Predictor Training')
    parser.add_argument('--eeg_feature_dir', type=str,
                        default="/mnt/dataset1/ldy/Workspace/Closed_loop_optimizing/kyw/closed-loop/EEG_feature/sub_features",
                        help='Directory containing per-subject EEG feature folders')
    parser.add_argument('--clip_feature_path', type=str,
                        default="/mnt/dataset1/ldy/Workspace/EEG_Image_decode_Wrong/Retrieval/ViT-H-14_features_train.pt",
                        help='Path to training CLIP features (.pt)')
    parser.add_argument('--clip_feature_test_path', type=str,
                        default="/mnt/dataset1/ldy/Workspace/EEG_Image_decode_Wrong/Retrieval/ViT-H-14_features_train.pt",
                        help='Path to test CLIP features (.pt)')
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/dataset1/ldy/Workspace/Closed_loop_optimizing/models/distance_predictor",
                        help='Where to save models and results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computation device (e.g. "cuda:0" or "cpu")')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--subjects', type=str, default='sub-01',
                        help='Comma-separated subject IDs (e.g. "sub-01,sub-02") or "all"')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of training set to use for validation')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only run test (requires --model_path)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pretrained model for testing only')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # prepare subject list
    if args.subjects.lower() == "all":
        subs = sorted(os.listdir(args.eeg_feature_dir))
    else:
        subs = [s.strip() for s in args.subjects.split(",")]
    
    for subject in subs:
        print(f"\n===== Processing {subject} =====")
        subj_out = os.path.join(args.output_dir, subject)
        os.makedirs(subj_out, exist_ok=True)
        
        if not args.test_only:
            # load and split train dataset
            train_ds = FeatureDistanceDataset(args.eeg_feature_dir, args.clip_feature_path, subject, train=True)
            total = len(train_ds)
            val_n = int(total * args.val_ratio)
            tr_n  = total - val_n
            train_sub, val_sub = torch.utils.data.random_split(train_ds, [tr_n, val_n])
            tr_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4)
            v_loader  = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            # init model
            sample = train_ds[0]
            eeg_dim = sample[0].shape[0]
            img_dim = sample[1].shape[0]
            model = FeatureDistancePredictor(eeg_dim=eeg_dim, img_dim=img_dim)
            model.to(device)
            print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            
            # train
            model, save_dir = train_model(model, tr_loader, v_loader, device, args, subject)
            model_path = os.path.join(save_dir, "best_model.pth")
        else:
            if args.model_path is None:
                raise ValueError("When using --test_only you must specify --model_path")
            model = FeatureDistancePredictor()  # dims will be overwritten on load
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            save_dir = os.path.dirname(args.model_path)
        
        # test
        test_ds = FeatureDistanceDataset(args.eeg_feature_dir, args.clip_feature_test_path, subject, train=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        evaluate_model(model, test_loader, device, save_dir, subject)
    
    print("\nAll done!")

if __name__ == "__main__":
    main()


# def predict_distance(model, eeg_feature1, img_feature1, eeg_feature2, device):
#     """
#     预测两个图像特征之间的距离
#
#     Args:
#         model: 训练好的模型
#         eeg_feature1: 第一个图像的EEG特征
#         img_feature1: 第一个图像的CLIP特征
#         eeg_feature2: 第二个图像的EEG特征
#         device: 计算设备
#
#     Returns:
#         predicted_distance: 预测的距离
#     """
#     model.eval()
#
#     # 确保输入是张量并移动到正确的设备
#     if not isinstance(eeg_feature1, torch.Tensor):
#         eeg_feature1 = torch.tensor(eeg_feature1, dtype=torch.float32)
#     if not isinstance(img_feature1, torch.Tensor):
#         img_feature1 = torch.tensor(img_feature1, dtype=torch.float32)
#     if not isinstance(eeg_feature2, torch.Tensor):
#         eeg_feature2 = torch.tensor(eeg_feature2, dtype=torch.float32)
#
#     # 添加批次维度
#     if eeg_feature1.dim() == 1:
#         eeg_feature1 = eeg_feature1.unsqueeze(0)
#     if img_feature1.dim() == 1:
#         img_feature1 = img_feature1.unsqueeze(0)
#     if eeg_feature2.dim() == 1:
#         eeg_feature2 = eeg_feature2.unsqueeze(0)
#
#     # 移动到设备
#     eeg_feature1 = eeg_feature1.to(device)
#     img_feature1 = img_feature1.to(device)
#     eeg_feature2 = eeg_feature2.to(device)
#
#     # 预测距离
#     with torch.no_grad():
#         predicted_distance = model(eeg_feature1, img_feature1, eeg_feature2)
#
#     return predicted_distance.item()