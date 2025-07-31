# train_bilstm_meta_and_baselines.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu') # MAC 환경 우선 설정
print("using", device)

DYNAMIC_COLS = [      # 시간에 따라 변하는 센서·제어 변수
    'fan','co2','heater','window1','window2',
    'curtain1','curtain2','curtain3','side_curtain',
    'in_temp','in_hum','in_co2','out_temp','out_hum',
    'solar_rad','wind_speed','wind_direction','rain_sensor'
]
META_COLS = [         # 식물 개체 고정 정보
    'crown_diameter','petiole_length','leaf_count','leaf_length','leaf_width',
    'fruit_count','plant_height','flower_count','numbers of plant'
]

def make_dataset(x_df, y_df, sample_col='Sample_Number', time_col='time'): # 동적 X_seq, 정적 X_meta, 라벨 y
    x_df['_sort_time'] = x_df[time_col]
    seq_list, meta_list, y_list = [], [], []
    for sn, g in x_df.groupby(sample_col):
        g = g.sort_values('_sort_time')
        X_seq = g[DYNAMIC_COLS].values
        meta = g[META_COLS].iloc[0].values.astype(float)
        if sn not in y_df[sample_col].values: continue
        y_val = y_df.loc[y_df[sample_col]==sn, 'CO2 final'].values[0]
        seq_list.append(X_seq)
        meta_list.append(meta)
        y_list.append(y_val)
    # pad 시퀀스
    T_max, Din = max(x.shape[0] for x in seq_list), seq_list[0].shape[1]
    seq_pad = np.zeros((len(seq_list), T_max, Din))
    for i, arr in enumerate(seq_list): seq_pad[i, :arr.shape[0], :] = arr
    return seq_pad, np.asarray(meta_list), np.asarray(y_list)

class SeqMetaDataset(Dataset):
    def __init__(self, X_seq, X_meta, y):
        self.Xs = torch.tensor(X_seq ,dtype=torch.float32)
        self.Xm = torch.tensor(X_meta,dtype=torch.float32)
        self.y  = torch.tensor(y     ,dtype=torch.float32).view(-1,1)

    def __len__(self): return len(self.Xs)

    def __getitem__(self, i): return self.Xs[i], self.Xm[i], self.y[i]

# ─────────────────────────────────── 2 트랙 모델 ──────────────────────────────────
class BiLSTM_Meta(nn.Module):
    def __init__(self, din_seq, dmeta, dh=64, dmeta_hidden=32):
        super().__init__()
        self.bilstm = nn.LSTM(din_seq, dh, batch_first=True, bidirectional=True)
        self.mlp_meta = nn.Sequential(
            nn.Linear(dmeta, dmeta_hidden), nn.ReLU(), nn.Linear(dmeta_hidden, dmeta_hidden))
        self.head = nn.Sequential(
            nn.Linear(dh*2 + dmeta_hidden, 64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_seq, x_meta):
        h, _ = self.bilstm(x_seq)
        seq_feat = h[:,-1,:]
        meta_feat = self.mlp_meta(x_meta)
        feat = torch.cat([seq_feat, meta_feat], dim=1)
        return self.head(feat)

# 4개 baseline (MLP, CNN, LSTM, GRU) --------------------------
class MLPReg(nn.Module):
    def __init__(self, T, Din, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T*Din, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1))
    def forward(self,x_seq,*_): return self.net(x_seq)

class CNNReg(nn.Module):
    def __init__(self, Din, h=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(Din,32,5), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,16,3), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(16, h), nn.ReLU(), nn.Linear(h,1))
    def forward(self,x_seq,*_):
        x = self.conv(x_seq.permute(0,2,1))
        x = self.pool(x).squeeze(-1)
        return self.head(x)

class LSTMReg(nn.Module):
    def __init__(self, Din, dh=64): super().__init__()
    def __init__(self, Din, dh=64):
        super().__init__()
        self.lstm = nn.LSTM(Din, dh, batch_first=True)
        self.fc = nn.Linear(dh,1)
    def forward(self,x_seq,*_):
        out,_ = self.lstm(x_seq)
        return self.fc(out[:,-1,:])

class GRUReg(nn.Module):
    def __init__(self, Din, dh=64):
        super().__init__()
        self.gru = nn.GRU(Din, dh, batch_first=True)
        self.fc = nn.Linear(dh,1)
    def forward(self,x_seq,*_):
        out,_ = self.gru(x_seq)
        return self.fc(out[:,-1,:])

# ─────────────────────────────────── 학습 루프 ──────────────────────────────────
def train(model, loaders, epochs=30, lr=1e-3):
    train_loader, test_loader = loaders
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xs,xm,y in train_loader:
            xs,xm,y = xs.to(device), xm.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(xs,xm), y)
            loss.backward()
            opt.step()
    # ─ 평가
    model.eval()
    yt,yp = [],[]
    with torch.no_grad():
        for xs,xm,y in test_loader:
            pred = model(xs.to(device), xm.to(device)).cpu().numpy()
            yp.extend(pred)
            yt.extend(y.numpy())
    yt,yp = np.array(yt).flatten(), np.array(yp).flatten()
    return yt,yp

def metrics(yt,yp):
    return dict(
        rmse = np.sqrt(mean_squared_error(yt,yp)),
        r2   = r2_score(yt,yp),
        mae  = mean_absolute_error(yt,yp))

# ─────────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("parameters", exist_ok=True)
    os.makedirs("result_figures", exist_ok=True)

    x_df = pd.read_csv("x_train.csv")
    y_df = pd.read_csv("y_train.csv")

    X_seq, X_meta, y = make_dataset(x_df, y_df)
    # 동적 
    scaler_seq = StandardScaler()
    X_seq = scaler_seq.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
    # 메타
    scaler_meta = StandardScaler()
    X_meta = scaler_meta.fit_transform(X_meta)

    # train / test split
    idx = np.random.permutation(len(X_seq))
    n_tr = int(0.8*len(idx))
    tr_idx, ts_idx = idx[:n_tr], idx[n_tr:]
    train_ds = SeqMetaDataset(X_seq[tr_idx], X_meta[tr_idx], y[tr_idx])
    test_ds  = SeqMetaDataset(X_seq[ts_idx], X_meta[ts_idx], y[ts_idx])
    loaders = (DataLoader(train_ds,32,True), DataLoader(test_ds,32))

    T, Din, Dmeta = X_seq.shape[1], X_seq.shape[2], X_meta.shape[1]

    models = {
        "bilstm_meta": BiLSTM_Meta(Din, Dmeta),
        "cnn" : CNNReg(Din),
        "mlp" : MLPReg(T, Din),
        "lstm": LSTMReg(Din),
        "gru" : GRUReg(Din)
    }

    results = []
    for name, model in models.items():
        print(f"\n {name.upper()} training")
        yt,yp = train(model, loaders, epochs=20, lr=1e-3)
        m = metrics(yt,yp)
        m['model']=name
        results.append(m)
        # 저장
        torch.save(model.state_dict(), f"parameters/{name}.pth")
        # 그림
        plt.figure()
        plt.scatter(yt,yp,alpha=.7)
        mn,mx = yt.min(), yt.max()
        plt.plot([mn,mx],[mn,mx],'k--')
        a,b = np.polyfit(yt,yp,1)
        plt.plot([mn,mx],[a*mn+b,a*mx+b],'r-')
        txt = f"R2={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nMAE={m['mae']:.3f}"
        plt.gca().text(0.02,0.95,txt,transform=plt.gca().transAxes, va='top',bbox=dict(fc='w',alpha=.7))
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"result_figures/{name}.png")
        plt.close()
    
    print("\n=== Summary (by RMSE) ===")
    print(pd.DataFrame(results).sort_values('rmse'))
