# train_bilstm_meta_and_baselines_window_weight.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
print("using", device)

# ────────────────────── 사용자 조정 파라미터 ──────────────────────
WINDOW_SIZE = 36        # 6 h = 36 × 10 min
STRIDE       = 36       # 0 겹침; 18 이면 3 h 겹침

# 동적 센서·제어 변수 가중치
WEIGHTS_DYN = {
    'fan'          : 1.3,
    'co2'          : 1.6,
    'heater'       : 1.1,
    'window1'      : 1.4,
    'window2'      : 1.4,
    'curtain1'     : 1.2,
    'curtain2'     : 1.2,
    'curtain3'     : 1.2,
    'side_curtain' : 1.1,
    'in_temp'      : 1.5,
    'in_hum'       : 1.2,
    'in_co2'       : 2.0,
    'out_temp'     : 0.8,
    'out_hum'      : 0.8,
    'solar_rad'    : 1.8,
    'wind_speed'   : 0.7,
    'wind_direction':0.6,
    'rain_sensor'  : 0.5
}

# 정적(메타) 특성 가중치
WEIGHTS_META = {
    'crown_diameter' : 1.4,
    'petiole_length' : 1.1,
    'leaf_count'     : 1.8,
    'leaf_length'    : 1.3,
    'leaf_width'     : 1.3,
    'numbers of plant': 2.0
}


DYN_COLS = [
    'fan','co2','heater','window1','window2',
    'curtain1','curtain2','curtain3','side_curtain',
    'in_temp','in_hum','in_co2','out_temp','out_hum',
    'solar_rad','wind_speed','wind_direction','rain_sensor'
]
META_COLS = [
    'crown_diameter','petiole_length','leaf_count','leaf_length','leaf_width',
    'fruit_count','plant_height','flower_count','numbers of plant'
]

# ────────────────── Sliding-Window 데이터 생성 ──────────────────
def make_window_dataset(x_df, y_df,
                        window_size=36, stride=36,
                        sample_col='Sample_Number', time_col='time'):
    """동적(윈도우 시퀀스)·메타·y 반환"""
    x_df['_sort_time'] = x_df[time_col]
    seqs, metas, labels = [], [], []
    for sn, g in x_df.groupby(sample_col):
        g = g.sort_values('_sort_time')
        x_arr = g[DYN_COLS].values
        T_all = x_arr.shape[0]
        meta_vec = g[META_COLS].iloc[0].values.astype(float)
        if sn not in y_df[sample_col].values:
            continue
        y_val   = y_df.loc[y_df[sample_col]==sn, 'CO2 final'].values[0]

        # 윈도우 추출
        for start in range(0, T_all - window_size + 1, stride):
            seqs.append(x_arr[start:start+window_size])
            metas.append(meta_vec.copy())
            labels.append(y_val)

    return (np.stack(seqs), np.asarray(metas), np.asarray(labels))

class SeqMetaDataset(Dataset):
    def __init__(self, Xs, Xm, y):
        self.Xs=torch.tensor(Xs,dtype=torch.float32)
        self.Xm=torch.tensor(Xm,dtype=torch.float32)
        self.y =torch.tensor(y ,dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self,i):
        return self.Xs[i],self.Xm[i],self.y[i]

# ────────────────── 2-트랙 Bi-LSTM + 베이스라인 ─────────────────
class BiLSTM_Meta(nn.Module):
    def __init__(self,din_seq,dmeta,dh=64,dmeta_h=32):
        super().__init__()
        self.bilstm=nn.LSTM(din_seq,dh,batch_first=True,bidirectional=True)
        self.meta  =nn.Sequential(nn.Linear(dmeta,dmeta_h),nn.ReLU(),
                                  nn.Linear(dmeta_h,dmeta_h))
        self.fc=nn.Sequential(nn.Linear(dh*2+dmeta_h,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,xs,xm):
        h,_=self.bilstm(xs)
        return self.fc(torch.cat([h[:,-1,:],self.meta(xm)],1))

# class MLPReg(nn.Module):
#     def __init__(self,T,D,h=64):
#         super().__init__()
#         self.net=nn.Sequential(nn.Flatten(),nn.Linear(T*D,h),nn.ReLU(),
#                                nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
#     def forward(self,xs,*_):
#         return self.net(xs)

# class CNNReg(nn.Module):
#     def __init__(self,D,h=64):
#         super().__init__()
#         self.cnn=nn.Sequential(nn.Conv1d(D,32,5),nn.BatchNorm1d(32),nn.ReLU(),
#                                nn.Conv1d(32,16,3),nn.ReLU())
#         self.pool=nn.AdaptiveAvgPool1d(1)
#         self.fc=nn.Sequential(nn.Linear(16,h),nn.ReLU(),nn.Linear(h,1))
#     def forward(self,xs,*_):
#         x=self.cnn(xs.permute(0,2,1))
#         x=self.pool(x).squeeze(-1)
#         return self.fc(x)

# class LSTMReg(nn.Module):
#     def __init__(self,D,dh=64):
#         super().__init__()
#         self.lstm=nn.LSTM(D,dh,batch_first=True)
#         self.fc=nn.Linear(dh,1)
#     def forward(self,xs,*_):
#         o,_=self.lstm(xs)
#         return self.fc(o[:,-1,:])

# class GRUReg(nn.Module):
#     def __init__(self,D,dh=64):
#         super().__init__()
#         self.gru=nn.GRU(D,dh,batch_first=True)
#         self.fc=nn.Linear(dh,1)
#     def forward(self,xs,*_):
#         o,_=self.gru(xs)
#         return self.fc(o[:,-1,:])

# ───────────────────────── 학습 루프 ────────────────────────────
def train(model, loaders, epochs=30, lr=1e-3):
    tr,ts = loaders
    model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr)
    crit=nn.MSELoss()
    for _ in tqdm(range(epochs), desc="Training", leave=True):
        model.train()
        for xs,xm,y in tr:
            xs,xm,y=[t.to(device) for t in (xs,xm,y)]
            opt.zero_grad()
            loss=crit(model(xs,xm),y)
            loss.backward()
            opt.step()

    model.eval()
    yt,yp=[],[]
    with torch.no_grad():
        for xs,xm,y in ts:
            pred=model(xs.to(device),xm.to(device)).cpu().numpy()
            yp.extend(pred)
            yt.extend(y.numpy())
    return np.array(yt).flatten(), np.array(yp).flatten()

def metrics(y,p):
    return dict(rmse=np.sqrt(mean_squared_error(y,p)), r2  =r2_score(y,p), mae =mean_absolute_error(y,p))

# ─────────────────────────── main ──────────────────────────────
if __name__=="__main__":
    os.makedirs("parameters",exist_ok=True)
    os.makedirs("result_figures",exist_ok=True)

    x_df=pd.read_csv("x_train.csv")
    y_df=pd.read_csv("y_train.csv")

    # 데이터 생성 (슬라이딩 윈도우) ----------------------------
    X_seq,X_meta,y = make_window_dataset(x_df, y_df, window_size=WINDOW_SIZE, stride=STRIDE)

    # 가중치 벡터 ----------------------------------------
    w_dyn  = np.ones(len(DYN_COLS))
    w_meta = np.ones(len(META_COLS))
    for k,v in WEIGHTS_DYN.items():
        if k in DYN_COLS:
            w_dyn [DYN_COLS.index(k)]  = v
    for k,v in WEIGHTS_META.items():
        if k in META_COLS:
            w_meta[META_COLS.index(k)] = v

    # 스케일 + 가중치 ----------------------------------------
    sc_seq,sc_meta = StandardScaler(),StandardScaler()
    X_seq = sc_seq.fit_transform(X_seq.reshape(-1,X_seq.shape[-1])).reshape(X_seq.shape)
    X_seq *= w_dyn
    X_meta = sc_meta.fit_transform(X_meta)
    X_meta *= w_meta

    # split ---------------------------------------------------
    idx=np.random.permutation(len(X_seq))
    n_tr=int(.8*len(idx))
    tr,ts = idx[:n_tr], idx[n_tr:]
    loaders=(DataLoader(SeqMetaDataset(X_seq[tr],X_meta[tr],y[tr]),32,True),
             DataLoader(SeqMetaDataset(X_seq[ts],X_meta[ts],y[ts]),32))

    T,D,Dm = X_seq.shape[1],X_seq.shape[2],X_meta.shape[1]
    models={
        "bilstm_meta":BiLSTM_Meta(D,Dm),
        # "cnn" :CNNReg(D),
        # "mlp" :MLPReg(T,D),
        # "lstm":LSTMReg(D),
        # "gru" :GRUReg(D)
    }

    results=[]
    for n,m in models.items():
        print(f"\n▶ {n.upper()} training")
        yt,yp = train(m,loaders,epochs=20)
        r=metrics(yt,yp)|{'model':n}
        results.append(r)
        torch.save(m.state_dict(),f"parameters/{n}.pth")

        plt.figure()
        plt.scatter(yt,yp,alpha=.7)
        mn,mx=yt.min(),yt.max()
        plt.plot([mn,mx],[mn,mx],'k--')
        a,b=np.polyfit(yt,yp,1)
        plt.plot([mn,mx],[a*mn+b,a*mx+b],'r-')
        txt=f"R2={r['r2']:.3f}\nRMSE={r['rmse']:.3f}\nMAE={r['mae']:.3f}"
        plt.gca().text(.02,.95,txt,transform=plt.gca().transAxes,va='top',
                       bbox=dict(fc='w',alpha=.7))
        plt.title(n)
        plt.tight_layout()
        plt.savefig(f"result_figures/{n}.png")
        plt.close()

    print("\n=== Summary (by RMSE) ===")
    print(pd.DataFrame(results).sort_values('rmse'))