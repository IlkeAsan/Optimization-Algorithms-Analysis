import json
import math
import random
import time
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# =========================================================
# BÖLÜM 1: VEKTÖR İŞLEMLERİ
# =========================================================

def vektor_carpimi(v1, v2):
    toplam = 0.0
    for i in range(len(v1)):
        toplam += v1[i] * v2[i]
    return toplam

def vektor_toplama(v1, v2):
    yeni = []
    for i in range(len(v1)):
        yeni.append(v1[i] + v2[i])
    return yeni

def vektor_cikarma(v1, v2):
    yeni = []
    for i in range(len(v1)):
        yeni.append(v1[i] - v2[i])
    return yeni

def sayi_carpma(vektor, sayi):
    yeni = []
    for eleman in vektor:
        yeni.append(eleman * sayi)
    return yeni

# --- Adam İçin ---
def vektor_eleman_carpimi(v1, v2):
    yeni = []
    for i in range(len(v1)):
        yeni.append(v1[i] * v2[i])
    return yeni

def vektor_bolme_ozel(v_pay, v_payda, epsilon):
    yeni = []
    for i in range(len(v_pay)):
        payda_degeri = math.sqrt(v_payda[i]) + epsilon
        yeni.append(v_pay[i] / payda_degeri)
    return yeni

# --- BAŞARI HESAPLAMA (ACCURACY) ---
def skor_hesapla(w, X_data, Y_data):
    dogru = 0
    for i in range(len(X_data)):
        z = vektor_carpimi(w, X_data[i])
        pred = math.tanh(z)
        # +1 veya -1 kararı
        tahmin = 1.0 if pred >= 0 else -1.0
        if tahmin == Y_data[i]:
            dogru += 1
    return (dogru / len(X_data)) * 100

# =========================================================
# BÖLÜM 2: VERİ YÜKLEME
# =========================================================

def veri_yukle(dosya_adi):
    print(f"'{dosya_adi}' aranıyor...")
    try:
        with open(dosya_adi, 'r', encoding='utf-8') as f:
            veri = json.load(f)
        X = [d['x'] for d in veri]
        Y = [d['y'] for d in veri]
        print(f"Başarılı! {len(X)} satır veri yüklendi.")
        return X, Y
    except FileNotFoundError:
        print(f"HATA: '{dosya_adi}' bulunamadı!")
        return [], []

# =========================================================
# BÖLÜM 3: ALGORİTMALAR
# =========================================================

# 1. GD
def train_gd(X, Y, X_test, Y_test, epochs, lr, w_init, deneme_no):
    w = copy.deepcopy(w_init)
    N = len(X)
    dim = len(w)
    
    loss_hist = []
    time_hist = []
    acc_hist = [] # Test Başarısı
    w_hist = [] 

    start_time = time.time()
    print(f"  [GD - Run {deneme_no}] Başlıyor...")
    
    for epoch in range(epochs):
        toplam_hata_karesi = 0.0
        grad_sum = [0.0] * dim
        
        for i in range(N):
            x = X[i]; y_true = Y[i]
            z = vektor_carpimi(w, x); pred = math.tanh(z)
            err = pred - y_true; toplam_hata_karesi += err**2
            
            d_tanh = 1.0 - pred**2; grad_factor = err * d_tanh
            grad_i = sayi_carpma(x, grad_factor)
            grad_sum = vektor_toplama(grad_sum, grad_i)
            
        avg_grad = sayi_carpma(grad_sum, 1.0/N)
        update = sayi_carpma(avg_grad, lr)
        w = vektor_cikarma(w, update)
        
        # Kayıtlar
        loss = (toplam_hata_karesi / 2) / N
        current_time = time.time() - start_time
        
        # HER TURDA TEST BAŞARISI ÖLÇÜMÜ
        acc = skor_hesapla(w, X_test, Y_test)
        
        loss_hist.append(loss)
        time_hist.append(current_time)
        acc_hist.append(acc)
        w_hist.append(copy.deepcopy(w))
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss:.4f} | Test Acc: {acc:.1f}% | Süre: {current_time:.2f}s")
            
    return loss_hist, time_hist, acc_hist, w_hist

# 2. SGD
def train_sgd(X, Y, X_test, Y_test, epochs, lr, w_init, deneme_no):
    w = copy.deepcopy(w_init)
    N = len(X)
    loss_hist = []
    time_hist = []
    acc_hist = []
    w_hist = []
    
    start_time = time.time()
    print(f"  [SGD - Run {deneme_no}] Başlıyor...")
    indices = list(range(N))
    
    for epoch in range(epochs):
        random.shuffle(indices)
        toplam_hata_karesi = 0.0
        
        for i in indices:
            x = X[i]; y_true = Y[i]
            z = vektor_carpimi(w, x); pred = math.tanh(z)
            err = pred - y_true; toplam_hata_karesi += err**2
            
            d_tanh = 1.0 - pred**2; grad_factor = err * d_tanh
            grad_i = sayi_carpma(x, grad_factor)
            
            update = sayi_carpma(grad_i, lr)
            w = vektor_cikarma(w, update)
            
        loss = (toplam_hata_karesi / 2) / N
        current_time = time.time() - start_time
        
        # Test Başarısı
        acc = skor_hesapla(w, X_test, Y_test)
        
        loss_hist.append(loss)
        time_hist.append(current_time)
        acc_hist.append(acc)
        w_hist.append(copy.deepcopy(w))
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss:.4f} | Test Acc: {acc:.1f}% | Süre: {current_time:.2f}s")
            
    return loss_hist, time_hist, acc_hist, w_hist

# 3. ADAM
def train_adam(X, Y, X_test, Y_test, epochs, lr, w_init, deneme_no, beta1=0.9, beta2=0.999, eps=1e-8):
    w = copy.deepcopy(w_init)
    N = len(X)
    dim = len(w)
    m = [0.0] * dim; v = [0.0] * dim
    
    loss_hist = []
    time_hist = []
    acc_hist = []
    w_hist = []
    
    start_time = time.time()
    print(f"  [Adam - Run {deneme_no}] Başlıyor...")
    
    t = 0
    for epoch in range(epochs):
        t += 1
        toplam_hata_karesi = 0.0
        grad_sum = [0.0] * dim
        
        for i in range(N):
            x = X[i]; y_true = Y[i]
            z = vektor_carpimi(w, x); pred = math.tanh(z)
            err = pred - y_true; toplam_hata_karesi += err**2
            
            d_tanh = 1.0 - pred**2; grad_factor = err * d_tanh
            grad_i = sayi_carpma(x, grad_factor)
            grad_sum = vektor_toplama(grad_sum, grad_i)
            
        g = sayi_carpma(grad_sum, 1.0/N)
        
        m_part1 = sayi_carpma(m, beta1); m_part2 = sayi_carpma(g, (1 - beta1))
        m = vektor_toplama(m_part1, m_part2)
        
        g_kare = vektor_eleman_carpimi(g, g)
        v_part1 = sayi_carpma(v, beta2); v_part2 = sayi_carpma(g_kare, (1 - beta2))
        v = vektor_toplama(v_part1, v_part2)
        
        m_hat = sayi_carpma(m, 1.0 / (1 - beta1**t))
        v_hat = sayi_carpma(v, 1.0 / (1 - beta2**t))
        
        update_term = vektor_bolme_ozel(m_hat, v_hat, eps)
        step = sayi_carpma(update_term, lr)
        w = vektor_cikarma(w, step)
        
        loss = (toplam_hata_karesi / 2) / N
        current_time = time.time() - start_time
        
        # Test Başarısı
        acc = skor_hesapla(w, X_test, Y_test)
        
        loss_hist.append(loss)
        time_hist.append(current_time)
        acc_hist.append(acc)
        w_hist.append(copy.deepcopy(w))
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss:.4f} | Test Acc: {acc:.1f}% | Süre: {current_time:.2f}s")
            
    return loss_hist, time_hist, acc_hist, w_hist

# =========================================================
# BÖLÜM 4: ANA ÇALIŞTIRMA VE GRAFİKLER
# =========================================================

X_train, Y_train = veri_yukle("egitim_vektorleri.json")
X_test, Y_test = veri_yukle("test_vektorleri.json")

if len(X_train) == 0 or len(X_test) == 0:
    exit()

input_dim = len(X_train[0])
EPOCH = 200
LR = 0.5 

sonuclar = {'GD': [], 'SGD': [], 'Adam': []}

print(f"\n{EPOCH} Epochluk 5 Farklı Deney Başlıyor...\n")

baslangic_w_listesi = []
for i in range(5):
    w_random = [random.uniform(-1.0, 1.0) for _ in range(input_dim)]
    baslangic_w_listesi.append(w_random)

for i in range(5):
    print(f"=== DENEME {i+1}/5 ===")
    w_sabit = baslangic_w_listesi[i]
    
    # Eğitim ve Test
    l_gd, t_gd, acc_gd, w_gd = train_gd(X_train, Y_train, X_test, Y_test, EPOCH, LR, w_sabit, i+1)
    sonuclar['GD'].append({'loss': l_gd, 'time': t_gd, 'acc': acc_gd, 'w': w_gd})
    print("") 
    
    l_sgd, t_sgd, acc_sgd, w_sgd = train_sgd(X_train, Y_train, X_test, Y_test, EPOCH, LR*0.05, w_sabit, i+1)
    sonuclar['SGD'].append({'loss': l_sgd, 'time': t_sgd, 'acc': acc_sgd, 'w': w_sgd})
    print("")
    
    l_adam, t_adam, acc_adam, w_adam = train_adam(X_train, Y_train, X_test, Y_test, EPOCH, 0.01, w_sabit, i+1)
    sonuclar['Adam'].append({'loss': l_adam, 'time': t_adam, 'acc': acc_adam, 'w': w_adam})
    print("-" * 40)

print("\nDeneyler Bitti! Grafikler Çiziliyor...")

algoritmalar = ['GD', 'SGD', 'Adam']
renkler = ['red', 'blue', 'green', 'orange', 'purple']

# --- GRAFİK 1: EPOCH vs. LOSS (Eğitim Hatası) ---
fig1, ax1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for idx, algo in enumerate(algoritmalar):
    for run in range(5):
        ax1[idx].plot(sonuclar[algo][run]['loss'], color=renkler[run], alpha=0.7, label=f"Run {run+1}")
    ax1[idx].set_title(f"{algo}: Epoch vs Loss")
    ax1[idx].set_xlabel("Epoch")
    if idx==0: ax1[idx].set_ylabel("Hata (Loss)")
    ax1[idx].grid(True, linestyle='--', alpha=0.5)
    ax1[idx].legend()
plt.suptitle("Karşılaştırma 1: Eğitim Hatası", fontsize=16)
plt.tight_layout()
plt.show()

# --- GRAFİK 2: SÜRE vs. LOSS ---
fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for idx, algo in enumerate(algoritmalar):
    for run in range(5):
        ax2[idx].plot(sonuclar[algo][run]['time'], sonuclar[algo][run]['loss'], color=renkler[run], alpha=0.7, label=f"Run {run+1}")
    ax2[idx].set_title(f"{algo}: Süre vs Loss")
    ax2[idx].set_xlabel("Geçen Süre (Saniye)")
    if idx==0: ax2[idx].set_ylabel("Hata (Loss)")
    ax2[idx].grid(True, linestyle='--', alpha=0.5)
    ax2[idx].legend()
plt.suptitle("Karşılaştırma 2: Zamana Göre Başarı (Hız)", fontsize=16)
plt.tight_layout()
plt.show()

# --- GRAFİK 3: EPOCH vs. TEST ACCURACY (Başarı) ---
fig3, ax3 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for idx, algo in enumerate(algoritmalar):
    for run in range(5):
        ax3[idx].plot(sonuclar[algo][run]['acc'], color=renkler[run], alpha=0.7, label=f"Run {run+1}")
    ax3[idx].set_title(f"{algo}: Epoch vs Test Başarısı")
    ax3[idx].set_xlabel("Epoch")
    if idx==0: ax3[idx].set_ylabel("Başarı (%)")
    ax3[idx].grid(True, linestyle='--', alpha=0.5)
    ax3[idx].legend()
plt.suptitle("Karşılaştırma 3: Test Başarısı", fontsize=16)
plt.tight_layout()
plt.show()

# --- GRAFİK 4: T-SNE YÖRÜNGELERİ ---
print("T-SNE hesaplanıyor...")
dev_w_listesi = []
for algo in algoritmalar:
    for run in range(5):
        dev_w_listesi.extend(sonuclar[algo][run]['w'])

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
w_2d = tsne.fit_transform(np.array(dev_w_listesi))

fig4, ax4 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
current_idx = 0 
for idx, algo in enumerate(algoritmalar):
    ax4[idx].set_title(f"{algo} Yörüngeleri (2D)")
    for run in range(5):
        uzunluk = len(sonuclar[algo][run]['w'])
        yol = w_2d[current_idx : current_idx + uzunluk]
        current_idx += uzunluk 
        ax4[idx].plot(yol[:,0], yol[:,1], marker='', linestyle='-', color=renkler[run], alpha=0.6, label=f"Run {run+1}")
        ax4[idx].plot(yol[0,0], yol[0,1], 'k^', markersize=8) 
        ax4[idx].plot(yol[-1,0], yol[-1,1], 'ro', markersize=8)
    ax4[idx].grid(True)
    if idx==2: ax4[idx].legend()

plt.suptitle("Karşılaştırma 4: Optimizasyon Yörüngeleri (T-SNE)", fontsize=16)
plt.tight_layout()
plt.show()